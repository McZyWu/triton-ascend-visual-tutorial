#!/usr/bin/env python3
"""Generate exact Triton source segments used by the Production Kernel Lab.

The generated TypeScript is pinned to one clean sgl-kernel-npu commit.  Each
logical Python statement is assigned to the visual stage it drives, and load /
store statements retain a symbolic tile-size formula derived from the
``tl.arange`` values that feed their offsets.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import textwrap
import tokenize
from pathlib import Path


PHASE_EXPLANATIONS = {
    "program": "确定当前 program、persistent 轮次或循环正在处理的逻辑 task。",
    "offset": "构造 tile 内 lane、全局坐标、展平 offset 与越界 mask。",
    "load": "按当前 offset 和 mask 从 GM 读取变量 tile，值进入 UB / 寄存器。",
    "compute": "在片上对已经加载的 tile 执行标量、Vector、Cube、归约或重排计算。",
    "store": "按输出 offset 和 mask 把结果或更新后的状态写回 GM。",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def is_triton_jit(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        name = ast.unparse(decorator)
        if (
            name == "triton.jit"
            or name.startswith("triton.jit(")
            or name.endswith(".jit")
            or name == "jit"
        ):
            return True
    return False


def logical_statements(
    source: str, node: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[tuple[int, int, str]]:
    lines = source.splitlines()
    body_start = node.body[0].lineno if node.body else node.lineno + 1
    doc_range: tuple[int, int] | None = None
    if node.body and isinstance(node.body[0], ast.Expr):
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            doc_range = (node.body[0].lineno, node.body[0].end_lineno or node.body[0].lineno)

    starts: list[tuple[int, int]] = []
    current_start: int | None = None
    reader = io.StringIO(source).readline
    for token in tokenize.generate_tokens(reader):
        # The function signature can span many physical lines.  It belongs to
        # the signature card, not to the executable source-to-visual timeline.
        if token.start[0] < body_start or token.start[0] > (node.end_lineno or node.lineno):
            continue
        if token.type in {
            tokenize.ENCODING,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NL,
            tokenize.COMMENT,
        }:
            continue
        if current_start is None and token.type not in {tokenize.NEWLINE, tokenize.ENDMARKER}:
            current_start = token.start[0]
        if token.type == tokenize.NEWLINE and current_start is not None:
            starts.append((current_start, max(current_start, token.end[0])))
            current_start = None

    result: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for start, end in starts:
        if (start, end) in seen:
            continue
        seen.add((start, end))
        if doc_range and start >= doc_range[0] and end <= doc_range[1]:
            continue
        code = textwrap.dedent("\n".join(lines[start - 1 : end])).rstrip()
        if code and not code.lstrip().startswith("#"):
            result.append((start, end, code))
    return result


def classify(code: str) -> str:
    flat = " ".join(code.lower().split())
    if re.search(r"tl\.(store|atomic_|store_tensor_descriptor|scatter_ub_to_out)", flat):
        return "store"
    if re.search(r"tl\.(load|load_tensor_descriptor|gather_out_to_ub)", flat):
        return "load"
    if (
        "tl.program_id" in flat
        or "tl.num_programs" in flat
        or re.search(r"\b(total_tasks|task_id|num_tasks|row_begin|row_end)\b", flat)
        or re.match(r"(for|while)\b", flat)
    ):
        return "program"
    if (
        "tl.arange" in flat
        or "make_block_ptr" in flat
        or re.search(
            r"\b(mask|offsets?|offs|idx|indices|base|page_id|head_offs|cols?|rows?)\b",
            flat,
        )
    ):
        return "offset"
    return "compute"


def identifiers(code: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_]\w*\b", code))


def assigned_name(code: str) -> str | None:
    match = re.match(r"\s*([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)", code)
    return match.group(1) if match else None


def arange_and_assignments(statements: list[tuple[int, int, str]]):
    aranges: dict[str, str] = {}
    assignments: dict[str, str] = {}
    for _, _, code in statements:
        name = assigned_name(code)
        if name:
            assignments[name] = code.split("=", 1)[1]
        match = re.search(
            r"^\s*([A-Za-z_]\w*)\s*=\s*tl\.arange\(\s*([^,]+),\s*([^)]+)\)",
            code,
            re.S,
        )
        if match:
            start, end = match.group(2).strip(), match.group(3).strip()
            aranges[match.group(1)] = end if start == "0" else f"({end}) - ({start})"
        elif name:
            # Address helpers often embed tl.arange directly, for example
            # ``ptr = base + tl.arange(0, BLOCK)``.  Propagate that extent to
            # later tl.load/tl.store statements through the assigned name.
            inline_extents = []
            for start, end in re.findall(r"tl\.arange\(\s*([^,]+),\s*([^)]+)\)", code, re.S):
                extent = end.strip() if start.strip() == "0" else f"({end.strip()}) - ({start.strip()})"
                if extent not in inline_extents:
                    inline_extents.append(extent)
            if inline_extents:
                aranges[name] = " × ".join(inline_extents)
    return aranges, assignments


def tile_formula(code: str, aranges: dict[str, str], assignments: dict[str, str]) -> str:
    frontier = list(identifiers(code))
    visited: set[str] = set()
    extents: list[str] = []
    while frontier:
        name = frontier.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in aranges and aranges[name] not in extents:
            extents.append(aranges[name])
        rhs = assignments.get(name)
        if rhs:
            frontier.extend(identifiers(rhs) - visited)
    return " × ".join(extents) if extents else "1"


def segment_explanation(phase: str, variables: list[str], result: str | None) -> str:
    names = "、".join(variables[:4])
    suffix = f" 关联变量：{names}。" if names else ""
    if result and phase in {"load", "compute"}:
        suffix += f" 产生片上值 {result}。"
    return PHASE_EXPLANATIONS[phase] + suffix


def generate(source_root: Path) -> tuple[dict, dict]:
    repo = source_root
    while repo != repo.parent and not (repo / ".git").exists():
        repo = repo.parent
    if not (repo / ".git").exists():
        raise SystemExit(f"cannot find git repository above {source_root}")

    commit = git(repo, "rev-parse", "HEAD")
    metadata = {
        "commit": commit,
        "shortCommit": commit[:7],
        "commitDate": git(repo, "show", "-s", "--format=%cI", "HEAD"),
        "subject": git(repo, "show", "-s", "--format=%s", "HEAD"),
        "repository": "https://github.com/sgl-project/sgl-kernel-npu",
    }

    kernels: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(source_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            raise SystemExit(f"cannot parse {path}: {error}") from error
        module = path.relative_to(source_root).as_posix()
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not is_triton_jit(node):
                continue
            statements = logical_statements(source, node)
            aranges, assignments = arange_and_assignments(statements)
            args = [arg.arg for arg in node.args.args]
            segments: list[dict] = []
            for index, (start, end, code) in enumerate(statements):
                phase = classify(code)
                used_args = [name for name in args if re.search(rf"\b{re.escape(name)}\b", code)]
                result = assigned_name(code)
                segments.append(
                    {
                        "id": f"{node.name}-{index + 1}",
                        "phase": phase,
                        "lineStart": start,
                        "lineEnd": end,
                        "code": code,
                        "variables": used_args,
                        "result": result,
                        "tileFormula": tile_formula(code, aranges, assignments)
                        if phase in {"load", "compute", "store"}
                        else None,
                        "explanation": segment_explanation(phase, used_args, result),
                    }
                )
            kernels.setdefault(module, {})[node.name] = segments
    return metadata, kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata, kernels = generate(args.source_root.resolve())
    kernel_count = sum(len(items) for items in kernels.values())
    segment_count = sum(len(segments) for items in kernels.values() for segments in items.values())
    module_dir = args.output.parent / "kernel-source-modules"
    module_dir.mkdir(parents=True, exist_ok=True)
    loader_lines: list[str] = []
    for module, module_kernels in kernels.items():
        stem = re.sub(r"[^A-Za-z0-9]+", "_", module).strip("_")
        module_output = module_dir / f"{stem}.ts"
        module_output.write_text(
            "// Generated source segments for " + module + "; do not edit.\n"
            "import type { KernelSourceSegment } from \"../kernel-source-segments\";\n"
            "const segments: Record<string, KernelSourceSegment[]> = "
            + json.dumps(module_kernels, ensure_ascii=False, separators=(",", ":"))
            + ";\nexport default segments;\n",
            encoding="utf-8",
        )
        loader_lines.append(
            f"  {json.dumps(module, ensure_ascii=False)}: () => "
            f'import("./kernel-source-modules/{stem}").then(module => module.default),'
        )
    output = (
        "// Generated by scripts/generate-kernel-source-segments.py; do not edit by hand.\n"
        "export type SourcePhase = \"program\" | \"offset\" | \"load\" | \"compute\" | \"store\";\n"
        "export type KernelSourceSegment = { id: string; phase: SourcePhase; lineStart: number; lineEnd: number; code: string; variables: string[]; result: string | null; tileFormula: string | null; explanation: string };\n"
        f"export const VISUAL_SOURCE = {json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))} as const;\n"
        "export const KERNEL_SOURCE_LOADERS: Record<string, () => Promise<Record<string, KernelSourceSegment[]>>> = {\n"
        + "\n".join(loader_lines)
        + "\n};\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(
        f"generated {kernel_count} Triton kernels / {segment_count} source segments "
        f"at {metadata['shortCommit']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
