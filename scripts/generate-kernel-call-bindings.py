"""Extract Triton launch-site argument bindings from sgl-kernel-npu source."""

from __future__ import annotations

import ast
import json
import pathlib
import sys


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def shape_index(node: ast.AST) -> tuple[str, int] | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Attribute):
        return None
    if node.value.attr != "shape" or not isinstance(node.value.value, ast.Name):
        return None
    index = node.slice
    if isinstance(index, ast.Constant) and isinstance(index.value, int):
        return node.value.value.id, index.value
    if isinstance(index, ast.UnaryOp) and isinstance(index.op, ast.USub) and isinstance(index.operand, ast.Constant):
        return node.value.value.id, -int(index.operand.value)
    return None


def infer_shapes(wrapper: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, string]:
    dims: dict[str, dict[int, str]] = {}
    direct: dict[str, str] = {}
    for node in ast.walk(wrapper):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        pairs: list[tuple[ast.AST, ast.AST]] = []
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)):
            pairs.extend(zip(target.elts, node.value.elts))
        else:
            pairs.append((target, node.value))
        for left, right in pairs:
            if isinstance(left, ast.Name):
                indexed = shape_index(right)
                if indexed:
                    tensor, axis = indexed
                    dims.setdefault(tensor, {})[axis] = left.id
            if not isinstance(left, ast.Name) or not isinstance(right, ast.Call):
                continue
            call_name = dotted_name(right.func) or ""
            if call_name.endswith((".empty_like", ".zeros_like", ".ones_like")) and right.args:
                direct[left.id] = f"same as {ast.unparse(right.args[0])}"
            elif call_name.endswith((".empty", ".zeros", ".ones", ".full")) and right.args:
                shape_args = right.args
                if len(shape_args) == 1 and isinstance(shape_args[0], (ast.Tuple, ast.List)):
                    shape_args = shape_args[0].elts
                direct[left.id] = "[" + ", ".join(ast.unparse(arg) for arg in shape_args) + "]"
            elif isinstance(right.func, ast.Attribute) and right.func.attr in {"view", "reshape", "unsqueeze"}:
                direct[left.id] = "[" + ", ".join(ast.unparse(arg) for arg in right.args) + "]"

    for tensor, axes in dims.items():
        positive = [axis for axis in axes if axis >= 0]
        negative = [axis for axis in axes if axis < 0]
        if positive:
            last = max(positive)
            parts = [axes.get(axis, "…") for axis in range(last + 1)]
            if negative:
                parts.extend(axes[axis] for axis in sorted(negative))
        else:
            parts = ["…", *(axes[axis] for axis in sorted(negative))]
        direct.setdefault(tensor, "[" + ", ".join(parts) + "]")
    return direct


def inferred_shape(expression: str, shapes: dict[str, str]) -> str | None:
    if expression in shapes:
        return shapes[expression]
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Call):
        call_name = dotted_name(node.func) or ""
        if call_name.endswith((".contiguous", ".clone", ".to")) and isinstance(node.func, ast.Attribute):
            return inferred_shape(ast.unparse(node.func.value), shapes)
        if call_name.endswith((".view", ".reshape")):
            return "[" + ", ".join(ast.unparse(arg) for arg in node.args) + "]"
    if isinstance(node, ast.Subscript):
        base = dotted_name(node.value)
        return f"selection from {base}" if base else None
    return None


def main() -> None:
    source_root = pathlib.Path(sys.argv[1]).resolve()
    output_path = pathlib.Path(sys.argv[2]).resolve()
    signature_path = output_path.parent / "kernel-signatures.ts"
    source = signature_path.read_text(encoding="utf-8")
    marker = "KERNEL_SIGNATURES: Record<string, Record<string, KernelSignature>> = "
    signatures = json.loads(source[source.index(marker) + len(marker) : source.rindex(";")])
    result: dict[str, dict[str, dict[str, object]]] = {}

    for module, kernels in signatures.items():
        path = source_root / module
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        candidates: dict[str, list[tuple[int, dict[str, object]]]] = {name: [] for name in kernels}
        stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

        class LaunchVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                stack.append(node)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                target = node.func
                if isinstance(target, ast.Subscript):
                    launch_name = dotted_name(target.value)
                    kernel = launch_name.rsplit(".", 1)[-1] if launch_name else None
                    if kernel in kernels and stack:
                        signature_args = kernels[kernel]["args"]
                        bindings = {
                            arg: ast.unparse(value)
                            for arg, value in zip(signature_args, node.args)
                        }
                        for keyword in node.keywords:
                            if keyword.arg in signature_args:
                                bindings[keyword.arg] = ast.unparse(keyword.value)
                        wrapper = stack[-1]
                        score = len(bindings) + (100 if wrapper.name != kernel else 0)
                        shapes = infer_shapes(wrapper)
                        bound_shapes = {
                            arg: shape
                            for arg, expression in bindings.items()
                            if (shape := inferred_shape(expression, shapes)) is not None
                        }
                        candidates[kernel].append(
                            (
                                score,
                                {
                                    "wrapper": wrapper.name,
                                    "line": node.lineno,
                                    "args": bindings,
                                    "shapes": bound_shapes,
                                },
                            )
                        )
                self.generic_visit(node)

        LaunchVisitor().visit(tree)
        chosen = {
            kernel: max(items, key=lambda item: item[0])[1]
            for kernel, items in candidates.items()
            if items
        }
        if chosen:
            result[module] = chosen

    output = (
        "export type KernelCallBinding = { wrapper: string; line: number; args: Record<string, string>; shapes: Record<string, string> };\n"
        "export const KERNEL_CALL_BINDINGS: Record<string, Record<string, KernelCallBinding>> = "
        + json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )
    output_path.write_text(output, encoding="utf-8")
    mapped = sum(len(module) for module in result.values())
    print(f"generated {mapped} kernel launch bindings in {output_path}")


if __name__ == "__main__":
    main()
