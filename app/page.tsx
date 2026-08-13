"use client";

import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";

const SOURCE = {
  quick: "https://github.com/triton-lang/triton-ascend/blob/main/docs/en/quick_start.md",
  docs: "https://triton-ascend.readthedocs.io/en/latest/index.html",
  sgl: "https://github.com/sgl-project/sgl-kernel-npu",
  rms: "https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu/norm/rmsnorm_without_weight.py",
  swiglu: "https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu/activation/swiglu_oai.py",
  softmax: "https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py",
  cache: "https://github.com/sgl-project/sgl-kernel-npu/blob/main/tests/python/sgl_kernel_npu/test_cache_assign.py",
};

const NAV = [
  ["start", "01", "起步"],
  ["anatomy", "02", "拆解 Kernel"],
  ["transfer", "03", "搬运实验室"],
  ["parallel", "04", "Grid 并行"],
  ["ub", "05", "UB 估算"],
  ["cases", "06", "真实算子"],
  ["api", "07", "接口对照"],
  ["debug", "08", "验证与排障"],
];

const ADD_KERNEL = `@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)`;

const VARIABLE_ROWS = [
  ["x_ptr / y_ptr", "输入张量首地址", "不是数据本身；+ offsets 后才得到每个元素的地址"],
  ["output_ptr", "输出张量首地址", "store 的目标，全局内存中的 C"],
  ["n_elements", "真实元素总数 N", "最后一个 program 用它生成 mask，避免越界"],
  ["BLOCK_SIZE", "每个 program 处理的槽位数", "编译期常量；影响 grid、并行粒度与 UB 占用"],
  ["pid", "当前 program 的一维编号", "由 tl.program_id(0) 取得，范围 0…grid-1"],
  ["block_start", "本 program 的全局起点", "pid × BLOCK_SIZE"],
  ["offsets", "一组全局元素下标", "block_start + [0, 1, …, BLOCK_SIZE-1]"],
  ["mask", "逐槽位有效标记", "offset < N 为真才 load/store；尾块会出现 false"],
  ["x / y", "搬进片上后参与计算的向量", "概念上驻留 UB/寄存器；编译器决定最终分配"],
  ["grid", "要启动的 program 数", "ceil(N / BLOCK_SIZE)，即 triton.cdiv(N, BLOCK_SIZE)"],
];

const API_ROWS = [
  ["tl.program_id(0)", "当前 program 编号", "无直接逐元素 Torch 等价；像外层 chunk 循环的 chunk_id"],
  ["tl.num_programs(0)", "本轴 program 总数", "grid[0]；可用于 pid, pid+grid… 的持久化调度"],
  ["tl.arange(0, B)", "生成 B 个 lane 下标", "torch.arange(B, device='npu')"],
  ["tl.load(p + o, mask, other)", "按地址向量加载", "torch.where(mask, flat[o], other)（概念等价）"],
  ["tl.store(p + o, v, mask)", "按地址向量写回", "flat[o[mask]] = v[mask]"],
  ["tl.sum(x, axis)", "片上归约求和", "torch.sum(x, dim=...)"],
  ["tl.max / tl.argmax", "最大值 / 最大值位置", "torch.max / torch.argmax"],
  ["tl.exp / tl.rsqrt", "逐元素指数 / 倒平方根", "torch.exp / torch.rsqrt"],
  ["tl.where(c, a, b)", "逐 lane 选择", "torch.where(c, a, b)"],
  ["tl.cdiv(a, b)", "向上整除", "(a + b - 1) // b"],
  ["tl.next_power_of_2(n)", "不小于 n 的最小 2 次幂", "1 << (n - 1).bit_length()"],
  ["x.stride(0)", "相邻行首地址距离", "PyTorch 同名；offset = row × stride_b + col"],
  ["@triton.jit", "将函数编译成设备 kernel", "torch.compile 是图级编译，语义层次不同"],
  ["kernel[grid](...)", "按 grid 启动 kernel", "普通 Torch 运算隐藏了设备 launch 细节"],
];

const CASES = [
  {
    id: "rms",
    no: "CASE 01",
    name: "RMSNorm · 行归约",
    source: SOURCE.rms,
    input: "x: [B, L, C]，eps: 标量",
    output: "y: [B, L, C]，与 x 同形",
    torch: "F.rms_norm(x, normalized_shape=(C,), eps=eps)",
    formula: "var = mean(x²)  →  rstd = rsqrt(var + eps)  →  y = x × rstd",
    grid: "grid = (num_vectorcore,)，program 以 pid + k×grid 跨步领取 token 块",
    vars: "B 批次；L token 数；C hidden_size；block_l 每次处理 token 数；kernel_num 向量核数",
  },
  {
    id: "swiglu",
    no: "CASE 02",
    name: "SwiGLU · 交错门控",
    source: SOURCE.swiglu,
    input: "hidden_states: [BS, 2D]，偶数列 gate，奇数列 up",
    output: "gated_output: [BS, D]",
    torch: "gate * sigmoid(gate * alpha) * (up + 1)，含上下限 clamp",
    formula: "split even/odd → clamp → sigmoid(gate×α) → gate×sigmoid → ×(up+1)",
    grid: "按 batch 行分块；每个 program 再用 MINIBLOCK_SIZE=16 循环",
    vars: "dim=2D；output_dim=D；BS 行数；alpha 门控斜率；limit 截断阈值",
  },
  {
    id: "softmax",
    no: "CASE 03",
    name: "Argmax + Softmax · 在线融合",
    source: SOURCE.softmax,
    input: "logits: [B, V]，最后一维连续",
    output: "argmax_id: [B] int64；最大 token 概率: [B] fp32",
    torch: "argmax = logits.float().argmax(-1); softmax(...).gather(...) ",
    formula: "逐 tile 更新 running max m、arg、sumexp s；最终最大项概率 = 1/s",
    grid: "grid = (min(num_cores, B),)，每个 program 跨步处理多行",
    vars: "B 行数；V 词表；stride_b 行跨度；BLOCK_V 词表 tile；m/s 在线状态",
  },
  {
    id: "cache",
    no: "CASE 04",
    name: "KV Cache · 位置写入",
    source: SOURCE.cache,
    input: "请求行号、每行 [start,end)、一维 out_cache_loc",
    output: "原地更新 req_to_token[request, token_position]",
    torch: "repeat_interleave 构造 row/col/src indices，再高级索引写入",
    formula: "lengths → exclusive cumsum → 展开行号 → 局部 offset → 写入二维池",
    grid: "不同请求 / token 段可以并行；地址必须由各行的独占前缀和还原",
    vars: "bs 请求数；lengths 每请求写入数；cumsum 每段源起点；out_cache_loc 新槽位",
  },
];

function CodeBlock({ children, label = "python" }: { children: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard?.writeText(children);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };
  return (
    <div className="code-block">
      <div className="code-head"><span>{label}</span><button onClick={copy}>{copied ? "已复制" : "复制"}</button></div>
      <pre><code>{children}</code></pre>
    </div>
  );
}

function TransferLab() {
  const [dims, setDims] = useState<1 | 2 | 3>(2);
  const [shape, setShape] = useState<[number, number, number]>([3, 7, 10]);
  const [block, setBlock] = useState<[number, number, number]>([2, 3, 4]);
  const [pid, setPid] = useState<[number, number, number]>([0, 1, 1]);
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const activeShape: [number, number, number] = [dims === 3 ? shape[0] : 1, dims >= 2 ? shape[1] : 1, shape[2]];
  const activeBlock: [number, number, number] = [dims === 3 ? block[0] : 1, dims >= 2 ? block[1] : 1, block[2]];
  const grid: [number, number, number] = activeShape.map((size, axis) => Math.ceil(size / activeBlock[axis])) as [number, number, number];
  const safePid: [number, number, number] = pid.map((value, axis) => Math.min(value, Math.max(0, grid[axis] - 1))) as [number, number, number];
  const starts: [number, number, number] = safePid.map((value, axis) => value * activeBlock[axis]) as [number, number, number];
  const lanes = useMemo(() => {
    const result: Array<{ local: [number, number, number]; global: [number, number, number]; offset: number; valid: boolean; x: number; y: number; sum: number }> = [];
    for (let lz = 0; lz < activeBlock[0]; lz++) {
      for (let ly = 0; ly < activeBlock[1]; ly++) {
        for (let lx = 0; lx < activeBlock[2]; lx++) {
          const global: [number, number, number] = [starts[0] + lz, starts[1] + ly, starts[2] + lx];
          const valid = global[0] < activeShape[0] && global[1] < activeShape[1] && global[2] < activeShape[2];
          const offset = (global[0] * activeShape[1] + global[1]) * activeShape[2] + global[2];
          const x = offset + 1;
          const y = (offset + 1) * 10;
          result.push({ local: [lz, ly, lx], global, offset, valid, x, y, sum: x + y });
        }
      }
    }
    return result;
  }, [activeBlock[0], activeBlock[1], activeBlock[2], activeShape[0], activeShape[1], activeShape[2], starts[0], starts[1], starts[2]]);
  const phases = ["选择 grid 中的 program", "生成多维坐标、展平 offset 与 mask", "X、Y 分别从 GM 搬入 UB", "UB 中逐 lane 计算 X + Y", "把结果 C 从 UB 写回 GM"];
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setPhase((p) => {
      if (p === 4) { setPlaying(false); return 4; }
      return p + 1;
    }), 900);
    return () => window.clearInterval(timer);
  }, [playing]);
  useEffect(() => {
    setPid((current) => current.map((value, axis) => Math.min(value, Math.max(0, grid[axis] - 1))) as [number, number, number]);
    setPhase(0);
  }, [dims, grid[0], grid[1], grid[2]]);

  const setAxis = (setter: typeof setShape | typeof setBlock | typeof setPid, values: [number, number, number], axis: number, value: number) => {
    const next = [...values] as [number, number, number];
    next[axis] = value;
    setter(next);
    setPlaying(false);
    setPhase(0);
  };
  const coordText = (coord: [number, number, number]) => dims === 1 ? `[${coord[2]}]` : dims === 2 ? `[${coord[1]},${coord[2]}]` : `[${coord.join(",")}]`;
  const programText = (coord: [number, number, number]) => dims === 1 ? `(${coord[2]})` : dims === 2 ? `(${coord[2]},${coord[1]})` : `(${coord[2]},${coord[1]},${coord[0]})`;
  const blockTuple = dims === 1 ? `(${activeBlock[2]})` : dims === 2 ? `(${activeBlock[1]}, ${activeBlock[2]})` : `(${activeBlock.join(", ")})`;
  const gridTuple = dims === 1 ? `(${grid[2]},)` : dims === 2 ? `(${grid[2]}, ${grid[1]})` : `(${grid[2]}, ${grid[1]}, ${grid[0]})`;
  const pidTuple = dims === 1 ? `(${safePid[2]})` : dims === 2 ? `(${safePid[2]}, ${safePid[1]})` : `(${safePid[2]}, ${safePid[1]}, ${safePid[0]})`;
  const offsetFormula = dims === 1
    ? `x = pidₓ×Bₓ+laneₓ；offset = x`
    : dims === 2
      ? `y = pidᵧ×Bᵧ+laneᵧ；x = pidₓ×Bₓ+laneₓ；offset = y×W+x`
      : `z/y/x = pid×BLOCK+lane；offset = (z×H+y)×W+x`;
  const exampleLane = lanes[0];
  const globalDerivation = dims === 1
    ? `x = pidₓ(${safePid[2]}) × Bₓ(${activeBlock[2]}) + laneₓ(${exampleLane.local[2]}) = ${exampleLane.global[2]}`
    : dims === 2
      ? `y = ${safePid[1]}×${activeBlock[1]}+${exampleLane.local[1]} = ${exampleLane.global[1]}；x = ${safePid[2]}×${activeBlock[2]}+${exampleLane.local[2]} = ${exampleLane.global[2]}`
      : `z = ${safePid[0]}×${activeBlock[0]}+${exampleLane.local[0]} = ${exampleLane.global[0]}；y = ${safePid[1]}×${activeBlock[1]}+${exampleLane.local[1]} = ${exampleLane.global[1]}；x = ${safePid[2]}×${activeBlock[2]}+${exampleLane.local[2]} = ${exampleLane.global[2]}`;
  const offsetDerivation = dims === 1
    ? `offset = x = ${exampleLane.offset}`
    : dims === 2
      ? `offset = y×W+x = ${exampleLane.global[1]}×${activeShape[2]}+${exampleLane.global[2]} = ${exampleLane.offset}`
      : `offset = (z×H+y)×W+x = (${exampleLane.global[0]}×${activeShape[1]}+${exampleLane.global[1]})×${activeShape[2]}+${exampleLane.global[2]} = ${exampleLane.offset}`;
  const maskDerivation = dims === 1
    ? `${exampleLane.global[2]} < N(${activeShape[2]}) → ${exampleLane.valid ? "TRUE，可读取/写入" : "FALSE，禁止访问"}`
    : dims === 2
      ? `y(${exampleLane.global[1]}) < H(${activeShape[1]}) 且 x(${exampleLane.global[2]}) < W(${activeShape[2]}) → ${exampleLane.valid ? "TRUE，可读取/写入" : "FALSE，禁止访问"}`
      : `z(${exampleLane.global[0]}) < D(${activeShape[0]}) 且 y(${exampleLane.global[1]}) < H(${activeShape[1]}) 且 x(${exampleLane.global[2]}) < W(${activeShape[2]}) → ${exampleLane.valid ? "TRUE，可读取/写入" : "FALSE，禁止访问"}`;

  const programPlanes = Array.from({ length: grid[0] }, (_, pz) => (
    <div className="program-plane" key={pz}>
      {dims === 3 && <small>grid z = {pz}</small>}
      <div className="program-grid" style={{ gridTemplateColumns: `repeat(${grid[2]}, minmax(40px, 1fr))` }}>
        {Array.from({ length: grid[1] }, (_, py) => Array.from({ length: grid[2] }, (_, px) => {
          const selected = safePid[0] === pz && safePid[1] === py && safePid[2] === px;
          const start = [pz * activeBlock[0], py * activeBlock[1], px * activeBlock[2]] as [number, number, number];
          return <button key={`${pz}-${py}-${px}`} className={selected ? "selected" : ""} onClick={() => { setPid([pz, py, px]); setPhase(0); setPlaying(false); }} aria-pressed={selected}>
            <b>pid {programText([pz, py, px])}</b><span>start {coordText(start)}</span>
          </button>;
        }))}
      </div>
    </div>
  ));

  const valueLayer = (name: string, stage: string, getter: (lane: typeof lanes[number]) => string | number, visible: boolean, accent: string) => (
    <div className={`value-layer ${visible ? "visible" : "waiting"}`}>
      <div className="value-layer-head"><b>{name}</b><span>{stage}</span></div>
      <div className="tile-slices">
        {Array.from({ length: activeBlock[0] }, (_, lz) => <div className="tile-slice" key={lz}>
          {dims === 3 && <small>local z={lz} → global z={starts[0] + lz}</small>}
          <div className="value-grid" style={{ gridTemplateColumns: `repeat(${activeBlock[2]}, minmax(44px, 1fr))`, "--layer-accent": accent } as CSSProperties}>
            {lanes.filter((lane) => lane.local[0] === lz).map((lane) => <div key={`${name}-${lane.local.join("-")}`} className={lane.valid ? "valid" : "masked"}>
              <i>{lane.valid && visible ? getter(lane) : lane.valid ? "·" : "MASK"}</i>
              <span>{coordText(lane.global)}</span>
            </div>)}
          </div>
        </div>)}
      </div>
    </div>
  );

  return (
    <div className="lab transfer-lab">
      <div className="dimension-tabs" aria-label="选择张量与 grid 维度">
        <span>DIMENSION</span>
        {([1, 2, 3] as const).map((value) => <button key={value} className={dims === value ? "active" : ""} onClick={() => { setDims(value); setPhase(0); setPlaying(false); }}>{value}D</button>)}
        <p>{dims === 1 ? "向量：一个 program 搬一段连续元素" : dims === 2 ? "矩阵：program (pidᵧ,pidₓ) 搬一个二维 tile" : "体张量：program (pid_z,pidᵧ,pidₓ) 搬一个三维 block"}</p>
      </div>
      <div className="lab-toolbar">
        <div className="control-group"><span>SHAPE</span>
          {dims === 3 && <label>D/Z <input type="range" min="2" max="4" value={shape[0]} onChange={(e) => setAxis(setShape, shape, 0, +e.target.value)} /><b>{shape[0]}</b></label>}
          {dims >= 2 && <label>H/Y <input type="range" min="4" max="9" value={shape[1]} onChange={(e) => setAxis(setShape, shape, 1, +e.target.value)} /><b>{shape[1]}</b></label>}
          <label>{dims === 1 ? "N/X" : "W/X"} <input type="range" min={dims === 1 ? 9 : 5} max={dims === 1 ? 32 : 12} value={shape[2]} onChange={(e) => setAxis(setShape, shape, 2, +e.target.value)} /><b>{shape[2]}</b></label>
        </div>
        <div className="control-group"><span>BLOCK_SIZE</span>
          {dims === 3 && <label>Bz <select value={block[0]} onChange={(e) => setAxis(setBlock, block, 0, +e.target.value)}><option>1</option><option>2</option></select></label>}
          {dims >= 2 && <label>By <select value={block[1]} onChange={(e) => setAxis(setBlock, block, 1, +e.target.value)}><option>2</option><option>3</option><option>4</option></select></label>}
          <label>Bx <select value={block[2]} onChange={(e) => setAxis(setBlock, block, 2, +e.target.value)}>{(dims === 1 ? [4, 8, 16] : [2, 3, 4]).map((value) => <option key={value}>{value}</option>)}</select></label>
        </div>
        <div className="control-group"><span>PROGRAM_ID</span>
          {dims === 3 && <label>pid_z <input type="range" min="0" max={grid[0] - 1} value={safePid[0]} onChange={(e) => setAxis(setPid, safePid, 0, +e.target.value)} /><b>{safePid[0]}</b></label>}
          {dims >= 2 && <label>pid_y <input type="range" min="0" max={grid[1] - 1} value={safePid[1]} onChange={(e) => setAxis(setPid, safePid, 1, +e.target.value)} /><b>{safePid[1]}</b></label>}
          <label>pid_x <input type="range" min="0" max={grid[2] - 1} value={safePid[2]} onChange={(e) => setAxis(setPid, safePid, 2, +e.target.value)} /><b>{safePid[2]}</b></label>
        </div>
        <div className="transport-actions">
        <button className="action" onClick={() => { setPhase(0); setPlaying(true); }}>{playing ? "搬运中…" : "▶ 播放搬运"}</button>
        <button onClick={() => { setPlaying(false); setPhase((phase + 1) % 5); }}>单步 →</button>
        </div>
      </div>
      <div className="equation-strip">
        <span>shape = <strong>{dims === 1 ? `[${shape[2]}]` : dims === 2 ? `[${shape[1]}, ${shape[2]}]` : `[${shape.join(", ")}]`}</strong></span>
        <span>BLOCK = <strong>{blockTuple}</strong></span>
        <span>grid = cdiv(shape, block) = <strong>{gridTuple}</strong></span>
        <span>pid = <strong>{pidTuple}</strong></span>
        <span>block_start = <strong>{coordText(starts)}</strong></span>
        <span>有效 lanes = <strong>{lanes.filter((lane) => lane.valid).length}/{lanes.length}</strong></span>
      </div>
      <div className="multidim-formula">
        <b>地址展开</b><code>{offsetFormula}</code><span>grid tuple 按 Triton 轴写作 x→axis 0、y→axis 1、z→axis 2；内存仍按最后一维 X 连续。</span>
      </div>
      <div className={`program-selector ${phase === 0 ? "active" : ""}`}>
        <div className="zone-title"><span>GRID / PROGRAM MAP</span><small>点击任一 program，观察它负责的 block</small></div>
        {programPlanes}
      </div>
      <div className="coordinate-ledger">
        <div className="zone-title"><span>块内 LANE → 张量坐标 → 内存 OFFSET</span><small>每一行代表当前 program 同时处理的一个数据槽位</small></div>
        <div className="coordinate-explainer">
          <div className="coordinate-reading">
            <span>这一行应该读成</span>
            <b>{coordText(exampleLane.local)}</b><i>块内第几个槽位</i><em>→</em>
            <b>{coordText(exampleLane.global)}</b><i>整个张量中的坐标</i><em>→</em>
            <b>{exampleLane.offset}</b><i>从首地址数第几个元素</i><em>→</em>
            <b className={exampleLane.valid ? "ok" : "no"}>{exampleLane.valid ? "TRUE" : "FALSE"}</b><i>{exampleLane.valid ? "允许 load / store" : "必须被 mask"}</i>
          </div>
          <div className="coordinate-derivation">
            <p><b>① 块内坐标</b><code>{coordText(exampleLane.local)}</code><span>由 <code>tl.arange</code> 生成，只表示它在当前 block 内的位置。</span></p>
            <p><b>② 张量坐标</b><code>{coordText(exampleLane.global)}</code><span>{globalDerivation}</span></p>
            <p><b>③ 展平 offset</b><code>{exampleLane.offset}</code><span>{offsetDerivation}；所以访问 <code>x_ptr + {exampleLane.offset}</code>。</span></p>
            <p><b>④ mask</b><code>{exampleLane.valid ? "TRUE" : "FALSE"}</code><span>{maskDerivation}</span></p>
          </div>
          <p className="coordinate-note"><b>注意：</b><code>offset</code> 是“元素下标”，不是字节数。若 dtype 是 fp32，第 {exampleLane.offset} 个元素的字节地址才是 <code>base + {exampleLane.offset} × 4</code>。</p>
        </div>
        <div className="coordinate-table"><div><b>① block 内坐标</b><b>② 张量全局坐标</b><b>③ 首地址 + 元素下标</b><b>④ 可否访问</b></div>{lanes.map((lane) => <div key={`coord-${lane.local.join("-")}`} className={lane.valid ? "" : "masked"}><code>{coordText(lane.local)}</code><code>{coordText(lane.global)}</code><code>base + {lane.offset}</code><strong>{lane.valid ? "TRUE · 搬运" : "FALSE · 跳过"}</strong></div>)}</div>
      </div>
      <div className="memory-stage-v2" aria-label={`${dims}维 X 和 Y 从全局内存搬到统一缓冲区、相加并写回 C 的模拟`}>
        <div className={`memory-column gm ${phase === 2 ? "active" : ""}`}>
          <div className="zone-title"><span>GM / SOURCE</span><small>两个独立输入张量</small></div>
          {valueLayer("X", `global ${dims}D tensor`, (lane) => lane.x, true, "var(--cyan)")}
          {valueLayer("Y", `global ${dims}D tensor`, (lane) => lane.y, true, "var(--orange)")}
        </div>
        <div className={`bus-v2 ${phase === 2 ? "active load" : ""}`}><span>LOAD X</span><span>LOAD Y</span><i /></div>
        <div className={`memory-column ub ${phase === 2 || phase === 3 ? "active" : ""}`}>
          <div className="zone-title"><span>UB / CURRENT BLOCK</span><small>{blockTuple} lanes</small></div>
          {valueLayer("X_tile", "tl.load(x_ptr + offsets)", (lane) => lane.x, phase >= 2, "var(--cyan)")}
          {valueLayer("Y_tile", "tl.load(y_ptr + offsets)", (lane) => lane.y, phase >= 2, "var(--orange)")}
          <div className="alu-sign"><span>X_tile</span><b>+</b><span>Y_tile</span><i>逐 lane 并行</i></div>
          {valueLayer("C_tile = X + Y", "UB compute result", (lane) => lane.sum, phase >= 3, "var(--acid)")}
        </div>
        <div className={`bus-v2 store ${phase === 4 ? "active" : ""}`}><span>STORE C</span><i /></div>
        <div className={`memory-column output ${phase === 4 ? "active" : ""}`}>
          <div className="zone-title"><span>GM / OUTPUT C</span><small>只写 mask=true 的位置</small></div>
          {valueLayer("C", `output shape ${dims}D`, (lane) => lane.sum, phase >= 4, "var(--acid)")}
        </div>
      </div>
      <div className="step-line"><b>STEP {phase + 1}/5</b><span>{phases[phase]}</span><code>{phase === 0 ? `pid = ${pidTuple} in grid ${gridTuple}` : phase === 1 ? offsetFormula : phase === 2 ? "x = tl.load(x_ptr + offsets, mask); y = tl.load(y_ptr + offsets, mask)" : phase === 3 ? "c = x + y  # X、Y、C 三组数值分开显示" : "tl.store(c_ptr + offsets, c, mask=mask)"}</code></div>
    </div>
  );
}

function ParallelLab() {
  const [tasks, setTasks] = useState(11);
  const [cores, setCores] = useState(4);
  const [tick, setTick] = useState(0);
  const rounds = Math.ceil(tasks / cores);
  const assignments = Array.from({ length: cores }, (_, pid) => Array.from({ length: rounds }, (_, r) => pid + r * cores).filter((t) => t < tasks));
  return (
    <div className="lab parallel-lab">
      <div className="lab-toolbar">
        <label>total_tasks <input type="range" min="5" max="18" value={tasks} onChange={(e) => { setTasks(+e.target.value); setTick(0); }} /><b>{tasks}</b></label>
        <label>grid / kernel_num <input type="range" min="2" max="6" value={cores} onChange={(e) => { setCores(+e.target.value); setTick(0); }} /><b>{cores}</b></label>
        <label>当前循环轮次 k <b>{tick < rounds ? tick : "结束"}</b></label>
        <button className="action" onClick={() => setTick((tick + 1) % (rounds + 1))}>推进一轮 k →</button>
      </div>
      <div className="task-scope-explainer">
        <div className="scope-answer"><span>total_tasks 到底数什么？</span><strong>{tasks} 个逻辑工作单元</strong><p>它由 kernel 定义“一个 task 做多少工作”，<b>不是某个固定变量的 numel，也不是 X、Y、C 等所有变量 numel 的总和。</b></p></div>
        <div className="task-examples">
          <p><b>向量加</b><code>task = 一段 BLOCK 元素</code><span>total_tasks = ceil(N / BLOCK)</span></p>
          <p><b>RMSNorm</b><code>task = 一行或 block_l 行</code><span>total_tasks = ceil(B×L / block_l)，C 是 task 内部元素</span></p>
          <p><b>矩阵计算</b><code>task = 一个 [BM,BN] 输出 tile</code><span>total_tasks = ceil(M/BM) × ceil(N/BN)</span></p>
        </div>
        <div className="task-scope-flow"><span>完整张量 / 输出空间</span><i>按 kernel 规则切分</i><strong>{tasks} tasks</strong><i>按 pid + k×P 分配</i><strong>{cores} programs</strong></div>
      </div>
      <div className="parallel-map">
        {assignments.map((lane, pid) => <div className="core-lane" key={pid}>
          <div className="core-label"><i />Program {pid}<small>pid={pid}</small></div>
          <div className="timeline">
            {lane.map((task, k) => <div key={task} className={`task ${k < tick ? "done" : k === tick ? "now" : ""}`}><b>k={k} → task {task}</b><small>{pid} + {k}×{cores} = {task}</small></div>)}
          </div>
        </div>)}
      </div>
      <p className="lab-note"><code>for task_id in range(pid, total_tasks, kernel_num)</code>：这正是 RMSNorm 案例的持久化分工。program 数固定为向量核数，先做自己的 pid，再每次跨过整个 grid 领取下一个任务。</p>
    </div>
  );
}

function UbCalculator() {
  const [tile, setTile] = useState(1024);
  const [bytes, setBytes] = useState(2);
  const [live, setLive] = useState(3);
  const [buffers, setBuffers] = useState(1);
  const [capacity, setCapacity] = useState(192);
  const aligned = Math.ceil(tile * bytes / 32) * 32;
  const mask = Math.ceil(tile / 8);
  const payload = aligned * live * buffers;
  const overhead = 16 * 1024 + mask;
  const total = payload + overhead;
  const kib = total / 1024;
  const pct = Math.min(100, kib / capacity * 100);
  return (
    <div className="lab ub-lab">
      <div className="ub-scope-explainer">
        <div className="ub-definition">
          <span>TILE 元素数</span>
          <p><b>Tile 元素数是一个 program 处理一个 task 的一轮中，单张工作 tile 的元素数量。</b></p>
          <strong>当前估算器：{tile.toLocaleString()} elements</strong>
        </div>
        <div className="ub-example-grid">
          <article>
            <small>逐元素算子 · 1D</small>
            <pre><code>{`BLOCK_SIZE = 1024

X_tile = 1024 个元素
Y_tile = 1024 个元素
C_tile = 1024 个元素

tile 元素数 = 1024
同时存活张量 = 3`}</code></pre>
            <p><code>X</code>、<code>Y</code>、<code>C</code> 各是一张 tile；三张形状相同，所以估算器填 <b>1024</b>，再把同时存活张量设为 <b>3</b>，不是把 tile 填成 3072。</p>
          </article>
          <article>
            <small>二维算子 · 2D</small>
            <pre><code>{`BLOCK_M = 32
BLOCK_N = 64

tile 元素数
= BLOCK_M × BLOCK_N
= 32 × 64
= 2048`}</code></pre>
            <p>二维 tile 先按两个轴相乘。这里一张 tile 覆盖 <b>32 行 × 64 列</b>，所以单张 tile 是 <b>2048</b> 个元素。</p>
          </article>
          <article>
            <small>归约算子 · Reduction</small>
            <pre><code>{`BLOCK_L = 4
C = 4096

X_tile 元素数
= BLOCK_L × C
= 4 × 4096
= 16384`}</code></pre>
            <p>归约输入 <code>X_tile</code> 同时覆盖 <b>4 行 × 4096 个归约列</b>。若输出或临时量的 tile 形状不同，应分别计算后相加，不能都套用 16384。</p>
          </article>
        </div>
        <div className="ub-round-cycle">
          <div><span>每个 program 每轮只处理一个 task</span><strong>第 k 轮</strong></div>
          <ol>
            <li>取一个 task</li><li>搬入这个 task 的 tile</li><li>在 UB 计算</li><li>写回</li><li>UB 空间复用</li><li>进入下一轮</li>
          </ol>
        </div>
        <p className="ub-no-multiply"><b>当前快捷估算（假设 {live} 张存活张量的 tile 大小相同）：</b><code>{tile} × {bytes} B × {live} 张同时存活 × {buffers} 份缓冲</code>。不要再乘 <code>total_tasks</code> 或 Grid program 数；task 分轮进入同一个 UB。若各张量 tile 大小不同，应改为逐张计算 <code>Σ align32(tileᵢ × dtypeᵢ)</code>。</p>
      </div>
      <div className="ub-controls">
        <label>单张 Tile 元素数 <input type="number" min="128" step="128" value={tile} onChange={(e) => setTile(Math.max(128, +e.target.value))} /><small>例如向量 BLOCK；二维 tile 填 BM×BN</small></label>
        <label>dtype <select value={bytes} onChange={(e) => setBytes(+e.target.value)}><option value="1">int8 · 1 B</option><option value="2">fp16/bf16 · 2 B</option><option value="4">fp32 · 4 B</option></select></label>
        <label>同时存活张量 <input type="range" min="1" max="6" value={live} onChange={(e) => setLive(+e.target.value)} /><b>{live}</b></label>
        <label>缓冲份数 <select value={buffers} onChange={(e) => setBuffers(+e.target.value)}><option value="1">单缓冲 ×1</option><option value="2">双缓冲 ×2</option></select></label>
        <label>教学 UB 上限 <select value={capacity} onChange={(e) => setCapacity(+e.target.value)}><option>128</option><option>192</option><option>256</option></select></label>
      </div>
      <div className="ub-result">
        <div className="gauge"><div style={{ width: `${pct}%` }} className={pct > 90 ? "danger" : pct > 70 ? "warn" : ""} /></div>
        <div className="ub-big"><strong>{kib.toFixed(1)} KiB</strong><span>/ {capacity} KiB 教学预算</span><em>{pct > 100 ? "估算溢出：减 tile / 存活量 / buffer" : `余量 ${(capacity - kib).toFixed(1)} KiB`}</em></div>
        <div className="formula-stack">
          <p><b>① 单张量</b><code>align32({tile} × {bytes} B) = {(aligned / 1024).toFixed(1)} KiB</code></p>
          <p><b>② 数据工作集</b><code>{(aligned / 1024).toFixed(1)} × {live} live × {buffers} buffer = {(payload / 1024).toFixed(1)} KiB</code></p>
          <p><b>③ 教学开销</b><code>mask {mask} B + 固定预留 16 KiB = {(overhead / 1024).toFixed(1)} KiB</code></p>
        </div>
      </div>
      <div className="truth-note"><b>这是“设计前估算”，不是编译器精确报告。</b> 实际 UB 还受临时量、数据布局、对齐、流水、多缓冲和编译器生命周期分析影响。仓库的融合 argmax 案例直接用 <code>_TILE_BYTES = 32 × 1024</code> 控制单个词表 tile，并指出过宽 tile 与自动多缓冲会触发 UB overflow——所以最终要结合编译日志与逐步减小 BLOCK 验证。</div>
    </div>
  );
}

function CaseSimulator() {
  const [kind, setKind] = useState("rms");
  const [values, setValues] = useState("1,2,3,4");
  const nums = useMemo(() => values.split(/[,\s]+/).map(Number).filter(Number.isFinite).slice(0, 8), [values]);
  const steps = useMemo(() => {
    if (!nums.length) return ["请输入数字"];
    if (kind === "rms") {
      const squares = nums.map((x) => x * x);
      const variance = squares.reduce((a, b) => a + b, 0) / nums.length;
      const rstd = 1 / Math.sqrt(variance + 1e-6);
      return [`x² = [${squares.map(x => x.toFixed(2)).join(", ")}]`, `mean(x²) = ${variance.toFixed(4)}`, `rsqrt(var+eps) = ${rstd.toFixed(4)}`, `y = [${nums.map(x => (x * rstd).toFixed(4)).join(", ")}]`];
    }
    if (kind === "swiglu") {
      const even = nums.filter((_, i) => i % 2 === 0);
      const odd = nums.filter((_, i) => i % 2 === 1);
      const out = even.slice(0, odd.length).map((g, i) => g * (1 / (1 + Math.exp(-g))) * (odd[i] + 1));
      return [`gate = 偶数列 [${even.join(", ")}]`, `up = 奇数列 [${odd.join(", ")}]`, `SiLU(gate) = [${even.map(g => (g / (1 + Math.exp(-g))).toFixed(4)).join(", ")}]`, `out = SiLU(gate) × (up+1) = [${out.map(x => x.toFixed(4)).join(", ")}]`];
    }
    const max = Math.max(...nums); const arg = nums.indexOf(max); const exps = nums.map(x => Math.exp(x - max)); const sum = exps.reduce((a, b) => a + b, 0);
    return [`m = max(logits) = ${max}，argmax = ${arg}`, `exp(x-m) = [${exps.map(x => x.toFixed(4)).join(", ")}]`, `sumexp = ${sum.toFixed(4)}`, `P(argmax) = 1/sumexp = ${(1 / sum).toFixed(4)}`];
  }, [kind, nums]);
  return (
    <div className="case-sim">
      <div className="sim-inputs"><div><span>算子</span>{[["rms", "RMSNorm"], ["swiglu", "SwiGLU"], ["softmax", "Argmax+Softmax"]].map(([id, name]) => <button key={id} className={kind === id ? "active" : ""} onClick={() => setKind(id)}>{name}</button>)}</div><label>输入小张量 <input value={values} onChange={(e) => setValues(e.target.value)} aria-label="逗号分隔的小张量数值" /></label></div>
      <div className="calc-pipeline">{steps.map((s, i) => <div key={s}><b>0{i + 1}</b><span>{s}</span></div>)}</div>
    </div>
  );
}

function MigrationDiff() {
  return <div className="migration-diff">
    <div><small>GPU</small><code>torch.cuda.current_device()</code><code>device=&quot;cuda&quot;</code><code>tensor.cuda()</code><code>torch.cuda.synchronize()</code></div>
    <div className="migration-arrow"><span>kernel 不改</span><b>→</b><em>@triton.jit</em></div>
    <div><small>NPU</small><code>torch.npu.current_device()</code><code>device=&quot;npu&quot;</code><code>tensor.npu()</code><code>torch.npu.synchronize()</code></div>
  </div>;
}

export default function Home() {
  const [menu, setMenu] = useState(false);
  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top"><span className="brand-mark">T↑</span><b>Triton Ascend<br/><em>Visual Lab</em></b></a>
        <nav>{NAV.slice(0, 6).map(([id, , label]) => <a key={id} href={`#${id}`}>{label}</a>)}</nav>
        <a className="github-link" href={SOURCE.sgl} target="_blank" rel="noreferrer">源代码 ↗</a>
        <button className="menu" onClick={() => setMenu(!menu)} aria-label="切换目录">目录</button>
      </header>

      <aside className={menu ? "sidebar open" : "sidebar"}>
        <div className="side-kicker">LEARNING PATH</div>
        {NAV.map(([id, no, label], i) => <a key={id} href={`#${id}`} onClick={() => setMenu(false)}><span>{no}</span><b>{label}</b>{i < 5 && <i />}</a>)}
        <div className="side-progress"><span>从张量到 NPU</span><div><i /></div><small>8 个章节 · 4 个实验</small></div>
      </aside>

      <div className="content" id="top">
        <section className="hero" id="start">
          <div className="hero-copy">
            <div className="eyebrow"><span>ASCEND NPU</span><i />INTERACTIVE TUTORIAL</div>
            <h1>别只读代码。<br/><em>看见数据怎么跑。</em></h1>
            <p>一份从 PyTorch 到 Triton-Ascend 的中文可视化教程：亲手调 <code>grid</code>、追踪 <code>offset</code>、估算 UB，并拆开真实推理算子的每一步。</p>
            <div className="hero-actions"><a href="#transfer" className="primary">开始搬运实验 <b>→</b></a><a href="#start-checklist">先配置环境</a></div>
            <div className="hero-meta"><span><b>4</b> 交互实验</span><span><b>14</b> 接口对照</span><span><b>4</b> 真实算子</span></div>
          </div>
          <div className="hero-visual" aria-label="program 并行处理张量示意图">
            <div className="visual-head"><span>VECTOR_ADD.TRITON</span><i>LIVE TRACE</i></div>
            <div className="matrix">
              {Array.from({ length: 32 }, (_, i) => <span key={i} className={i >= 8 && i < 16 ? "hot" : i >= 24 ? "tail" : ""}>{i}</span>)}
            </div>
            <div className="trace"><span>program_id</span><b>01</b><i>×</i><span>BLOCK_SIZE</span><b>08</b><i>=</i><strong>offset 08</strong></div>
            <div className="core-track"><span>NPU Core 0</span><span className="active">NPU Core 1 · COMPUTE</span><span>NPU Core 2</span></div>
            <div className="scanline" />
          </div>
        </section>

        <section className="quick-section" id="start-checklist">
          <div className="section-head"><span>01 / QUICK START</span><h2>先跑通，再拆开。</h2><p>覆盖官方 Quick Start 的安装、验证和 CUDA→NPU 迁移，并补上每一步为什么。</p></div>
          <div className="requirements">
            <article><span>硬件</span><b>Atlas A2 / A3 / A5</b><p>Linux aarch64 / x86_64；官方快速开始建议单卡 32 GB 内存。</p></article>
            <article><span>软件</span><b>Python 3.9–3.11</b><p>CANN 推荐 9.0.0；quick start 当前匹配 torch_npu 2.7.1.post4。</p></article>
            <article><span>安装</span><b>triton-ascend 3.2.1+</b><p>此版本起声明 Triton 依赖，缓解后装依赖覆盖 Ascend 版本的问题。</p></article>
          </div>
          <div className="setup-grid">
            <div>
              <h3><span>①</span> 安装 wheel</h3>
              <CodeBlock label="shell">{`pip install triton-ascend==3.2.1 \\
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple`}</CodeBlock>
              <h3><span>②</span> 加载 CANN 并运行官方向量加</h3>
              <CodeBlock label="shell">{`source /usr/local/Ascend/ascend-toolkit/set_env.sh
git clone https://github.com/triton-lang/triton-ascend.git
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py`}</CodeBlock>
            </div>
            <div className="expected">
              <span>EXPECTED SIGNAL</span><h3>看到两份张量一致，最大误差为 0.0</h3>
              <pre>torch  → tensor([0.8329, 1.0024, ...]){"\n"}triton → tensor([0.8329, 1.0024, ...]){"\n"}<b>max diff = 0.0 ✓</b></pre>
              <p>这验证的是：NPU 能启动 Triton kernel，并且结果与原生 PyTorch 对齐。它不等于“性能已经最优”。</p>
            </div>
          </div>
          <h3 className="subhead">GPU 脚本迁到 NPU：只替换设备边界</h3>
          <MigrationDiff />
          <div className="run-note"><code>pytest test_add.py</code><span>kernel 的 <code>@triton.jit</code> 逻辑与 <code>grid</code> 启动方式通常保持不变；若缺 pytest，先 <code>pip install pytest</code>。</span></div>
        </section>

        <section id="anatomy">
          <div className="section-head"><span>02 / KERNEL ANATOMY</span><h2>11 行 kernel，10 个关键变量。</h2><p>指针决定“从哪取”，offset 决定“取哪一格”，mask 决定“这一格算不算”。</p></div>
          <div className="anatomy-grid">
            <CodeBlock>{ADD_KERNEL}</CodeBlock>
            <div className="variable-list">{VARIABLE_ROWS.map(([name, role, detail], i) => <div key={name}><span>{String(i + 1).padStart(2, "0")}</span><code>{name}</code><b>{role}</b><p>{detail}</p></div>)}</div>
          </div>
          <div className="address-rule"><span>地址公式</span><strong>element address = base pointer + logical offset × element_size</strong><p>Triton 的指针加法按元素类型缩放；二维展平常写成 <code>row × stride + col</code>，不要把“元素下标”误当成“字节地址”。</p></div>
        </section>

        <section id="transfer" className="wide-section dark-section">
          <div className="section-head light"><span>03 / DATA MOVEMENT</span><h2>切换 1D / 2D / 3D，看每个 block 怎样搬。</h2><p>逐轴设定 shape、BLOCK_SIZE 与 program_id，观察 X、Y 从 GM 进入 UB、并行相加并把 C 写回 GM 的完整生命周期。</p></div>
          <TransferLab />
        </section>

        <section id="parallel">
          <div className="section-head"><span>04 / GRID PARALLELISM</span><h2>Grid 不是数据；它是一张任务清单。</h2><p><code>grid=(P,)</code> 启动 P 个 program。硬件何时调度它们由运行时决定，因此 program 间不能假设先后顺序。</p></div>
          <div className="concept-row">
            <article><b>1D grid</b><code>pid = tl.program_id(0)</code><p>向量、行归约常用。第 pid 个 program 处理一段连续元素或若干 token。</p></article>
            <article><b>2D grid</b><code>pid_m, pid_n</code><p>矩阵 tile 常用。一个 program 对应输出矩阵中的一个二维块。</p></article>
            <article className="persistent-concept"><b>Persistent grid</b><code>task = pid + k×P</code><p>program 数贴近核心数，同一个 program 留在设备上循环领取任务。</p><dl><div><dt>k</dt><dd>循环轮次，从 0 开始：0、1、2…；不是新的 program_id</dd></div><div><dt>P</dt><dd>Grid 中 program 总数，即 <code>tl.num_programs(0)</code></dd></div><div><dt>停止</dt><dd>当 <code>pid + k×P ≥ total_tasks</code> 时不再领取</dd></div></dl></article>
          </div>
          <ParallelLab />
          <div className="grid-math"><div><span>向量加</span><strong>grid = (ceil(N / BLOCK),)</strong><small>N=98,432；BLOCK=1,024 → 97 programs</small></div><div><span>RMSNorm</span><strong>grid = (num_vectorcore,)</strong><small>total_tasks = ceil(B×L / block_l)，跨步消费</small></div><div><span>二维矩阵</span><strong>grid = (ceil(M/BM), ceil(N/BN))</strong><small>program (pm,pn) → C 的 [BM,BN] tile</small></div></div>
        </section>

        <section id="ub" className="ub-section">
          <div className="section-head"><span>05 / UNIFIED BUFFER</span><h2>UB 不是“输入大小”，是峰值存活工作集。</h2><p>先估算，再编译验证。决定是否溢出的，是某个时刻同时活着的 tile、临时量、mask、对齐与流水缓冲。</p></div>
          <UbCalculator />
          <div className="ub-checklist"><article><span>01</span><b>画生命周期</b><p>load 进来的 x、y 何时能释放？输出和 fp32 累加是否同时存在？</p></article><article><span>02</span><b>逐项乘 dtype</b><p>bf16 输入 2 B，但归约常转 fp32 变 4 B；别只看输入 dtype。</p></article><article><span>03</span><b>算对齐与多缓冲</b><p>连续块按硬件对齐；流水化可能让上一个 tile 与下一个 tile 同时驻留。</p></article><article><span>04</span><b>保留安全余量</b><p>估算逼近上限就不稳。出现 UB overflow，先减 BLOCK / tile，再减少临时量。</p></article></div>
        </section>

        <section id="cases" className="cases-section">
          <div className="section-head"><span>06 / PRODUCTION CASES</span><h2>从 SGL Kernel NPU 拆 4 个真实算子。</h2><p>每个案例都给出输入、输出、grid、变量含义、PyTorch 参考和计算链；点击源码可回到仓库核对。</p></div>
          <CaseSimulator />
          <div className="case-list">{CASES.map((item) => <article key={item.id}>
            <div className="case-title"><span>{item.no}</span><h3>{item.name}</h3><a href={item.source} target="_blank" rel="noreferrer">源码 ↗</a></div>
            <div className="io-flow"><div><small>INPUT</small><p>{item.input}</p></div><i>→</i><div><small>OUTPUT</small><p>{item.output}</p></div></div>
            <div className="compute-chain">{item.formula.split("→").map((x, i) => <span key={i}>{x.trim()}</span>)}</div>
            <dl><div><dt>GRID</dt><dd>{item.grid}</dd></div><div><dt>VARIABLES</dt><dd>{item.vars}</dd></div><div><dt>TORCH REFERENCE</dt><dd><code>{item.torch}</code></dd></div></dl>
          </article>)}</div>
        </section>

        <section id="api" className="wide-section api-section">
          <div className="section-head"><span>07 / API ATLAS</span><h2>Triton 接口 ↔ PyTorch 心智模型。</h2><p>“等价”指计算语义对应，不保证内存行为、并行粒度或性能相同。</p></div>
          <div className="table-wrap"><table><thead><tr><th>Triton / Python</th><th>设备侧含义</th><th>PyTorch 对应理解</th></tr></thead><tbody>{API_ROWS.map(r => <tr key={r[0]}><td><code>{r[0]}</code></td><td>{r[1]}</td><td>{r[2]}</td></tr>)}</tbody></table></div>
          <div className="api-principle"><b>边界原则</b><p>PyTorch 负责张量创建、形状、设备、参考结果与测试；Triton kernel 负责显式索引、片上计算和写回。先写正确的 Torch reference，再优化 Triton。</p><code>torch reference → Triton kernel → assert_close → profile → tune</code></div>
        </section>

        <section id="debug">
          <div className="section-head"><span>08 / VERIFY & DEBUG</span><h2>正确性、UB、性能：按这个顺序排。</h2><p>一次只改变一个变量，才能知道是哪一步让结果或性能发生变化。</p></div>
          <div className="debug-flow"><article><span>01</span><h3>小尺寸手算</h3><p>N 取 17、BLOCK 取 8，专门覆盖尾块。打印或模拟每个 pid 的 offsets 与 mask。</p><code>assert mask.sum() == valid_count</code></article><article><span>02</span><h3>对齐 Torch</h3><p>用 float32 reference；根据 dtype 设置 rtol / atol。分别测随机、全零、极值、非连续 stride。</p><code>torch.testing.assert_close(out, ref)</code></article><article><span>03</span><h3>定位 UB</h3><p>减半 BLOCK。若能编译，回查临时 fp32、广播结果、归约树和多缓冲的峰值重叠。</p><code>BLOCK: 32768 → 16384 → 8192</code></article><article><span>04</span><h3>最后再调优</h3><p>同步后计时；比较多个 shape。用 autotune 的 key 覆盖真正改变最优配置的维度。</p><code>torch.npu.synchronize()</code></article></div>
          <div className="pitfalls"><h3>常见误区</h3><div><p><b>把 grid 当线程数</b><span>Triton program 是较粗的并行实例，内部向量 lane 由编译器映射。</span></p><p><b>漏掉尾块 mask</b><span>load 与 store 两边都要 mask；other 值要适合归约语义。</span></p><p><b>只算输入 UB</b><span>广播、fp32 累加、临时结果和流水 buffer 才经常是峰值来源。</span></p><p><b>计时不 synchronize</b><span>NPU 异步执行，不同步会把 launch 时间误当成 kernel 时间。</span></p></div></div>
        </section>

        <section className="source-section">
          <div><span>SOURCE LEDGER</span><h2>继续深入，而不是停在这页。</h2></div>
          <div className="source-links"><a href={SOURCE.quick} target="_blank" rel="noreferrer"><b>官方 Quick Start</b><span>安装、向量加、GPU→NPU 迁移 ↗</span></a><a href={SOURCE.docs} target="_blank" rel="noreferrer"><b>Triton-Ascend 文档</b><span>Vector / Cube / CV / Autotune / 调试 ↗</span></a><a href={SOURCE.sgl} target="_blank" rel="noreferrer"><b>SGL Kernel NPU</b><span>推理算子、DeepEP 与测试用例 ↗</span></a></div>
          <p>内容基于上游仓库当前公开文档与源码整理。硬件支持、版本匹配与接口会变化，实际部署前请以链接中的最新上游说明为准。</p>
        </section>
      </div>
    </main>
  );
}
