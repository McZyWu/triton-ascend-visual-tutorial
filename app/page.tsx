"use client";

import { useEffect, useMemo, useState } from "react";

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
  const [n, setN] = useState(19);
  const [block, setBlock] = useState(8);
  const [pid, setPid] = useState(1);
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const grid = Math.ceil(n / block);
  const safePid = Math.min(pid, Math.max(0, grid - 1));
  const offsets = Array.from({ length: block }, (_, i) => safePid * block + i);
  const valid = offsets.map((x) => x < n);
  const phases = ["定位 program", "GM → UB：load x, y", "UB：x + y", "UB → GM：store output"];
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setPhase((p) => {
      if (p === 3) { setPlaying(false); return 3; }
      return p + 1;
    }), 900);
    return () => window.clearInterval(timer);
  }, [playing]);
  useEffect(() => { setPid((p) => Math.min(p, Math.max(0, grid - 1))); setPhase(0); }, [grid]);

  return (
    <div className="lab transfer-lab">
      <div className="lab-toolbar">
        <label>N 元素 <input type="range" min="9" max="32" value={n} onChange={(e) => setN(+e.target.value)} /><b>{n}</b></label>
        <label>BLOCK_SIZE <select value={block} onChange={(e) => setBlock(+e.target.value)}><option>4</option><option>8</option><option>16</option></select></label>
        <label>program_id <input type="range" min="0" max={Math.max(0, grid - 1)} value={safePid} onChange={(e) => { setPid(+e.target.value); setPhase(0); }} /><b>{safePid}</b></label>
        <button className="action" onClick={() => { setPhase(0); setPlaying(true); }}>{playing ? "搬运中…" : "▶ 播放搬运"}</button>
        <button onClick={() => { setPlaying(false); setPhase((phase + 1) % 4); }}>单步 →</button>
      </div>
      <div className="equation-strip">
        <span>grid = ceil({n}/{block}) = <strong>{grid}</strong></span>
        <span>block_start = {safePid}×{block} = <strong>{safePid * block}</strong></span>
        <span>有效 lane = <strong>{valid.filter(Boolean).length}/{block}</strong></span>
      </div>
      <div className="memory-stage" aria-label="全局内存到统一缓冲区的搬运模拟">
        <div className={`memory-zone gm ${phase === 1 || phase === 3 ? "active" : ""}`}>
          <div className="zone-title"><span>GM / HBM</span><small>大容量 · 高延迟</small></div>
          <div className="tensor-row">{Array.from({ length: n }, (_, i) => <i key={i} className={offsets.includes(i) && phase <= 1 ? "selected" : ""}>{i}</i>)}</div>
        </div>
        <div className={`bus ${phase === 1 ? "to-ub" : phase === 3 ? "to-gm" : ""}`}><span>{phase === 3 ? "STORE" : "LOAD"}</span><i /></div>
        <div className={`memory-zone ub ${phase === 2 ? "active" : ""}`}>
          <div className="zone-title"><span>UB / 片上工作集</span><small>当前 program 私有视图</small></div>
          <div className="lane-grid">
            {offsets.map((off, i) => <div key={i} className={`lane ${valid[i] ? "valid" : "masked"} ${phase === 2 ? "computing" : ""}`}><b>L{i}</b><span>off {off}</span><em>{valid[i] ? (phase >= 2 ? `${off}+${off + 1}=${off * 2 + 1}` : `x[${off}], y[${off}]`) : "MASK"}</em></div>)}
          </div>
        </div>
      </div>
      <div className="step-line"><b>STEP {phase + 1}/4</b><span>{phases[phase]}</span><code>{phase === 0 ? `offsets = ${safePid * block} + arange(0, ${block})` : phase === 1 ? "tl.load(ptr + offsets, mask=offsets < N)" : phase === 2 ? "result = x + y  # 各 lane 同时" : "tl.store(output_ptr + offsets, result, mask)"}</code></div>
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
        <button className="action" onClick={() => setTick((tick + 1) % (rounds + 1))}>推进一轮 →</button>
      </div>
      <div className="parallel-map">
        {assignments.map((lane, pid) => <div className="core-lane" key={pid}>
          <div className="core-label"><i />Program {pid}<small>pid={pid}</small></div>
          <div className="timeline">
            {lane.map((task, r) => <div key={task} className={`task ${r < tick ? "done" : r === tick ? "now" : ""}`}><b>task {task}</b><small>{pid} + {r}×{cores}</small></div>)}
          </div>
        </div>)}
      </div>
      <p className="lab-note"><code>for task_id in range(pid, total_tasks, kernel_num)</code>：这正是 RMSNorm 案例的持久化分工。program 数固定为向量核数，先做自己的 pid，再每次跨过整个 grid 领取下一个任务。</p>
    </div>
  );
}

function UbCalculator() {
  const [tile, setTile] = useState(8192);
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
      <div className="ub-controls">
        <label>Tile 元素数 <input type="number" min="128" step="128" value={tile} onChange={(e) => setTile(Math.max(128, +e.target.value))} /></label>
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
          <div className="section-head light"><span>03 / DATA MOVEMENT</span><h2>拖动 pid，看尾块怎样被 mask。</h2><p>设定 N 与 BLOCK_SIZE，观察 GM → UB → ALU → GM 的完整生命周期。</p></div>
          <TransferLab />
        </section>

        <section id="parallel">
          <div className="section-head"><span>04 / GRID PARALLELISM</span><h2>Grid 不是数据；它是一张任务清单。</h2><p><code>grid=(P,)</code> 启动 P 个 program。硬件何时调度它们由运行时决定，因此 program 间不能假设先后顺序。</p></div>
          <div className="concept-row">
            <article><b>1D grid</b><code>pid = tl.program_id(0)</code><p>向量、行归约常用。第 pid 个 program 处理一段连续元素或若干 token。</p></article>
            <article><b>2D grid</b><code>pid_m, pid_n</code><p>矩阵 tile 常用。一个 program 对应输出矩阵中的一个二维块。</p></article>
            <article><b>Persistent grid</b><code>task = pid + k×P</code><p>program 数贴近核心数，循环领取更多任务，减少过多 program 的调度开销。</p></article>
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
