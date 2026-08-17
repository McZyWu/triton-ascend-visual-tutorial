import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { BoundCalculator, FieldExplorer, TraceViewer } from "./profiling-lab";
import "./profiling.css";

const CUTOFF = "2a87cda308d3c5e38c6a0da74c2146146efbe625";

export const metadata: Metadata = {
  title: "Ascend Profiling：从采集到 Bound 与优化",
  description: "单算子采集、kernel_details.csv、trace.json、流水、热点、Bound 与优化收益的可视化实战教程。",
  openGraph: {
    title: "Ascend Profiling Field Guide",
    description: "从单算子采集，到流水、Bound、源码热点与优化验收。",
    images: [{ url: "/og.png", width: 1200, height: 630 }],
  },
};

const HOTSPOTS = [
  ["_causal_conv1d_linear_verify_kernel_0", "74.914", 100],
  ["aclnnMatmul_MatMulCommon_MatMulV2", "53.316", 71],
  ["AivKernel", "49.211", 66],
  ["GroupedMatmul", "32.407", 43],
  ["QuantBatchMatmulV3", "23.874", 32],
  ["MoeLowLatencyCombineV2", "18.253", 24],
] as const;

const CASES = [
  { model: "Kimi-K3", kernel: "_situ_deepep_kernel_0", count: "276 calls", duration: "10.44 μs", range: "8.74 — 21.30 μs", vector: "22.55%", scalar: "10.70%", mte2: "10.30%", mte3: "1.85%", image: "/profiling-data/k3-situ-profile.png", alt: "Kimi-K3 SiTU 生产 profiling 时长和流水占比图" },
  { model: "Qwen3.5 / Next", kernel: "split_qkv_rmsnorm_rope_kernel_0", count: "15 calls", duration: "12.32 μs", range: "11.52 — 23.50 μs", vector: "20.00%", scalar: "18.60%", mte2: "29.10%", mte3: "14.30%", image: "/profiling-data/qwen-fused-profile.png", alt: "Qwen fused split RMSNorm RoPE 生产 profiling 时长和流水占比图" },
] as const;

export default function ProfilingPage() {
  return <main className="profile-page">
    <header className="pf-topbar">
      <Link href="/kernel-lab">← Production Kernel Lab</Link>
      <div><b>ASCEND PROFILING FIELD GUIDE</b><span>采集 → 流水 → 热点 → Bound → 优化 → 验收</span></div>
      <a href={`https://github.com/sgl-project/sgl-kernel-npu/commit/${CUTOFF}`} target="_blank" rel="noreferrer">visual cutoff · 2a87cda ↗</a>
    </header>

    <section className="pf-hero">
      <div className="pf-hero-copy"><span>PROFILING / 单独专题</span><h1>从一条事件<br />走到<em>优化证据</em></h1><p>这不是只教你打开一张时间图。页面把单算子采集、<code>kernel_details.csv</code> 字段、<code>trace.json</code>、MTE/Vector/Cube 流水、热点源码、Roofline Bound、UB 工作集和优化验收串成一条可复现的证据链。</p><div className="pf-actions"><a href="#capture">从采集开始</a><a href="/profiling-data/k3-qwen-trace-extract.json" download>下载真实 trace extract</a></div></div>
      <aside className="pf-provenance"><span>DATA PROVENANCE</span><b>209 · 2026-08-05</b><dl><div><dt>current capture</dt><dd>20,153 rows · 1,265 kernel names</dd></div><div><dt>baseline context</dt><dd>79,342 rows · 1,635 kernel names</dd></div><div><dt>two cases</dt><dd>Kimi-K3 + Qwen3.5/Next</dd></div><div><dt>trace source</dt><dd>299,671 events · exact extract</dd></div><div><dt>evidence rule</dt><dd>不同 shape 不计算 speedup</dd></div></dl><p>当前 SGLang 主线没有名为 Qwen3.6 的模型类；原任务中的“Qwen3.6/Next”在仓库证据中对应 Qwen3.5 与 Qwen3-Next 共享算子路径。</p></aside>
    </section>

    <nav className="pf-jump" aria-label="Profiling 六部分">
      <a href="#capture"><b>01</b>单算子采集</a><a href="#pipeline"><b>02</b>流水 / Trace</a><a href="#hotspot"><b>03</b>热点代码</a><a href="#bound"><b>04</b>Bound 计算</a><a href="#optimize"><b>05</b>怎么优化</a><a href="#results"><b>06</b>结果收益</a>
    </nav>

    <section className="pf-section" id="capture">
      <header><span>01 / SINGLE OP CAPTURE</span><h2>怎么采集、怎么读 <code>kernel_details.csv</code></h2><p>先把编译、host 调度和 NPU 执行拆开。固定 shape、dtype、CANN/Triton 版本与设备；预热编译后，只包围待测调用，并在每轮同步和校验数值。</p></header>
      <div className="capture-flow" aria-label="单算子采集流程"><article><b>1</b><strong>隔离调用</strong><p>只保留一个 operator；打印输入/输出 shape、dtype、stride 和模型阶段。</p></article><i>→</i><article><b>2</b><strong>预热</strong><p>让 JIT、autotune、内存池稳定，预热数据不进入统计。</p></article><i>→</i><article><b>3</b><strong>Active 采集</strong><p>Level1 + PipeUtilization，执行多轮并 <code>synchronize()</code>。</p></article><i>→</i><article><b>4</b><strong>导出/分组</strong><p>按 Name 聚合 count、sum、median、P95，再查看各流水。</p></article></div>
      <div className="command-grid"><article><span>TORCH_NPU · 板端统计</span><pre><code>{`config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=ProfilerLevel.Level1,
    aic_metrics=AiCMetrics.PipeUtilization,
)
with torch_npu.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=1, warmup=2, active=5),
    experimental_config=config,
    on_trace_ready=tensorboard_trace_handler(out_dir),
) as prof:
    for _ in range(8):
        output = op(*inputs)
        torch.npu.synchronize()
        prof.step()`}</code></pre></article><article><span>MSPROF OP · 单 kernel / 行号</span><pre><code>{`export TRITON_DISABLE_LINE_INFO=false
msprof op --kernel-name=<exact_kernel_name> \\
  python3 benchmark_single_op.py

# 仿真器工作流输出：
# trace.json             → chrome://tracing/
# visualize_data.bin     → MindStudio Insight

# torch_npu/CANN 导出：
# kernel_details.csv     → 聚合、流水占比、shape
# trace_view.json        → Chrome Trace 事件`}</code></pre></article></div>
      <div className="sample-row"><span>REAL ROW · 209</span><code>_situ_deepep_kernel_0</code><b>Duration 10.44 μs median</b><b>AI_VECTOR_CORE</b><b>MTE2 10.30%</b><small>自定义 kernel 的 shape / Block Num 字段为空，必须回到调用参数与源码；不能把 0 解释为“没有并行”。</small></div>
      <FieldExplorer />
    </section>

    <section className="pf-section pf-dark" id="pipeline">
      <header><span>02 / PIPELINE & TRACE.JSON</span><h2>怎么采集、怎么看流水</h2><p><code>kernel_details.csv</code> 负责统计视图，<code>trace_view.json</code> / <code>trace.json</code> 负责事件时序。先找长尾和高总耗时 kernel，再放大一条事件，判断搬运与计算是否重叠。</p></header>
      <div className="trace-how"><article><b>A · kernel_details</b><p>Duration 排序找慢核；<code>count × duration</code> 找总热点；Pipeline ratio 找下一步方向。</p></article><article><b>B · trace_view.json</b><p>在 <code>chrome://tracing/</code> Load 文件，搜索 kernel 名，点击事件看 ts/dur/Task Id。</p></article><article><b>C · simulator trace.json</b><p>展开 MTE2、Vector、Scalar、MTE3 轨道；结合源码行号观察 load/compute/store 的先后和重叠。</p></article></div>
      <TraceViewer />
      <div className="chrome-steps"><div><span>CHROME 操作</span><ol><li>地址栏输入 <code>chrome://tracing/</code>。</li><li>点击 Load，选择下载的 <code>k3-qwen-trace-extract.json</code>。</li><li>按 <kbd>W</kbd>/<kbd>S</kbd> 缩放、<kbd>A</kbd>/<kbd>D</kbd> 平移；搜索 kernel 名。</li><li>点击事件读取 Duration；同一 stream 上判断间隔，不用 host wall time 替代。</li></ol><a href="/profiling-data/k3-qwen-trace-extract.json" download>下载 10 条真实 NPU 事件</a></div><p><b>为什么是 extract？</b>原 current trace 为 73,572,131 B，baseline trace 约 2.97 GB。这里从 299,671 条事件中原样抽取两个目标 kernel 各 5 条，只把时间戳整体减去首事件；name、dur、stream、Task Id 和 connection_id 未改。完整文件太大，不适合作为网页教学附件。</p></div>
      <div className="pipeline-rule"><b>不要相加</b><span>Duration ≠ Vector + Scalar + MTE2 + MTE3</span><p>它们是可能重叠的独立流水活动。MTE2/MTE3 高不自动等于 memory-bound；还要看字节量、理论搬运下限、有效带宽和是否已与计算隐藏。</p></div>
    </section>

    <section className="pf-section" id="hotspot">
      <header><span>03 / HOTSPOT → SOURCE</span><h2>从总热点映射到 load / compute / store</h2><p>“最慢一次”不一定是优化优先级最高。先按 <code>sum(Duration)</code> 排 Pareto，再进入目标 kernel，把每段 Triton 源码对应到 grid、tile、UB 工作集和流水。</p></header>
      <div className="hotspot-board"><div><span>209 CURRENT CAPTURE · TOP TOTAL DEVICE TIME</span>{HOTSPOTS.map(([name, total, width]) => <div className="hotspot-row" key={name}><code>{name}</code><i><b style={{ width: `${width}%` }} /></i><strong>{total} ms</strong></div>)}</div><aside><b>总耗时公式</b><code>hotspot_time = count × mean(Duration)</code><p>例如 verify kernel 单次 359.78 μs、调用 207 次，总计 74.914 ms；它比只看单次耗时更能反映端到端收益上限。</p></aside></div>

      <div className="case-stack">
        <article className="source-case">
          <header><span>CASE A · KIMI-K3</span><h3>SiTU：从按 row 到按 hidden tile 调度</h3><p><code>x [..., 2d]</code> 是 MoE gate 与 up 投影拼接结果；左半是 gate，右半是 up。输出 <code>[..., d]</code> 回到后续专家 FFN 路径。</p></header>
          <div className="tensor-contract"><div><span>INPUT</span><b>x</b><code>[tokens, 2 × d] · BF16</code><small>gate | up</small></div><i>→</i><div><span>COMPUTE</span><b>SiTU × up</b><code>β·tanh(gate/β)·sigmoid(gate)</code><small>可选 linear_beta 限幅 up</small></div><i>→</i><div><span>OUTPUT</span><b>out</b><code>[tokens, d]</code><small>同输入 dtype</small></div></div>
          <div className="code-map"><pre><code>{`pid = tl.program_id(0)
total_elements = total_rows * HALF_COLS
num_tiles = cdiv(total_elements, BLOCK_H)
for tile_idx in range(pid, num_tiles, NUM_CORES):
    linear = tile_idx * BLOCK_H + lane
    row = linear // HALF_COLS
    h   = linear % HALF_COLS
    gate = tl.load(x + row*2d + h)       # MTE2
    up   = tl.load(x + row*2d + d + h)   # MTE2
    out  = situ(gate) * up                # Vector
    tl.store(y + linear, out)             # MTE3`}</code></pre><div><div><span>GRID</span><b>(num_vectorcore,)</b><p><code>task = pid + k × NUM_CORES</code></p></div><div><span>BLOCK / TILE</span><b>BLOCK_H elements</b><p><code>lane = 0…BLOCK_H-1</code></p></div><div><span>ACTIVE PROGRAMS</span><b>min(C, ceil(T×d/BLOCK_H))</b><p>T=1,d=33792,BLOCK_H=4096 → 9 个 hidden tiles，可暴露 9 个并行 program；旧 row 调度只有 1 个。</p></div></div></div>
          <div className="ub-ledger"><b>SiTU · BLOCK_H=4096 的逻辑工作集</b><span>GM 一轮最低流量：gate 8 KiB + up 8 KiB + out 8 KiB = <strong>24 KiB</strong></span><span>FP32 活跃 payload 下界：gate 16 KiB + up 16 KiB + out 16 KiB = <strong>48 KiB</strong></span><small>tanh/sigmoid 临时量、编译器生命周期复用和 multibuffer 会改变真实 UB；最终以编译产物的 local-memory/.stack 记录为准，不能把源码变量机械求和。</small></div>
        </article>

        <article className="source-case">
          <header><span>CASE B · QWEN3.5 / QWEN3-NEXT SHARED PATH</span><h3>Split QK/Gate + RMSNorm + RoPE：一次 kernel 内完成</h3><p><code>input</code> 是投影后的 Q/Gate/K/V 拼接隐藏状态。这个融合族在 Qwen3.5 与 Qwen3-Next 路径复用；生产 profile 名为 <code>split_qkv_rmsnorm_rope_kernel_0</code>，仓库严格测试覆盖带 gate 的 Gemma-style 变体。</p></header>
          <div className="tensor-contract tensor-wide"><div><span>INPUT</span><b>input</b><code>[T, 2Q + 2KV]</code><small>Q|gate interleaved + K + V</small></div><div><span>AUX</span><b>sin / cos</b><code>[T, 1, 1, rope_dim]</code><small>q_weight/k_weight [head_dim]</small></div><i>→</i><div><span>OUTPUTS</span><b>q, k, v, gate</b><code>[T,Q], [T,KV]×2, [T,Q]</code><small>V/gate 要求 bit-exact</small></div></div>
          <div className="grid-sim"><div><span>示例输入</span><b>T=3 · q_heads=8 · kv_heads=2</b><code>head_dim=128 · rope_dim=64</code></div><i>→</i><div><span>块大小</span><b>KV_BLOCK=128</b><code>Q_BLOCK=512 · Q_GATE_BLOCK=1024</code></div><i>→</i><div><span>二维 GRID</span><b>(20, 2, 1) = 40 programs</b><code>n_cols=256/128=2 · n_rows=ceil(40/2)=20</code></div></div>
          <div className="program-map"><div><b>program (row_pid=0, col_pid=0)</b><span>Q/Gate heads 0–3</span><span>K/V head 0</span><code>rows: 0, 20, 40…</code></div><div><b>program (0, 1)</b><span>Q/Gate heads 4–7</span><span>K/V head 1</span><code>rows: 0, 20, 40…</code></div><div><b>program (1, 0)</b><span>同列下一 row stream</span><span>处理 token 1, 21…</span><code>row_idx = row_pid + k×20</code></div></div>
          <div className="stage-map"><div><span>MTE2</span><b>load Q|gate</b><code>1024 BF16</code></div><i>→</i><div><span>VECTOR</span><b>square → mean → rsqrt</b><code>RMSNorm + weight</code></div><i>→</i><div><span>MTE2 + VECTOR</span><b>load sin/cos → rotate</b><code>RoPE on Q/K</code></div><i>→</i><div><span>MTE3</span><b>store q/k/v/gate</b><code>四个输出</code></div></div>
          <div className="ub-ledger"><b>示例 program 的 Q-stage payload</b><span>input Q|gate：1024 × BF16 = <strong>2 KiB GM tile</strong></span><span>FP32 input + normalized Q + weights + sin/cos：4 + 2 + 0.5 + 0.5 = <strong>至少 7 KiB 活跃 payload</strong></span><small>square、reduction、rot_x、cat_x 的 SSA 临时量与编译复用尚未包含。Q、K、V 三阶段顺序复用 UB，所以不要把三个阶段峰值相加；真实 UB 仍以该 shape 编译产物为准。</small></div>
        </article>
      </div>
    </section>

    <section className="pf-section pf-dark" id="bound">
      <header><span>04 / BOUND</span><h2>怎么算 memory / vector / scalar / cube bound</h2><p>先算硬件理论下限，再用流水做交叉验证。最小必要字节量来自输入与输出，不包含 UB 内部临时量；运算量要按统一口径估算。对于 tanh、sigmoid、rsqrt 这类特殊函数，单一 FLOP 数并不能完全描述 Vector 吞吐。</p></header>
      <div className="formula-strip"><div><span>搬运下限</span><code>t_mem = bytes / bandwidth</code></div><div><span>计算下限</span><code>t_compute = ops / peak</code></div><div><span>算术强度</span><code>AI = ops / bytes</code></div><div><span>Roofline 拐点</span><code>ridge = peak / bandwidth</code></div></div>
      <BoundCalculator />
      <div className="bound-evidence"><article><span>SiTU 快照</span><b>Vector 22.55% · MTE2 10.30%</b><p>tanh/sigmoid 是 Vector 特殊函数，不能只用 BF16 峰值 TOPS 断言。生产快照支持“Vector/Memory 混合”的优化方向，但不是完整受控 bound 证明。</p></article><article><span>Qwen fused 快照</span><b>MTE2 29.10% 为最高流水占比</b><p>优先核对 Q/Gate 连续读取、sin/cos 复用与 tile；同时 Scalar 18.60%，地址/循环也值得检查。MTE2 与 Vector 可重叠，不相加。</p></article><article><span>Cube kernel</span><b>看 aic_mac + MTE1 + FIXPIPE</b><p>MAC 高且接近计算下限是 cube-bound；MTE1/FIXPIPE 主导则检查 L1/UB 搬运、布局和写回。不要用 AIV 字段解释 matmul。</p></article></div>
    </section>

    <section className="pf-section" id="optimize">
      <header><span>05 / OPTIMIZE</span><h2>看到哪条流水，就改哪类代码</h2><p>优化目标必须能落回一条可测试假设：改 grid 解决空闲核，改 tile/访问解决搬运，融合解决中间张量落盘，减少地址与分支解决 Scalar，改变矩阵分块解决 Cube。</p></header>
      <div className="decision-grid"><article><span>MTE2 / MTE3</span><h3>Memory-side</h3><p>连续访问、对齐、减少重复 load/store、融合中间张量、增大可承受 tile、用双缓冲隐藏搬运。</p><code>证据：有效 GB/s + trace 重叠</code></article><article><span>VECTOR</span><h3>Vector-side</h3><p>减少特殊函数次数，复用归约结果，检查 dtype fallback，扩大每次向量长度但不溢出 UB。</p><code>证据：vec ratio + 指令 trace</code></article><article><span>SCALAR / FLOWCTRL</span><h3>Control-side</h3><p>预计算 offset，减少动态循环和分支，避免过小 tile，提高每个 program 的有效工作量。</p><code>证据：scalar ratio + lane 利用</code></article><article><span>CUBE / MAC</span><h3>Compute-side</h3><p>匹配矩阵 tile、布局和精度，复用 L1/UB 数据，减少边界 tile 与格式转换。</p><code>证据：MAC 利用 + achieved TOPS</code></article></div>
      <div className="optimization-diff"><div><span>KIMI-K3 · SiTU</span><h3>优化前：task = row</h3><code>decode T=1 → 只有 program 0 有效</code><div className="core-strip"><b className="on">P0</b>{Array.from({length: 11}, (_, i) => <b key={i}>P{i+1}</b>)}</div></div><i>→</i><div><span>HIDDEN TILE SCHEDULING</span><h3>优化后：task = hidden tile</h3><code>active = min(C, ceil(T×d/BLOCK_H))</code><div className="core-strip"><b className="on">P0</b><b className="on">P1</b><b className="on">P2</b><b className="on">P3</b><b className="on">P4</b><b className="on">P5</b><b className="on">P6</b><b className="on">P7</b><b className="on">P8</b><b>P9</b><b>P10</b><b>P11</b></div></div></div>
      <div className="fusion-benefit"><span>QWEN · FUSION</span><div><b>未融合</b><code>split → GM → RMSNorm → GM → RoPE → GM</code><small>多个 launch + 中间张量往返</small></div><i>→</i><div><b>融合 kernel</b><code>load → split/norm/rope in UB → store final</code><small>一次 grid；中间值不落 GM</small></div></div>
      <div className="test-matrix"><article><span>SiTU 严格精度</span><b>6 个核心 shape + grouped + invalid</b><code>(T,d,dtype) = (1,3072,BF16), (2,3584,BF16), (7,6144,FP16), (49,3072,BF16), (1,33792,BF16), (3,1024,FP32)</code><p>对照 Torch FP32 reference；BF16/FP16/FP32 分别使用明确 atol/rtol；覆盖 linear_beta=None 与 group_list。</p></article><article><span>Qwen fused 严格精度</span><b>4 个 token/head/rope 组合</b><code>(1,4,2,128,32,BF16) … (49,4,2,128,64,FP16)</code><p>Q/K 用严格容差；V 与 gate 要求逐 bit 相等。性能优化若破坏精度，直接不通过验收。</p></article></div>
    </section>

    <section className="pf-section pf-results" id="results">
      <header><span>06 / RESULTS & BENEFIT</span><h2>优化后看什么结果，哪些能叫“收益”</h2><p>受控 A/B 必须是同模型阶段、同 shape、同 dtype、同 CANN/Triton 版本和同设备。下面两张图来自同一份 209 生产快照，可证明当前行为和优化方向；旧 baseline 的 shape 不同，因此只作背景，不计算加速比。</p></header>
      <div className="result-cases">{CASES.map(item => <article key={item.kernel}><header><span>{item.model}</span><h3>{item.kernel}</h3><code>{item.count} · median {item.duration} · range {item.range}</code></header><Image unoptimized src={item.image} width={1600} height={900} sizes="(max-width: 900px) 100vw, 50vw" alt={item.alt} /><div><b>Vector {item.vector}</b><b>Scalar {item.scalar}</b><b>MTE2 {item.mte2}</b><b>MTE3 {item.mte3}</b></div></article>)}</div>
      <div className="benefit-ledger"><article><span>SiTU 结构收益</span><b>decode 不再被“token 行数”锁死</b><p>并行暴露从 <code>min(C,T)</code> 变为 <code>min(C,ceil(T×d/BLOCK_H))</code>。对 T=1,d=33792,BLOCK_H=4096，理论活跃 program 从 1 变为 9。</p></article><article><span>Qwen 融合收益</span><b>中间 Q/K 不必跨 kernel 落回 GM</b><p>split、RMSNorm、RoPE 在同一 program 中连续执行；收益来源是减少 launch 和 compulsory GM round trips，而不是把多个旧 profile 时长直接相加。</p></article><article><span>干净的受控样例</span><b>mul_add_kernel · 3.240 μs median</b><p>B=48,H=4096,BF16，5 次 active；compiled local memory 精确为 32 KiB。它展示完整的“同形状实测 + UB 编译证据”范式。</p><Link href="/kernel-lab?op=muladd">打开逐段模拟 →</Link></article></div>
      <div className="evidence-contract"><b>只有满足下面四项，才在结果表写 speedup</b><ol><li>同 shape/dtype/model stage 的 before 与 after。</li><li>预热后多轮 device Duration，报告 median/P95，而非只报最好一次。</li><li>同一套最低字节量/OPS 口径，比较 GB/s、TOPS 与主导流水。</li><li>Torch reference 与严格多 shape 回归全部通过。</li></ol><p>当前历史 baseline 中 <code>_situ_and_mul_kernel</code> 的中位数为 58.039 μs，但主要 shape 是 <code>[2048,768]</code>；current custom kernel 的 shape 字段缺失。所以本页没有把它与 10.44 μs 相除。</p></div>
      <figure className="muladd-figure"><Image unoptimized src="/profiling-data/mul-add-pipeline-timeline.png" width={1600} height={900} sizes="(max-width: 900px) 100vw, 1200px" alt="209 单算子 mul_add 的真实 Duration 与流水活动时间图" /><figcaption>受控单算子示例：流水可重叠，Scalar + Vector + MTE2 + MTE3 不等于 Duration。</figcaption></figure>
    </section>

    <section className="pf-section pf-resources">
      <header><span>DATA / REFERENCES</span><h2>把教程变成可复现工作流</h2></header>
      <div className="pf-downloads"><a href="/profiling-data/k3-qwen-trace-extract.json" download><b>k3-qwen-trace-extract.json</b><span>两个真实 kernel，各 5 条 Chrome Trace 事件</span></a><a href="/profiling-data/k3-qwen-selected-kernels.csv" download><b>k3-qwen-selected-kernels.csv</b><span>current 与 baseline 的聚合值及证据限制</span></a><a href="/profiling-data/mul-add-kernel-details.csv" download><b>mul-add-kernel-details.csv</b><span>同 shape 5 次 active 的完整字段样例</span></a><a href="/profiling-data/mul-add-trace-view.json" download><b>mul-add-trace-view.json</b><span>可直接导入 Chrome Trace 的单算子事件</span></a></div>
      <div className="pf-links"><a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0002.html" target="_blank" rel="noreferrer"><b>昇腾 CANN Profiling 指南</b><span>官方板端采集与分析入口 ↗</span></a><a href="https://triton-ascend.readthedocs.io/en/latest/debug_guide/profiling.html" target="_blank" rel="noreferrer"><b>Triton Ascend · Profiling</b><span>msprof op、simulator、trace.json ↗</span></a><a href="https://triton-ascend.readthedocs.io/en/latest/index.html" target="_blank" rel="noreferrer"><b>Triton Ascend 文档</b><span>Quick Start、API、调试与优化 ↗</span></a><a href={`https://github.com/sgl-project/sgl-kernel-npu/commit/${CUTOFF}`} target="_blank" rel="noreferrer"><b>可视化源码截止 commit</b><span>{CUTOFF.slice(0, 12)} ↗</span></a></div>
    </section>
  </main>;
}
