import Link from "next/link";
import Image from "next/image";
import "./profiling.css";

const CUTOFF = "2a87cda308d3c5e38c6a0da74c2146146efbe625";
const RUNS = [
  { run:1, duration:"3.240", aiv:"1.586", vector:"0.413", scalar:"0.765", mte2:"0.263", mte3:"0.207" },
  { run:2, duration:"3.140", aiv:"1.455", vector:"0.413", scalar:"0.661", mte2:"0.240", mte3:"0.198" },
  { run:3, duration:"3.100", aiv:"1.469", vector:"0.413", scalar:"0.687", mte2:"0.229", mte3:"0.193" },
  { run:4, duration:"3.520", aiv:"1.722", vector:"0.413", scalar:"0.878", mte2:"0.286", mte3:"0.209" },
  { run:5, duration:"3.320", aiv:"1.575", vector:"0.413", scalar:"0.768", mte2:"0.266", mte3:"0.192" },
];

export default function ProfilingPage() {
  return <main className="profile-page">
    <header className="pf-topbar"><Link href="/kernel-lab?op=muladd">← Production Kernel Lab</Link><div><b>ASCEND PROFILING FIELD GUIDE</b><span>实测数据、UB 证据与 trace 阅读方法</span></div><a href={`https://github.com/sgl-project/sgl-kernel-npu/commit/${CUTOFF}`} target="_blank" rel="noreferrer">source 2a87cda ↗</a></header>

    <section className="pf-hero">
      <div><span>NEW / PROFILING</span><h1>从一个 kernel 的<br/><em>3.24 µs</em> 看懂流水</h1><p>不是只给一张时间图：这里把 209 板端采集条件、<code>kernel_details.csv</code> 字段、MTE 搬运时间、Vector/Scalar 计算时间、UB 逐项计算和 Chrome Trace 阅读方法放在同一条证据链上。</p><div className="pf-actions"><a href="/profiling-data/mul-add-kernel-details.csv" download>下载 kernel_details.csv</a><a href="/profiling-data/mul-add-trace-view.json" download>下载 trace_view.json</a></div></div>
      <aside><span>CAPTURE ID</span><b>209 / mul-add-b48-h4096</b><dl><div><dt>kernel</dt><dd>mul_add_kernel</dd></div><div><dt>shape</dt><dd>[48, 4096] × 2</dd></div><div><dt>dtype</dt><dd>BF16 · factor=0.5</dd></div><div><dt>core</dt><dd>AI_VECTOR_CORE</dd></div><div><dt>iterations</dt><dd>5 active · warmed up</dd></div><div><dt>correctness</dt><dd>max abs = 0.0</dd></div></dl></aside>
    </section>

    <section className="pf-section pf-evidence">
      <header><span>01 / SOURCE IDENTITY</span><h2>可视化、源码与实测是同一版本</h2><p>Production Kernel Lab 固定到完整 commit，不再跟随会变化的 <code>main</code>。209 容器内安装文件与该 commit 的 <code>moe/mul_add.py</code> Git blob 都是 <code>f9f172c…</code>。</p></header>
      <div className="pf-proof-grid"><article><span>VISUAL CUTOFF</span><b>2a87cda</b><code>release A5 image (#683)</code></article><article><span>PROFILED SOURCE BLOB</span><b>f9f172c…</b><code>installed == cutoff</code></article><article><span>GRID FROM CSV</span><b>40 programs</b><code>Block Num = 40</code></article><article><span>COMPILED LOCAL MEMORY</span><b>32 KiB</b><code>.stack record = 0x8000</code></article></div>
    </section>

    <section className="pf-section pf-dark">
      <header><span>02 / REAL TIMELINE</span><h2>209 实测时间图</h2><p>上半部分是 5 次 NPU kernel duration；下半部分是中位数流水活动。Scalar、Vector、MTE2、MTE3 是可重叠的独立流水，不可直接相加。</p></header>
      <figure className="pf-timeline"><Image unoptimized src="/profiling-data/mul-add-pipeline-timeline.png" width={1600} height={900} sizes="(max-width: 900px) 100vw, 1320px" alt="209 上 mul_add_kernel 五次执行时长与 Scalar、Vector、MTE2、MTE3 中位时间图"/><figcaption>由同一次 209 实测的 <code>kernel_details.csv</code> 与 <code>trace_view.json</code> 重绘，数值没有人工拟造；这是可读性更高的数据图，不冒充 Chrome UI 截图。</figcaption></figure>
      <div className="pf-metrics"><article><span>MEDIAN DURATION</span><b>3.240 <i>µs</i></b><small>min 3.100 · max 3.520</small></article><article><span>MEDIAN AIV ACTIVE</span><b>1.575 <i>µs</i></b><small>Vector core activity scope</small></article><article><span>MEDIAN MTE2</span><b>0.263 <i>µs</i></b><small>GM → UB / input</small></article><article><span>MEDIAN MTE3</span><b>0.198 <i>µs</i></b><small>UB → GM / output</small></article></div>
    </section>

    <section className="pf-section">
      <header><span>03 / KERNEL_DETAILS.CSV</span><h2>哪些列有用，应该怎样解释</h2><p><code>Duration</code> 是整次 NPU kernel 的事件时长；<code>aiv_*_time</code> 是 AIV 内各条流水的活动时间。两者测量范围不同，流水之间又能重叠，所以不能拿四条流水简单求和去等于 Duration。</p></header>
      <div className="pf-table-wrap"><table><thead><tr><th>run</th><th>Duration</th><th>aiv_time</th><th>Vector</th><th>Scalar</th><th>MTE2 · load</th><th>MTE3 · store</th></tr></thead><tbody>{RUNS.map(row => <tr key={row.run}><td>{row.run}</td><td>{row.duration} µs</td><td>{row.aiv} µs</td><td>{row.vector} µs</td><td>{row.scalar} µs</td><td>{row.mte2} µs</td><td>{row.mte3} µs</td></tr>)}</tbody></table></div>
      <div className="pf-field-notes">
        <article><b>Duration</b><p>先用它比较端到端 kernel 快慢，并看 5 次分布是否稳定。本例中位数 3.240 µs。</p></article>
        <article><b>Vector / Scalar</b><p>Vector 覆盖逐元素乘加等向量路径；Scalar 包含地址、循环、控制等标量活动。本例 Scalar 中位数 0.765 µs。</p></article>
        <article><b>MTE2 / MTE3</b><p>MTE2 观察输入搬入，MTE3 观察结果写回。要结合字节量、连续性、块大小与是否和计算重叠判断瓶颈。</p></article>
        <article><b>Wait Time</b><p>本次 CSV 中约 404–522 µs，主要来自逐 step 同步和 host 间隔，不能解释成 kernel 内部等待；分析单核时不要用它替代 Duration。</p></article>
      </div>
    </section>

    <section className="pf-section pf-ub-proof">
      <header><span>04 / UB CALCULATION</span><h2>每个变量怎样凑成 32 KiB</h2><p>输入 hidden size 是 4096，<code>BLOCK_SIZE = next_power_of_2(4096) = 4096</code>。一个 program 的一轮只处理一个 row task，下一轮复用同一片 UB。</p></header>
      <div className="pf-ub-flow"><article><span>tl.load input1_ptr</span><b>routed_values</b><code>4096 × BF16 2 B = 8192 B</code><strong>8 KiB</strong></article><i>+</i><article><span>tl.load input2_ptr</span><b>shared_values</b><code>4096 × BF16 2 B = 8192 B</code><strong>8 KiB</strong></article><i>+</i><article><span>Vector compute</span><b>buffered_values</b><code>4096 × FP32 4 B = 16384 B</code><strong>16 KiB</strong></article><i>=</i><article className="total"><span>peak working set</span><b>compiled match</b><code>8192 + 8192 + 16384</code><strong>32 KiB</strong></article></div>
      <div className="pf-cycle"><code>第 k 轮：task = pid + k × 40</code><span>取一行 task</span><i>→</i><span>两条 BF16 tile 搬入 UB</span><i>→</i><span>FP32 乘加</span><i>→</i><span>BF16 写回</span><i>→</i><span>UB 复用</span></div>
      <p className="pf-traffic"><b>GM 逻辑流量：</b>每行 8 KiB + 8 KiB + 8 KiB = 24 KiB；48 行共 1,179,648 B = 1.125 MiB。40 个 program 中 P0–P7 各处理两行，其余各处理一行。</p>
      <pre className="pf-code"><code>{`row_start = tl.program_id(0)        # pid: 0..39
row_step  = tl.num_programs(0)      # P = 40
cols = tl.arange(0, BLOCK_SIZE)     # lane: 0..4095
for row_idx in tl.range(row_start, batch_size, row_step):
    routed_values = tl.load(input1_ptr + input_offsets, mask=valid_mask)
    shared_values = tl.load(input2_ptr + input_offsets, mask=valid_mask)
    buffered_values = routed_values * factor + shared_values
    tl.store(output_ptr + row_idx * hidden_size + cols, buffered_values, mask=valid_mask)`}</code></pre>
      <Link className="pf-deep-link" href="/kernel-lab?op=muladd&kernel=mul_add_kernel">打开逐段源码与搬运模拟 ↗</Link>
    </section>

    <section className="pf-section pf-dark">
      <header><span>05 / CAPTURE WORKFLOW</span><h2>从采集到定位的完整路径</h2><p>板端统计用于找到慢 kernel 和流水占比；仿真 trace 用于进一步观察指令级流水、源码行关联和重叠情况。</p></header>
      <ol className="pf-steps"><li><b>01</b><div><strong>隔离单算子并预热</strong><p>固定 shape、dtype、factor 和设备；先预热编译缓存，再只包围待测调用。每轮都做正确性检查。</p></div></li><li><b>02</b><div><strong>torch_npu profiler 采集</strong><pre><code>{`experimental_config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=ProfilerLevel.Level1,
    aic_metrics=AiCMetrics.PipeUtilization,
)
with torch_npu.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    experimental_config=experimental_config,
    on_trace_ready=tensorboard_trace_handler(output_dir),
) as prof:
    for _ in range(5):
        mul_add(input1, input2, 0.5)
        torch.npu.synchronize()
        prof.step()`}</code></pre></div></li><li><b>03</b><div><strong>先读 kernel_details.csv</strong><p>按 Name 筛选 kernel，比较 Duration 分布，再看 Block Num、Core Type、Vector/Scalar/MTE2/MTE3。不要把 pipeline time 相加。</p></div></li><li><b>04</b><div><strong>需要源码行与细流水时再跑 msprof op</strong><pre><code>{`export TRITON_DISABLE_LINE_INFO=false
msprof op --kernel-name=mul_add_kernel <your-command>
# simulator 路径会输出 trace.json 与 visualize_data.bin`}</code></pre></div></li></ol>
    </section>

    <section className="pf-section pf-chrome">
      <header><span>06 / CHROME TRACE</span><h2>用 chrome://tracing/ 看时间线</h2><p>Chrome Trace 展示的是事件时间轴；它适合回答“谁先发生、持续多久、是否重叠”，不是 UB 容量计算器。</p></header>
      <div className="pf-chrome-grid"><div><ol><li>在 Chrome 地址栏打开 <code>chrome://tracing/</code>。</li><li>把本页下载的 <code>trace_view.json</code> 拖入窗口。</li><li>搜索 <code>mul_add_kernel</code>，找到 NPU 事件；本次五个事件为 3.10–3.52 µs。</li><li>使用 <kbd>W</kbd>/<kbd>S</kbd> 缩放、<kbd>A</kbd>/<kbd>D</kbd> 左右移动，点击事件读取起止时间和元数据。</li><li>如果分析 simulator 的 <code>trace.json</code>，展开 Vector、Scalar、MTE2、MTE3 轨道，观察搬运与计算重叠。</li></ol><a href="/profiling-data/mul-add-trace-view.json" download>下载可拖入 Chrome 的 trace_view.json</a></div><figure><Image unoptimized src="/profiling-data/mul-add-pipeline-timeline.png" width={1600} height={900} sizes="(max-width: 1050px) 100vw, 58vw" alt="与 Chrome trace 数据一致的 mul_add_kernel 时间图"/><figcaption>当前图由 trace 数据重绘。由于执行环境的 ChatGPT Chrome 扩展与 native host 未安装，无法自动操控 <code>chrome://tracing/</code> 并截取浏览器 UI；数据文件已原样附上，可直接本机打开。</figcaption></figure></div>
    </section>

    <section className="pf-section pf-resources">
      <header><span>07 / FILES & REFERENCES</span><h2>原始数据与官方资料</h2></header>
      <div className="pf-downloads"><a href="/profiling-data/mul-add-kernel-details.csv" download><b>kernel_details.csv</b><span>5 次 kernel 与 PipeUtilization 字段</span></a><a href="/profiling-data/mul-add-trace-view.json" download><b>trace_view.json</b><span>Chrome Trace 事件时间轴</span></a><a href="/profiling-data/mul-add-msprof.json" download><b>msprof.json</b><span>本次 profiler 导出元数据</span></a><a href="/profiling-data/mul-add-pipeline-timeline.png" download><b>timeline.png</b><span>实测数据重绘时间图</span></a></div>
      <div className="pf-links"><a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0002.html" target="_blank" rel="noreferrer"><b>昇腾 CANN Profiling 指南</b><span>官方采集与分析入口 ↗</span></a><a href="https://triton-ascend.readthedocs.io/en/latest/debug_guide/profiling.html" target="_blank" rel="noreferrer"><b>Triton Ascend 性能分析</b><span>msprof op、simulator 与 Chrome Trace ↗</span></a><a href="https://triton-ascend.readthedocs.io/en/latest/debug_guide/ub_overflow.html" target="_blank" rel="noreferrer"><b>UB Overflow 调试</b><span>UB 溢出定位与分块策略 ↗</span></a><a href="https://triton-ascend.readthedocs.io/en/latest/index.html" target="_blank" rel="noreferrer"><b>Triton Ascend 文档首页</b><span>Quick Start、调试与 API ↗</span></a></div>
    </section>
  </main>;
}
