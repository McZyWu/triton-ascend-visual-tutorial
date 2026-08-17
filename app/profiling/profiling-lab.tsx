"use client";

import { useMemo, useState } from "react";

const FIELDS = [
  { name: "Name", group: "身份", example: "_situ_deepep_kernel_0", meaning: "设备上真正执行的 kernel 名。先按它 groupby，统计 count、sum、median、P95，不能只看单行。" },
  { name: "Type", group: "身份", example: "kernel / runtime task", meaning: "Profiler 记录的任务类型或算子类型。不同 CANN 版本可能与 Name 相同，也可能是框架层类别。" },
  { name: "Accelerator Core", group: "调度", example: "AI_VECTOR_CORE", meaning: "kernel 跑在 Vector Core 还是 Cube Core。它决定应该优先看 aiv_* 还是 aic_* 字段。" },
  { name: "Start Time(us)", group: "时间", example: "1785933978517390.140", meaning: "设备事件起点。绝对值通常很大；分析间隔时用相邻 start 做差，做图时可以减去首事件进行 rebasing。" },
  { name: "Duration(us)", group: "时间", example: "10.44 μs median", meaning: "单次 NPU kernel 从开始到结束的事件时长，是同条件性能对比的主指标。先预热，再看多次分布。" },
  { name: "Wait Time(us)", group: "时间", example: "0.95 μs median", meaning: "任务进入执行前的等待/排队信息，不等于 kernel 内部 stall，也不能替代 Duration。" },
  { name: "Block Num", group: "调度", example: "40（标准 JIT）/ 0（部分自定义核）", meaning: "Profiler 能识别时表示发射的 block/program 数。自定义二进制显示 0 不代表没有并行 program，而是该字段未被导出。" },
  { name: "Input/Output Shapes", group: "张量", example: "\"2048,768;2048,384\"", meaning: "按分号分隔多个张量的 shape，用来还原最低 GM 字节量。自定义 kernel 可能为空，此时必须回到调用参数和源码。" },
  { name: "Input/Output Data Types", group: "张量", example: "DT_BF16", meaning: "结合 shape 计算字节：BF16/FP16=2 B，FP32=4 B，INT8=1 B。输出同样要计入 compulsory traffic。" },
  { name: "Input/Output Formats", group: "张量", example: "ND / NZ", meaning: "物理存储格式。相同逻辑 shape 在 ND、NZ 下可能产生不同搬运连续性和格式转换成本。" },
  { name: "aiv_time(us)", group: "流水", example: "Vector Core activity scope", meaning: "AIV 活动窗口；不是 Duration 的同义词。多个子流水可以在这个窗口内重叠。" },
  { name: "aiv_vec_time / ratio", group: "流水", example: "22.55%", meaning: "Vector 流水的活动时间/占比，覆盖逐元素算术、激活、部分归约等向量指令。" },
  { name: "aiv_scalar_time / ratio", group: "流水", example: "10.70%", meaning: "标量地址计算、循环和控制流活动。占比高时检查 offset、分支、过小 tile 与循环次数。" },
  { name: "aiv_mte2_time / ratio", group: "流水", example: "29.10%", meaning: "通常对应 GM→UB 的输入搬运活动。占比高要继续核查实际带宽、连续访问、tile 和是否可与计算重叠。" },
  { name: "aiv_mte3_time / ratio", group: "流水", example: "14.30%", meaning: "通常对应 UB→GM 的输出写回活动。与 MTE2、Vector、Scalar 可以重叠，四项不能直接相加。" },
  { name: "aic_* / cube_utilization", group: "流水", example: "MAC / MTE1 / FIXPIPE", meaning: "Cube kernel 的矩阵计算、L1 搬运、写回与利用率字段。Vector kernel 中为 0 是正常现象。" },
];

const TRACE_EVENTS = [
  { name: "split_qkv_rmsnorm_rope", stream: 80, start: 0, duration: 15.2, color: "var(--cyan)" },
  { name: "split_qkv_rmsnorm_rope", stream: 80, start: 28, duration: 17.0, color: "var(--cyan)" },
  { name: "split_qkv_rmsnorm_rope", stream: 80, start: 60, duration: 19.08, color: "var(--cyan)" },
  { name: "_situ_deepep", stream: 83, start: 8, duration: 21.3, color: "var(--acid)" },
  { name: "_situ_deepep", stream: 83, start: 45, duration: 11.9, color: "var(--acid)" },
  { name: "_situ_deepep", stream: 83, start: 72, duration: 10.62, color: "var(--acid)" },
];

export function FieldExplorer() {
  const [selected, setSelected] = useState(4);
  const field = FIELDS[selected];
  return <div className="field-explorer">
    <div className="field-list" role="list" aria-label="kernel_details.csv 字段">
      {FIELDS.map((item, index) => <button key={item.name} onClick={() => setSelected(index)} className={selected === index ? "active" : ""}>
        <span>{item.group}</span>{item.name}
      </button>)}
    </div>
    <article className="field-detail" aria-live="polite">
      <span>SELECTED FIELD · {field.group}</span>
      <h3>{field.name}</h3>
      <code>{field.example}</code>
      <p>{field.meaning}</p>
      <small>{selected + 1} / {FIELDS.length} · 点击左侧字段逐个查看</small>
    </article>
  </div>;
}

export function TraceViewer() {
  const [zoom, setZoom] = useState(1);
  const [selected, setSelected] = useState(0);
  const event = TRACE_EVENTS[selected];
  return <div className="trace-demo">
    <div className="trace-toolbar"><b>TRACE VIEWER · 教学缩放</b><label>zoom <input type="range" min="1" max="3" step="0.25" value={zoom} onChange={e => setZoom(Number(e.target.value))} /></label><code>{zoom.toFixed(2)}×</code></div>
    <div className="trace-scroll"><div className="trace-canvas" style={{ width: `${Math.max(100, zoom * 100)}%` }}>
      {[80, 83].map(stream => <div className="trace-lane" key={stream}><b>Stream {stream}</b><div>{TRACE_EVENTS.map((item, index) => item.stream === stream && <button key={`${stream}-${index}`} className={selected === index ? "selected" : ""} style={{ left: `${item.start}%`, width: `${Math.max(5, item.duration / 1.2)}%`, background: item.color }} onClick={() => setSelected(index)} title={`${item.name}: ${item.duration} μs`}><span>{item.duration} μs</span></button>)}</div></div>)}
      <div className="trace-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100 μs（为教学并排；原始时间戳保留在下载文件）</span></div>
    </div></div>
    <div className="trace-inspector"><span>选中事件</span><b>{event.name}</b><code>stream={event.stream} · duration={event.duration} μs</code><p>Chrome 中点击事件后，右下角会显示 name、ts、dur、Task Id、connection_id 等参数。</p></div>
  </div>;
}

export function BoundCalculator() {
  const [bytes, setBytes] = useState(1179648);
  const [ops, setOps] = useState(393216);
  const [duration, setDuration] = useState(3.24);
  const [bandwidth, setBandwidth] = useState(1600);
  const [peak, setPeak] = useState(40);
  const result = useMemo(() => {
    const memoryUs = bytes / (bandwidth * 1000);
    const computeUs = ops / (peak * 1_000_000);
    const intensity = ops / bytes;
    const ridge = peak * 1000 / bandwidth;
    return {
      memoryUs,
      computeUs,
      intensity,
      ridge,
      achievedBw: bytes / (duration * 1000),
      achievedTops: ops / (duration * 1_000_000),
      bound: intensity < ridge ? "MEMORY-SIDE" : "COMPUTE-SIDE",
    };
  }, [bytes, ops, duration, bandwidth, peak]);
  return <div className="bound-lab">
    <div className="bound-inputs">
      <label>最低 GM 字节量<input type="number" min="1" value={bytes} onChange={e => setBytes(Number(e.target.value))} /><small>input + output；不要把 UB 临时量算进去</small></label>
      <label>估算运算量 OPS<input type="number" min="1" value={ops} onChange={e => setOps(Number(e.target.value))} /><small>乘、加按项目约定计数，并保持前后一致</small></label>
      <label>实测 Duration μs<input type="number" min="0.001" step="0.01" value={duration} onChange={e => setDuration(Number(e.target.value))} /></label>
      <label>设备带宽 GB/s<input type="number" min="1" value={bandwidth} onChange={e => setBandwidth(Number(e.target.value))} /></label>
      <label>对应精度峰值 TOPS<input type="number" min="0.1" step="0.1" value={peak} onChange={e => setPeak(Number(e.target.value))} /></label>
    </div>
    <div className="roofline-card">
      <span>ROOFLINE FIRST PASS</span><h3>{result.bound}</h3>
      <div className="roofline-plot"><i style={{ left: `${Math.min(94, Math.max(4, result.intensity / Math.max(result.ridge, .001) * 50))}%` }} /><b style={{ left: "50%" }}>ridge</b><small>memory</small><small>compute</small></div>
      <dl><div><dt>理论搬运下限</dt><dd>{result.memoryUs.toFixed(3)} μs</dd></div><div><dt>理论计算下限</dt><dd>{result.computeUs.toFixed(3)} μs</dd></div><div><dt>算术强度</dt><dd>{result.intensity.toFixed(3)} ops/B</dd></div><div><dt>ridge point</dt><dd>{result.ridge.toFixed(3)} ops/B</dd></div><div><dt>有效带宽下界</dt><dd>{result.achievedBw.toFixed(1)} GB/s</dd></div><div><dt>实际吞吐下界</dt><dd>{result.achievedTops.toFixed(4)} TOPS</dd></div></dl>
      <p>Roofline 是第一层判断；最终还要与 MTE2/MTE3、Vector/Scalar/Cube 占比和 trace 重叠情况交叉验证。</p>
    </div>
  </div>;
}
