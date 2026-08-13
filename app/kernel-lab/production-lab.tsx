"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { KERNEL_GROUPS, KERNEL_OPS, type KernelOp, TOTAL_TRITON_KERNELS } from "../kernel-ops";
import { KERNEL_SIGNATURES } from "../kernel-signatures";

type ArgKind = "input" | "output" | "inout" | "index" | "stride" | "shape" | "block" | "flag" | "scalar" | "helper";
type Coord = { lane: number; local: number[]; global: number[]; flat: number; valid: boolean };

const STEPS = [
  ["GRID", "启动 Grid"], ["PROGRAM", "program 领任务"], ["OFFSET", "lane 算地址"],
  ["LOAD", "逐变量 load"], ["UB", "UB 工作集"], ["COMPUTE", "片上计算"],
  ["STORE", "逐变量 store"], ["TORCH", "Torch 对照"],
] as const;

const SOURCE_ROOT = "https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu";

function outputArg(name: string) {
  const n = name.toLowerCase();
  if (/^(out|o|y|output|att_out|gated_output|zero_result|accept_index|accept_token_num|predicts|argmax|prob|v_new|ht|ad|ai)$/.test(n)) return true;
  if (/(^|_)(out|output|dst|result|mean|rstd|variance|candidate_scores|candidate_indices|topk_indices|mixed_qkvz|mixed_ba)(_|$)/.test(n)) return true;
  return /^(q_ptr|k_ptr|v_ptr|gate_ptr|scale_ptr)$/.test(n) && false;
}

function pointerArg(name: string) {
  const n = name.toLowerCase();
  if (/(stride|block|size|num_|^num|dim|head|batch|total|scale$|eps|beta$|threshold|limit|width|len$|^t$|^h$|^k$|^v$|^b$)/.test(n) && !n.endsWith("_ptr")) return false;
  return n.endsWith("_ptr") || /(^|_)(cache|buffer|table|indices|index|lens|offsets)$/.test(n) || ["q","k","v","w","x","y","o","g","a","b","h","h0","ht","beta","query","sinks","predicts","candidates","target_predict","hidden_states","gated_output"].includes(n);
}

function classifyArg(name: string, loads: string[] = [], stores: string[] = []): ArgKind {
  const n = name.toLowerCase();
  const isPointer = pointerArg(name);
  if (isPointer && loads.includes(name) && stores.includes(name)) return "inout";
  if (isPointer && stores.includes(name)) return "output";
  if (isPointer && loads.includes(name) && /(index|indices|table|lens|length|offset|loc|pool|seqlen|cum_seq)/.test(n)) return "index";
  if (isPointer && loads.includes(name)) return "input";
  if (n === "x" && name.length === 1) return "input";
  if (outputArg(name)) return "output";
  if (pointerArg(name) && /(index|indices|table|lens|length|offset|loc|pool|seqlen|cum_seq)/.test(n)) return "index";
  if (pointerArg(name)) return "input";
  if (n.startsWith("stride") || n.includes("_stride")) return "stride";
  if (/^(block|blk|bt$|bk$|bv$|bd$|bhv$|mbs$|nbd$|bd$|topk$|rows_per_iter|dim_block)/.test(n) || n.includes("block_size") || n.endsWith("_block")) return "block";
  if (/^(has_|use_|is_|do_|need_|save_|store_|head_first|reverse|bias$|norms$|run_|guarded|fill_only|prefetch)/.test(n)) return "flag";
  if (name === "_builder") return "helper";
  if (/^(t|b|h|k|v|d|n|m|bs|hv|hg|nb|nt)$/.test(n) || /(size|dim|rows|cols|heads|tokens|batch|seq_len|num_|total|page_size|max_|steps|width|rank)/.test(n)) return "shape";
  return "scalar";
}

const KIND_TEXT: Record<ArgKind, string> = {
  input: "输入张量指针", output: "输出 / 更新目标", inout: "原地读取并更新", index: "索引 / 长度张量", stride: "元素步长",
  shape: "形状 / 循环边界", block: "编译期 Block", flag: "编译期分支", scalar: "标量参数", helper: "编译器 helper",
};

function explainArg(name: string, kind: ArgKind) {
  const n = name.toLowerCase();
  if (kind === "stride") return `${name} 个元素跨到下一维；地址公式中与对应坐标相乘，不是字节数。`;
  if (kind === "block") return `${name} 决定一个 program 当前 tile 的覆盖范围，也影响 lane、mask 与 UB 占用。`;
  if (kind === "flag") return `${name} 在编译期选择代码分支；关闭时相关 load / compute / store 会被消除。`;
  if (kind === "shape") return `${name} 是真实边界或循环次数；lane 坐标与它比较生成 mask。`;
  if (kind === "index") return `${name} 先从 GM 读取索引，再把逻辑 task / page 映射到第二个物理地址。`;
  if (kind === "output") return `${name} 接收当前 program 的有效 lane；store 使用与输出布局对应的 stride 和 mask。`;
  if (kind === "inout") return `${name} 先从 GM load 到片上，计算后再写回同一张量或状态缓存。`;
  if (kind === "input") return `${name} 的基址不等于元素地址；要加上 program 坐标、lane 坐标和 stride 的组合。`;
  if (/scale|alpha|beta/.test(n)) return `${name} 是计算中的缩放系数，load 后通常转成计算精度再广播到 tile。`;
  if (/eps/.test(n)) return `${name} 防止归一化分母为零，通常加在 variance / norm 内。`;
  return `${name} 是 kernel 的运行时标量或编译期常量，参与地址、边界或数学表达式。`;
}

function profileFor(op: KernelOp) {
  const persistent = /persistent/i.test(op.grid);
  if (op.group === "Attention" || op.group === "Indexer") return { labels:["batch / Q-block","head / chunk","D / page lane"], shape:[4,6,16], block:[1,2,8], persistent };
  if (op.group === "FLA / KDA") return { labels:["sequence / chunk","head","K/V lane"], shape:[8,4,16], block:[1,1,8], persistent };
  if (op.group === "Mamba / Cache") return { labels:["request / layer","channel block","window / tail"], shape:[8,8,8], block:[1,2,4], persistent };
  if (op.group === "Norm / RoPE") return { labels:["token / row","head / column group","hidden lane"], shape:[8,4,32], block:[1,1,8], persistent };
  if (op.group === "MoE / Sample") return { labels:["token / request","expert / head","hidden / top-k"], shape:[8,4,16], block:[1,1,8], persistent };
  return { labels:["row / token","expert / group","hidden lane"], shape:[8,2,32], block:[1,1,8], persistent };
}

function stableNumber(text: string) {
  let h = 17;
  for (const c of text) h = (h * 31 + c.charCodeAt(0)) % 997;
  return h;
}

function sampleValue(name: string, kind: ArgKind, shape: number[], block: number[], lane = 0) {
  const n = name.toLowerCase();
  if (kind === "flag") return "true";
  if (kind === "stride") {
    if (/(_d|dim|col|token)$/.test(n)) return "1 element";
    if (/(_h|head|window)$/.test(n)) return `${shape[2]} elements`;
    return `${shape[1] * shape[2]} elements`;
  }
  if (kind === "block") return String(/(_d|_n|_c|hidden|col|tail|block$)/.test(n) ? block[2] : block[0]);
  if (kind === "shape") {
    if (/batch|rows|tokens|^b$|^m$/.test(n)) return String(shape[0]);
    if (/head|^h$/.test(n)) return String(shape[1]);
    return String(shape[2]);
  }
  if (kind === "scalar") return /eps/.test(n) ? "1e-6" : /scale|alpha|beta/.test(n) ? "0.125" : /limit|threshold/.test(n) ? "7.0" : "1";
  if (kind === "helper") return "compiler supplied";
  const raw = ((stableNumber(name) % 19) - 9) / 4 + lane * .125;
  return raw.toFixed(3);
}

function addressFormula(name: string, kind: ArgKind, coord: Coord) {
  if (kind === "index") return `${name}[task] → physical=${(coord.flat * 7 + 3) % 23} → base + physical × row_stride + lane`;
  if (kind === "output" || kind === "inout") return `${name} + (${coord.global[0]}×stride₀ + ${coord.global[1]}×stride₁ + ${coord.global[2]})`;
  return `${name} + pid·BLOCK + lane = base + ${coord.flat} elements`;
}

function splitCompute(text: string) {
  return text.split(/；|→/).map(x => x.trim()).filter(Boolean);
}

function KernelProductionLab() {
  const params = useSearchParams();
  const initialId = params.get("op") || KERNEL_OPS[0].id;
  const initialOp = KERNEL_OPS.find(x => x.id === initialId) ?? KERNEL_OPS[0];
  const [selectedId, setSelectedId] = useState(initialOp.id);
  const [kernelIndex, setKernelIndex] = useState(() => Math.max(0, initialOp.kernels.indexOf(params.get("kernel") || "")));
  const [group, setGroup] = useState("全部");
  const [query, setQuery] = useState("");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [round, setRound] = useState(0);
  const selected = KERNEL_OPS.find(x => x.id === selectedId) ?? KERNEL_OPS[0];
  const [shape, setShape] = useState(() => profileFor(selected).shape);
  const [block, setBlock] = useState(() => profileFor(selected).block);
  const [pid, setPid] = useState([0,0,0]);
  const [variable, setVariable] = useState("");
  const kernel = selected.kernels[Math.min(kernelIndex, selected.kernels.length - 1)];
  const signature = KERNEL_SIGNATURES[selected.module]?.[kernel];
  const args = useMemo(() => signature?.args ?? [], [signature]);
  const profile = profileFor(selected);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setStep(s => s === STEPS.length - 1 ? 0 : s + 1), 1100);
    return () => window.clearInterval(timer);
  }, [playing]);

  const argRows = useMemo(() => args.map(name => ({ name, kind: classifyArg(name, signature?.loads, signature?.stores), loaded:(signature?.loads.includes(name) ?? false) && pointerArg(name), stored:(signature?.stores.includes(name) ?? false) && pointerArg(name) })), [args, signature]);
  const pointerRows = argRows.filter(x => ["input","output","inout","index"].includes(x.kind));
  const inputs = pointerRows.filter(x => x.kind !== "output");
  const outputs = pointerRows.filter(x => x.kind === "output" || x.kind === "inout");
  const selectedVar = argRows.find(x => x.name === variable) ?? pointerRows[0] ?? argRows[0];
  const grid = shape.map((n,i) => Math.max(1, Math.ceil(n / Math.max(1, block[i]))));
  const logicalTasks = grid[0] * grid[1] * grid[2];
  const physicalPrograms = profile.persistent ? Math.min(8, logicalTasks) : logicalTasks;
  const directFlatPid = (pid[0] * grid[1] + pid[1]) * grid[2] + pid[2];
  const physicalPid = profile.persistent ? pid[0] : directFlatPid;
  const task = profile.persistent ? physicalPid + round * physicalPrograms : directFlatPid;
  const taskPid = profile.persistent ? [Math.floor(task / (grid[1] * grid[2])) % grid[0], Math.floor(task / grid[2]) % grid[1], task % grid[2]] : pid;
  const tileElements = block[0] * block[1] * block[2];
  const shownLanes = Math.min(32, tileElements);
  const coords: Coord[] = Array.from({ length: shownLanes }, (_, lane) => {
    const l2 = lane % block[2];
    const l1 = Math.floor(lane / block[2]) % block[1];
    const l0 = Math.floor(lane / (block[2] * block[1]));
    const local = [l0,l1,l2];
    const global = local.map((v,i) => taskPid[i] * block[i] + v);
    const valid = global.every((v,i) => v < shape[i]);
    return { lane, local, global, valid, flat:(global[0] * shape[1] + global[1]) * shape[2] + global[2] };
  });
  const activeCoord = coords.find(x => x.valid) ?? coords[0];
  const workBytes = inputs.length * tileElements * 2 + Math.max(1, outputs.length) * tileElements * 4;
  const sourceUrl = `${SOURCE_ROOT}/${selected.module}#L${signature?.line ?? 1}`;
  const visible = KERNEL_OPS.filter(op => (group === "全部" || op.group === group) && `${op.title} ${op.module} ${op.kernels.join(" ")}`.toLowerCase().includes(query.toLowerCase()));

  const chooseOp = (op: KernelOp) => {
    const p = profileFor(op);
    setSelectedId(op.id); setKernelIndex(0); setShape(p.shape); setBlock(p.block); setPid([0,0,0]); setRound(0); setStep(0); setPlaying(false); setVariable("");
    window.history.replaceState(null, "", `/kernel-lab?op=${encodeURIComponent(op.id)}`);
  };
  const changeAxis = (setter: (v: number[]) => void, values: number[], axis: number, value: number) => setter(values.map((x,i) => i === axis ? Math.max(1,value) : x));
  const displayPrograms = Array.from({ length: Math.min(48, profile.persistent ? physicalPrograms : logicalTasks) }, (_, flat) => {
    if (profile.persistent) return { flat, p:[flat,0,0] };
    return { flat, p:[Math.floor(flat/(grid[1]*grid[2])), Math.floor(flat/grid[2])%grid[1], flat%grid[2]] };
  });

  return <main className="production-lab-page">
    <header className="pl-topbar">
      <Link href="/#cases">← 返回教程</Link>
      <div><b>Triton Ascend · Production Kernel Lab</b><span>{KERNEL_OPS.length} modules / {TOTAL_TRITON_KERNELS} JIT kernels</span></div>
      <a href={sourceUrl} target="_blank" rel="noreferrer">当前源码 ↗</a>
    </header>

    <aside className="pl-sidebar">
      <div className="pl-search"><label>搜索真实算子或 kernel<input value={query} onChange={e => setQuery(e.target.value)} placeholder="rmsnorm / cache / decode" /></label></div>
      <div className="pl-groups">{KERNEL_GROUPS.map(g => <button key={g} className={group === g ? "active" : ""} onClick={() => setGroup(g)}>{g}</button>)}</div>
      <div className="pl-op-list">{visible.map(op => <button key={op.id} className={selected.id === op.id ? "active" : ""} onClick={() => chooseOp(op)}><small>{op.module}</small><b>{op.title}</b><span>{op.kernels.length} JIT</span></button>)}</div>
    </aside>

    <div className="pl-main">
      <section className="pl-title">
        <div><span>{selected.group} · {selected.module}</span><h1>{selected.title}</h1><p>{selected.input}</p></div>
        <div className="pl-title-actions"><a href={sourceUrl} target="_blank" rel="noreferrer">在 GitHub 打开该 kernel ↗</a><code>{profile.persistent ? "PERSISTENT GRID" : "DIRECT GRID"}</code></div>
      </section>

      <section className="pl-kernel-picker"><b>本模块 JIT kernel</b>{selected.kernels.map((name,i) => <button key={name} className={i === kernelIndex ? "active" : ""} onClick={() => { setKernelIndex(i); setStep(0); setVariable(""); window.history.replaceState(null,"",`/kernel-lab?op=${selected.id}&kernel=${encodeURIComponent(name)}`); }}>{name}</button>)}</section>

      <section className="pl-signature"><span>SOURCE SIGNATURE · line {signature?.line ?? "?"}</span><code>def {kernel}({args.join(", ")})</code><p>下面的变量清单直接来自该 <code>@triton.jit</code> 函数签名；点击变量可追踪它在当前 lane 的地址和值。</p></section>

      <section className="pl-controls">
        <div><span>教学输入 shape</span>{profile.labels.map((label,i) => <label key={label}>{label}<input type="number" min="1" value={shape[i]} onChange={e => changeAxis(setShape,shape,i,+e.target.value)} /></label>)}</div>
        <div><span>当前 program 的 BLOCK / tile</span>{profile.labels.map((label,i) => <label key={label}>BLOCK_{i}<input type="number" min="1" value={block[i]} onChange={e => changeAxis(setBlock,block,i,+e.target.value)} /></label>)}</div>
        <div className="pl-derived"><p><span>源码 Grid</span><code>{selected.grid}</code></p><p><span>教学展开</span><code>grid = ({grid.join(", ")}) · {logicalTasks} tasks</code></p><p><span>当前 Tile</span><code>{block.join(" × ")} = {tileElements} lanes</code></p></div>
      </section>

      <nav className="pl-steps" aria-label="模拟步骤">{STEPS.map(([key,label],i) => <button key={key} className={step === i ? "active" : step > i ? "done" : ""} onClick={() => {setStep(i);setPlaying(false)}}><span>{String(i+1).padStart(2,"0")}</span><b>{key}</b><small>{label}</small></button>)}</nav>

      <section className="pl-step-summary"><div><span>当前阶段</span><strong>{STEPS[step][1]}</strong></div><p>{[selected.grid, `program ${taskPid.join(",")} 正在处理 task ${task}`, `lane → local → global → flat offset；尾块由 mask 保护`, selected.load, `${selected.tile}；当前估算 ${(workBytes/1024).toFixed(2)} KiB`, selected.compute, selected.store, selected.torch][step]}</p><div className="pl-play"><button onClick={() => setStep(Math.max(0,step-1))} disabled={step===0}>←</button><button className="play" onClick={() => setPlaying(!playing)}>{playing ? "暂停" : "自动播放"}</button><button onClick={() => setStep(Math.min(STEPS.length-1,step+1))} disabled={step===STEPS.length-1}>→</button></div></section>

      <section className={`pl-grid-panel ${step <= 1 ? "focus" : ""}`}>
        <header><div><span>GRID / PROGRAM MAP</span><h2>Grid 怎样拆成并行 task</h2></div><p>{profile.persistent ? <><code>task = pid + k × P</code>；P={physicalPrograms}，当前 k={round}，所以 task={physicalPid}+{round}×{physicalPrograms}={task}。</> : <>每个 program 直接领取一个逻辑 tile；program 坐标乘 BLOCK 得到 tile 起点。</>}</p></header>
        {profile.persistent && <label className="pl-round">persistent 轮次 k <input type="range" min="0" max={Math.max(0,Math.ceil(logicalTasks/physicalPrograms)-1)} value={round} onChange={e => setRound(+e.target.value)} /><b>{round}</b></label>}
        <div className="pl-program-map">{displayPrograms.map(item => {
          const assignedTask = item.flat + round * physicalPrograms;
          const assignedPid = [Math.floor(assignedTask / (grid[1] * grid[2])) % grid[0], Math.floor(assignedTask / grid[2]) % grid[1], assignedTask % grid[2]];
          const idle = profile.persistent && assignedTask >= logicalTasks;
          return <button key={item.flat} className={`${profile.persistent ? physicalPid === item.flat ? "active" : "" : pid.every((x,i)=>x===item.p[i]) ? "active" : ""} ${idle ? "idle" : ""}`} onClick={() => !idle && setPid(item.p)} disabled={idle}><b>P{item.flat}</b><span>{profile.persistent ? idle ? "idle" : `task ${assignedTask}` : `[${item.p.join(",")}]`}</span><small>{idle ? "本轮没有剩余 task" : `origin [${(profile.persistent ? assignedPid : item.p).map((x,i)=>x*block[i]).join(",")}]`}</small></button>;
        })}</div>
        {!profile.persistent && logicalTasks > 48 && <p className="pl-clipped">只绘制前 48 个 program；完整 Grid 共 {logicalTasks} 个，计算规则相同。</p>}
      </section>

      <section className={`pl-offset-panel ${step === 2 ? "focus" : ""}`}>
        <header><div><span>BLOCK / LANE ADDRESSING</span><h2>当前 program 内每个 lane 的坐标</h2></div><code>global[d] = pid[d] × BLOCK[d] + local[d]</code></header>
        <div className="pl-offset-formulas"><p><span>program 坐标</span><b>[{taskPid.join(", ")}]</b></p><p><span>tile 起点</span><b>[{taskPid.map((x,i)=>x*block[i]).join(", ")}]</b></p><p><span>展平公式</span><b>(g₀ × shape₁ + g₁) × shape₂ + g₂</b></p><p><span>mask</span><b>g₀&lt;S₀ &amp;&amp; g₁&lt;S₁ &amp;&amp; g₂&lt;S₂</b></p></div>
        <div className="pl-lane-table"><div><b>LANE</b><b>LOCAL COORD</b><b>GLOBAL COORD</b><b>FLAT OFFSET</b><b>MASK</b></div>{coords.map(c => <div key={c.lane} className={c.valid ? "valid" : "masked"}><code>{c.lane}</code><code>[{c.local.join(",")}]</code><code>[{c.global.join(",")}]</code><strong>{c.flat}</strong><b>{String(c.valid).toUpperCase()}</b></div>)}</div>
        {tileElements > shownLanes && <p className="pl-clipped">当前 tile 共 {tileElements} lanes，为保持可读性展示前 {shownLanes} 个；其余 lane 按同一公式继续展开。</p>}
      </section>

      <section className={`pl-variable-panel ${step === 3 || step === 6 ? "focus" : ""}`}>
        <header><div><span>ALL KERNEL ARGUMENTS</span><h2>每个变量在 load / compute / store 中的作用</h2></div><p><b>{args.length}</b> 个签名变量 · 指针 {pointerRows.length} · 标量/形状/stride {args.length-pointerRows.length}</p></header>
        <div className="pl-variable-grid">{argRows.map((arg,i) => <button key={`${arg.name}-${i}`} className={`${arg.kind} ${selectedVar?.name === arg.name ? "active" : ""}`} onClick={() => setVariable(arg.name)}><span>{String(i+1).padStart(2,"0")} · {KIND_TEXT[arg.kind]} · {arg.loaded ? "tl.load" : "no load"}{arg.stored ? " + tl.store" : ""}</span><code>{arg.name}</code><b>{sampleValue(arg.name,arg.kind,shape,block)}</b><p>{explainArg(arg.name,arg.kind)}</p></button>)}</div>
        {selectedVar && <div className="pl-variable-trace"><div><span>选中变量</span><code>{selectedVar.name}</code><b>{KIND_TEXT[selectedVar.kind]}</b></div><p><span>当前 lane 地址</span><code>{addressFormula(selectedVar.name,selectedVar.kind,activeCoord)}</code></p><p><span>load 值</span><code>{sampleValue(selectedVar.name,selectedVar.kind,shape,block,activeCoord.lane)}</code></p><p><span>进入计算</span><code>{selected.compute}</code></p><p><span>store</span><code>{selectedVar.kind === "output" || selectedVar.kind === "inout" ? selected.store : "此变量只读；结果写到输出变量"}</code></p></div>}
      </section>

      <section className={`pl-memory-panel ${step >= 3 && step <= 6 ? "focus" : ""}`}>
        <header><div><span>GM → UB / REG → GM</span><h2>真实变量搬运与计算流水</h2></div><p>蓝色是 load，黄绿色是片上存活值，橙色是 store。每列都是当前 program 的一轮，不乘整个 Grid。</p></header>
        <div className="pl-memory-flow">
          <div className={`pl-mem-column gm ${step===3 ? "active" : ""}`}><span>GLOBAL MEMORY · INPUT</span>{inputs.map((arg,i)=><button key={`${arg.name}-${i}`} onClick={()=>setVariable(arg.name)} className={selectedVar?.name===arg.name?"selected":""}><code>{arg.name}</code><small>{arg.kind === "index" ? "先取索引，再二次寻址" : addressFormula(arg.name,arg.kind,activeCoord)}</small><b>{coords.slice(0,8).map(c=>c.valid?sampleValue(arg.name,arg.kind,shape,block,c.lane):"×").join("  ")}</b></button>)}</div>
          <div className={`pl-flow-arrow load ${step===3 ? "active" : ""}`}><b>tl.load</b><span>mask + other</span><i>→</i></div>
          <div className={`pl-mem-column ub ${step===4 || step===5 ? "active" : ""}`}><span>UNIFIED BUFFER / REG</span>{inputs.map((arg,i)=><div key={`${arg.name}-${i}`}><code>{arg.name}_tile</code><small>{tileElements} elements · fp16/bf16≈{(tileElements*2/1024).toFixed(2)} KiB</small></div>)}<div className="acc"><code>acc / temporary</code><small>{tileElements} elements · fp32≈{(tileElements*4/1024).toFixed(2)} KiB</small></div><strong>教学峰值 ≈ {(workBytes/1024).toFixed(2)} KiB</strong></div>
          <div className={`pl-compute ${step===5 ? "active" : ""}`}><span>COMPUTE</span>{splitCompute(selected.compute).map((x,i)=><div key={`${x}-${i}`}><b>{i+1}</b><code>{x}</code></div>)}</div>
          <div className={`pl-flow-arrow store ${step===6 ? "active" : ""}`}><b>tl.store</b><span>same mask</span><i>→</i></div>
          <div className={`pl-mem-column out ${step===6 ? "active" : ""}`}><span>GLOBAL MEMORY · OUTPUT / STATE</span>{(outputs.length ? outputs : [{name:"return / compiler value",kind:"output" as ArgKind}]).map((arg,i)=><button key={`${arg.name}-${i}`} onClick={()=>setVariable(arg.name)}><code>{arg.name}</code><small>{addressFormula(arg.name,arg.kind,activeCoord)}</small><b>{coords.slice(0,8).map(c=>c.valid?`y${c.lane}`:"skip").join("  ")}</b></button>)}</div>
        </div>
        <div className="pl-ub-ledger"><b>UB 逐项估算</b><code>{inputs.length} 个输入 tile × {tileElements} × 2 B + {Math.max(1,outputs.length)} 个输出/acc tile × {tileElements} × 4 B = {workBytes} B</code><span>这是按签名和教学 dtype 的显式工作集；实际编译器还会改变生命周期、对齐、复用与双缓冲。</span></div>
        <div className="pl-source-trace">
          <div><span>源码中的 tl.load</span>{signature?.loadExprs.length ? signature.loadExprs.map((expr,i)=><code key={`${expr}-${i}`}>{expr}</code>) : <code>这个 helper 没有直接 tl.load；值由调用方或参数传入。</code>}</div>
          <div><span>源码中的关键赋值 / 计算</span>{signature?.computeExprs.length ? signature.computeExprs.map((expr,i)=><code key={`${expr}-${i}`}>{expr}</code>) : <code>{selected.compute}</code>}</div>
          <div><span>源码中的 tl.store / atomic</span>{signature?.storeExprs.length ? signature.storeExprs.map((expr,i)=><code key={`${expr}-${i}`}>{expr}</code>) : <code>这个 helper 返回计算值，没有直接写 GM。</code>}</div>
        </div>
      </section>

      <section className={`pl-torch-panel ${step===7 ? "focus" : ""}`}>
        <div><span>TRITON 过程</span><p><b>Grid</b>{selected.grid}</p><p><b>Load</b>{selected.load}</p><p><b>Compute</b>{selected.compute}</p><p><b>Store</b>{selected.store}</p></div>
        <div><span>PYTORCH 语义参考</span><pre><code>{selected.torch}</code></pre><p>PyTorch 对照保证数学语义；它不会复现 Triton 的 program 数量、地址向量、UB 生命周期或 NPU 并行映射。</p></div>
      </section>

      <p className="pl-boundary"><b>模拟边界：</b>坐标、mask、stride 公式和源码签名是可核对的；页面使用缩小后的教学 shape 展开。最终 GM 合并访存、UB/寄存器实际分配、Vector/Cube 映射、流水与双缓冲由 Triton‑Ascend 编译器和具体 NPU 决定。</p>
    </div>
  </main>;
}

export default KernelProductionLab;
