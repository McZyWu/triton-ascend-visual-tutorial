import sharp from "sharp";

const runs = [3.24, 3.14, 3.10, 3.52, 3.32];
const lanes = [
  ["Scalar", 0.765, "#d6ff3f"],
  ["Vector", 0.413, "#5ce1e6"],
  ["MTE2 · GM→UB", 0.263, "#75a7ff"],
  ["MTE3 · UB→GM", 0.198, "#ff9f43"],
];
const esc = value => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const runBars = runs.map((duration, index) => {
  const width = duration / 3.6 * 990;
  const y = 194 + index * 54;
  return `<text x="180" y="${y + 20}" class="label">run ${index + 1}</text><rect x="310" y="${y}" width="${width}" height="30" rx="3" fill="#5ce1e6" opacity=".88"/><text x="${325 + width}" y="${y + 20}" class="value">${duration.toFixed(3)} µs</text>`;
}).join("");
const laneBars = lanes.map(([label, duration, color], index) => {
  const width = Number(duration) / 1.8 * 990;
  const y = 570 + index * 62;
  return `<text x="180" y="${y + 22}" class="label">${esc(label)}</text><rect x="310" y="${y}" width="${width}" height="34" rx="3" fill="${color}"/><text x="${325 + width}" y="${y + 23}" class="value">${Number(duration).toFixed(3)} µs</text>`;
}).join("");

const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
  <rect width="1600" height="900" fill="#0e1915"/>
  <style>
    text{font-family:ui-monospace,Consolas,"Noto Sans CJK SC",sans-serif}.kicker{font-size:19px;fill:#d6ff3f;letter-spacing:2px}.title{font-size:46px;font-weight:700;fill:#f4f2e7}.sub{font-size:18px;fill:#8da098}.label{font-size:18px;fill:#dce9e3}.value{font-size:17px;fill:#f4f2e7}.axis{font-size:15px;fill:#71847b}.note{font-size:16px;fill:#8da098}
  </style>
  <text x="90" y="70" class="kicker">ASCEND 209 · TORCH_NPU PROFILER · REAL MEASUREMENT</text>
  <text x="90" y="125" class="title">mul_add_kernel · B=48 · H=4096 · BF16</text>
  <text x="90" y="158" class="sub">5 active iterations · AI_VECTOR_CORE · Block Num 40 · validation max abs 0.0</text>
  <text x="90" y="205" class="kicker">NPU KERNEL DURATION</text>
  ${runBars}
  <line x1="310" x2="1300" y1="480" y2="480" stroke="#40534a"/>
  <text x="310" y="505" class="axis">0</text><text x="795" y="505" class="axis">1.8 µs</text><text x="1260" y="505" class="axis">3.6 µs</text>
  <text x="90" y="560" class="kicker">MEDIAN PIPE ACTIVITY · INDEPENDENT LANES</text>
  ${laneBars}
  <line x1="310" x2="1300" y1="822" y2="822" stroke="#40534a"/>
  <text x="310" y="847" class="axis">0</text><text x="785" y="847" class="axis">0.9 µs</text><text x="1250" y="847" class="axis">1.8 µs</text>
  <text x="90" y="882" class="note">Pipeline lanes may overlap; do not add Scalar + Vector + MTE2 + MTE3. Re-rendered from kernel_details.csv and trace_view.json.</text>
</svg>`;

await sharp(Buffer.from(svg)).png().toFile("public/profiling-data/mul-add-pipeline-timeline.png");

function caseCard({ kicker, title, subtitle, durations, median, min, max, lanes, note }) {
  const maxDuration = Math.max(...durations, max) * 1.08;
  const eventBars = durations.map((duration, index) => {
    const width = duration / maxDuration * 690;
    const y = 222 + index * 58;
    return `<text x="130" y="${y + 22}" class="label">event ${index + 1}</text><rect x="270" y="${y}" width="${width}" height="34" rx="3" fill="#5ce1e6"/><text x="${285 + width}" y="${y + 23}" class="value">${duration.toFixed(2)} µs</text>`;
  }).join("");
  const laneBars = lanes.map(([label, ratio, color], index) => {
    const width = Number(ratio) / 45 * 430;
    const y = 230 + index * 95;
    return `<text x="1070" y="${y}" class="label">${esc(label)}</text><rect x="1070" y="${y + 15}" width="430" height="28" rx="3" fill="#22342c"/><rect x="1070" y="${y + 15}" width="${width}" height="28" rx="3" fill="${color}"/><text x="1070" y="${y + 68}" class="value">${Number(ratio).toFixed(2)}%</text>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
    <rect width="1600" height="900" fill="#0e1915"/>
    <style>text{font-family:ui-monospace,Consolas,"Noto Sans CJK SC",sans-serif}.kicker{font-size:18px;fill:#d6ff3f;letter-spacing:2px}.title{font-size:42px;font-weight:700;fill:#f4f2e7}.sub{font-size:17px;fill:#8da098}.label{font-size:16px;fill:#dce9e3}.value{font-size:16px;fill:#f4f2e7}.big{font-size:44px;font-weight:700;fill:#d6ff3f}.note{font-size:15px;fill:#8da098}</style>
    <text x="80" y="64" class="kicker">${esc(kicker)}</text><text x="80" y="118" class="title">${esc(title)}</text><text x="80" y="153" class="sub">${esc(subtitle)}</text>
    <rect x="80" y="180" width="920" height="520" rx="4" fill="#14211c"/><text x="110" y="205" class="kicker">EXACT EXTRACT · FIRST 5 EVENTS</text>${eventBars}
    <rect x="1030" y="180" width="500" height="520" rx="4" fill="#14211c"/><text x="1070" y="205" class="kicker">MEDIAN PIPE RATIO</text>${laneBars}
    <text x="90" y="765" class="big">median ${median.toFixed(2)} µs</text><text x="790" y="756" class="sub">all captured calls</text><text x="790" y="786" class="value">min ${min.toFixed(2)} µs · max ${max.toFixed(2)} µs</text>
    <line x1="80" x2="1530" y1="825" y2="825" stroke="#40534a"/><text x="80" y="860" class="note">${esc(note)}</text>
  </svg>`;
}

const situSvg = caseCard({
  kicker: "ASCEND 209 · STORED PRODUCTION CAPTURE · 2026-08-05",
  title: "Kimi-K3 · _situ_deepep_kernel_0",
  subtitle: "276 calls · AI_VECTOR_CORE · custom-kernel shape fields unavailable",
  durations: [21.3, 11.9, 10.62, 11.14, 9.4], median: 10.44, min: 8.74, max: 21.3,
  lanes: [["Vector", 22.55, "#5ce1e6"], ["Scalar", 10.7, "#d6ff3f"], ["MTE2 · GM→UB", 10.3, "#75a7ff"], ["MTE3 · UB→GM", 1.85, "#ff9f43"]],
  note: "Production snapshot, not a controlled shape-matched A/B. Source: current K3 capture; exact events are included in k3-qwen-trace-extract.json.",
});
const qwenSvg = caseCard({
  kicker: "ASCEND 209 · STORED PRODUCTION CAPTURE · 2026-08-05",
  title: "Qwen family · split_qkv_rmsnorm_rope_kernel_0",
  subtitle: "15 calls · fused split / RMSNorm / RoPE path · AI_VECTOR_CORE",
  durations: [15.2, 17.0, 19.08, 23.5, 15.38], median: 12.32, min: 11.52, max: 23.5,
  lanes: [["Vector", 20.0, "#5ce1e6"], ["Scalar", 18.6, "#d6ff3f"], ["MTE2 · GM→UB", 29.1, "#75a7ff"], ["MTE3 · UB→GM", 14.3, "#ff9f43"]],
  note: "MTE2 is the largest median pipeline ratio. Lanes may overlap; percentages must not be added into Duration.",
});
await Promise.all([
  sharp(Buffer.from(situSvg)).png().toFile("public/profiling-data/k3-situ-profile.png"),
  sharp(Buffer.from(qwenSvg)).png().toFile("public/profiling-data/qwen-fused-profile.png"),
]);
console.log("wrote mul_add, K3 SiTU, and Qwen fused profiling figures");
