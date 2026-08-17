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
console.log("wrote public/profiling-data/mul-add-pipeline-timeline.png");
