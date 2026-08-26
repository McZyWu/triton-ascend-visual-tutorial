import assert from "node:assert/strict";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${pathname}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(`http://localhost${pathname}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Triton Ascend tutorial", async () => {
  const response = await render("/");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /Triton Ascend Visual Lab/);
  assert.match(html, /06 \/ PRODUCTION CASES/);
  assert.match(html, /\/kernel-lab/);
  assert.match(html, /84<!-- --> 个 JIT kernel|84 个 JIT kernel/);
  assert.match(html, /Tile 是工作块，lane 是块内位置/);
  assert.match(html, /lane 3/);
  assert.match(html, /lane 5/);
  assert.match(html, /DATA MOVEMENT 播放与单步控制/);
  const dataMovementStage = html.indexOf('class="memory-stage-v2"');
  const dataMovementControls = html.indexOf('class="transfer-playback-controls"');
  const dataMovementStep = html.indexOf('class="step-line"');
  assert.ok(dataMovementStage >= 0 && dataMovementStage < dataMovementControls, "DATA MOVEMENT controls should follow the transfer visualization");
  assert.ok(dataMovementControls < dataMovementStep, "DATA MOVEMENT controls should stay attached to the transfer visualization");
  assert.doesNotMatch(html, /Your site is taking shape|codex-preview/);
});

test("server-renders the full production kernel lab", async () => {
  const response = await render("/kernel-lab?op=rms");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Production Kernel Lab/);
  assert.match(html, /GRID \/ PROGRAM MAP/);
  assert.match(html, /BLOCK \/ LANE ADDRESSING/);
  assert.match(html, /ALL KERNEL ARGUMENTS/);
  assert.match(html, /每个变量是什么、什么 shape、从模型哪里传入/);
  assert.match(html, /GM → UB \/ REG → GM/);
  assert.match(html, /SOURCE → VISUAL STAGE/);
  assert.match(html, /2a87cda/);
  assert.doesNotMatch(html, /基址不等于元素地址/);
});

test("mul-add exposes both draggable persistent rounds", async () => {
  const response = await render("/kernel-lab?op=muladd");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /MoE Mul‑Add/);
  assert.match(html, /persistent 轮次 k/);
  assert.match(html, /data-round-max="1"/);
  assert.match(html, /共 (?:<!-- -->)?2(?:<!-- -->)? 轮/);
  assert.match(html, /可拖动滑块、点 ± 或用方向键/);
  assert.match(html, /搬运播放与单步控制/);
  assert.match(html, /播放搬运/);
  assert.match(html, /单步后退/);
  assert.match(html, /单步前进/);
  const memoryFlow = html.indexOf('class="pl-memory-flow"');
  const playbackControls = html.indexOf('class="pl-memory-controls"');
  const ubLedger = html.indexOf('class="pl-ub-ledger"');
  assert.ok(memoryFlow >= 0 && memoryFlow < playbackControls, "playback controls should follow the transfer visualization");
  assert.ok(playbackControls < ubLedger, "playback controls should remain attached to the transfer visualization");
});

test("server-renders the profiling evidence page", async () => {
  const response = await render("/profiling");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /ASCEND PROFILING FIELD GUIDE/);
  assert.match(html, /00 \/ MEMORY TERMINOLOGY/);
  assert.match(html, /GM 是仓库，UB 是当前核的工作台/);
  assert.match(html, /每轮最低 GM 流量/);
  assert.match(html, /真实 UB 峰值/);
  assert.match(html, /MTE2\/MTE3 告诉你“搬运流水运行了多久”/);
  assert.match(html, /mul_add_kernel/);
  assert.match(html, /2a87cda/);
  assert.match(html, /32 KiB/);
  assert.match(html, /MTE2/);
  assert.match(html, /kernel_details\.csv/);
  assert.match(html, /chrome:\/\/tracing\//);
  assert.match(html, /mul-add-trace-view\.json/);
  assert.match(html, /01 \/ SINGLE OP CAPTURE/);
  assert.match(html, /02 \/ PIPELINE &amp; TRACE\.JSON/);
  assert.match(html, /03 \/ HOTSPOT/);
  assert.match(html, /04 \/ BOUND/);
  assert.match(html, /05 \/ OPTIMIZE/);
  assert.match(html, /06 \/ RESULTS &amp; BENEFIT/);
  assert.match(html, /_situ_deepep_kernel_0/);
  assert.match(html, /split_qkv_rmsnorm_rope_kernel_0/);
  assert.match(html, /Start Time\(us\)/);
  assert.match(html, /aiv_mte2_time \/ ratio/);
  assert.match(html, /k3-qwen-trace-extract\.json/);
  assert.match(html, /不同 shape 不计算 speedup/);
});
