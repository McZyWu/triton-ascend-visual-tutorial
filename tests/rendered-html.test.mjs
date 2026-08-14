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
  assert.match(html, /源码中的 tl\.load/);
  assert.doesNotMatch(html, /基址不等于元素地址/);
});
