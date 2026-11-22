// The capture harness's wait for work that grows with the model (renderer/harness/context.ts `untilSteady`): it keeps
// waiting while the work advances, and gives up once it stalls. The real Case 1D building's history scan (101 steps,
// 8 parts) outran the fixed 30 s wait it replaced, while still advancing.

import assert from "node:assert/strict";
import { test } from "node:test";
import { untilSteady } from "../src/renderer/harness/context.ts";

test("untilSteady: waits past the stall time while the work advances, and returns as soon as it is done", async () => {
  let done = 0;
  const timer = setInterval(() => done++, 10);
  try {
    // 300 ms of work against a 100 ms stall time: a fixed 100 ms wait would fail it.
    const t0 = performance.now();
    assert.equal(await untilSteady(() => done >= 30, () => done, 100, 5000), true);
    assert.ok(performance.now() - t0 < 2000);
  } finally {
    clearInterval(timer);
  }
});

test("untilSteady: gives up once the work stops advancing, and at the ceiling", async () => {
  const t0 = performance.now();
  assert.equal(await untilSteady(() => false, () => 7, 100, 5000), false, "stalled");
  assert.ok(performance.now() - t0 < 1000, "after the stall time, not the ceiling");
  let n = 0;
  assert.equal(await untilSteady(() => false, () => n++, 100, 200), false, "advancing forever stops at the ceiling");
});
