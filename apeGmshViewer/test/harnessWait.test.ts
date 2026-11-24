// The capture harness's waits (renderer/harness/context.ts). `untilSteady` is for work that grows with the model: it
// keeps waiting while the work advances, and gives up once it stalls (the real Case 1D building's history scan, 101
// steps and 8 parts, outran the fixed 30 s wait it replaced, while still advancing). Both `until` and `untilSteady`
// throw on timeout, naming the step (the 2026-11-24 review, finding 3): they used to return false, and most call sites
// ignored it and captured a still of a view that had not arrived.

import assert from "node:assert/strict";
import { test } from "node:test";
import { poll, until, untilSteady, WaitTimeout } from "../src/renderer/harness/context.ts";

test("untilSteady: waits past the stall time while the work advances, and returns as soon as it is done", async () => {
  let done = 0;
  const timer = setInterval(() => done++, 10);
  try {
    // 300 ms of work against a 100 ms stall time: a fixed 100 ms wait would fail it.
    const t0 = performance.now();
    await untilSteady("capture s: the work", () => done >= 30, () => done, 100, 5000);
    assert.ok(performance.now() - t0 < 2000);
  } finally {
    clearInterval(timer);
  }
});

test("untilSteady: throws once the work stops advancing, and at the ceiling, naming the step", async () => {
  const t0 = performance.now();
  await assert.rejects(untilSteady("capture s-1: the scan", () => false, () => 7, 100, 5000), (e: unknown) => e instanceof WaitTimeout && /^capture s-1: the scan: not met after 100 ms$/.test(e.message));
  assert.ok(performance.now() - t0 < 1000, "after the stall time, not the ceiling");
  let n = 0;
  await assert.rejects(untilSteady(() => "advancing forever", () => false, () => n++, 100, 200), { name: "WaitTimeout", message: "advancing forever" });
});

test("until: resolves as soon as the condition holds", async () => {
  let ready = false;
  setTimeout(() => (ready = true), 60);
  const t0 = performance.now();
  await until("capture u: ready", () => ready, 5000);
  assert.ok(performance.now() - t0 < 1000);
});

test("until: throws a WaitTimeout with the step's name instead of returning false", async () => {
  await assert.rejects(until("capture combine-listed: the User Combination to show", () => false, 100), (e: unknown) => {
    assert.ok(e instanceof WaitTimeout);
    assert.equal(e.message, "capture combine-listed: the User Combination to show: not met after 100 ms");
    return true;
  });
  // A message written at the timeout, from the state it timed out in.
  let tries = 0;
  await assert.rejects(until(() => `capture x: gave up after ${tries} tries`, () => (tries++, false), 60), { name: "WaitTimeout", message: /^capture x: gave up after \d+ tries$/ });
});

test("poll: the non-throwing wait, for the one caller that handles a timeout itself (measure's refused contour)", async () => {
  assert.equal(await poll(() => false, 60), false);
  assert.equal(await poll(() => true, 60), true);
});
