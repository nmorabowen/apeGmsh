// ADR 0137 D4 stage 1 (#3057): the memory warning's estimate, its two thresholds on both sides,
// its words, and the banner line the selectors give for the model on screen. The door side (main
// reads an injected memory reading, sends the page its status, and still opens the model) is in
// test/main/loader.test.ts.

import assert from "node:assert/strict";
import { test } from "node:test";
import { AVAILABLE_FLOOR_MB, COMMIT_FACTOR, estimatePeakMB, memoryNotice, memoryReasons } from "../src/state/memory.ts";
import { reduce, initialState } from "../src/state/store.ts";
import { noticesOf } from "../src/state/selectors.ts";
import type { MemoryStatus, State } from "../src/state/types.ts";

test("the estimate is the #3056 whole-app peak: 250 MB + 10 KB per cell for the page, plus 650 MB", () => {
  assert.equal(estimatePeakMB(0), 900);
  assert.equal(estimatePeakMB(1024), 910);
  // The building ADR 0137 states its size from: 128,923 cells.
  assert.equal(Math.round(estimatePeakMB(128_923)), 2159);
});

test("available RAM warns under 2 GB and not at it", () => {
  assert.equal(AVAILABLE_FLOOR_MB, 2048);
  assert.deepEqual(memoryReasons(2048, null, null), []);
  assert.deepEqual(memoryReasons(2047, null, null), ["2.0 GB of RAM is available, under 2.0 GB"]);
  assert.deepEqual(memoryReasons(1500, null, 1000), ["1.5 GB of RAM is available, under 2.0 GB"]);
});

test("commit headroom warns under twice the estimate and not at it", () => {
  assert.equal(COMMIT_FACTOR, 2);
  const est = 1000;
  assert.deepEqual(memoryReasons(8192, 2000, est), []);
  assert.deepEqual(memoryReasons(8192, 1999, est), ["2.0 GB of memory can still be committed, under twice the estimate"]);
  // Both at once: one reason each.
  assert.equal(memoryReasons(1000, 1000, est).length, 2);
});

test("the commit rule needs both figures: no headroom reading (Linux, macOS) or no cell count skips it", () => {
  assert.deepEqual(memoryReasons(8192, null, 1_000_000), []);
  assert.deepEqual(memoryReasons(8192, 10, null), []);
});

const status = (over: Partial<MemoryStatus> = {}): MemoryStatus => ({
  path: "/m/s.h5",
  phase: "before",
  cells: 128_923,
  estimateMB: estimatePeakMB(128_923),
  availableMB: 1500,
  commitHeadroomMB: 9000,
  reasons: memoryReasons(1500, 9000, estimatePeakMB(128_923)),
  ...over,
});

test("the notice states the reasons, the estimate with its cell count, and what may happen; none without a reason", () => {
  assert.equal(
    memoryNotice(status()),
    "Memory is short: 1.5 GB of RAM is available, under 2.0 GB, for a load estimated to peak at about 2.1 GB (128,923 cells). " +
      "The app may be slow or stop responding; closing other programs frees memory.",
  );
  assert.match(memoryNotice(status({ phase: "during" }))!, /^Memory is short while the model loads: /);
  assert.match(memoryNotice(status({ phase: "after" }))!, /^Memory was short while the model loaded: /);
  assert.equal(memoryNotice(status({ cells: null, estimateMB: null })), "Memory is short: 1.5 GB of RAM is available, under 2.0 GB. The app may be slow or stop responding; closing other programs frees memory.");
  assert.equal(memoryNotice(status({ availableMB: 4096, reasons: [] })), null);
});

const opened = (path: string): State => reduce(initialState, { type: "fileOpened", artifact: "model", path });
const memoryLines = (s: State) => noticesOf(s).filter((n) => n.startsWith("Memory "));

test("the banner shows the memory notice for the model being opened, and not for another path", () => {
  const s = reduce(opened("/m/s.h5"), { type: "memoryStatus", status: status() });
  assert.equal(memoryLines(s).length, 1);
  const other = reduce(opened("/m/t.h5"), { type: "memoryStatus", status: status() });
  assert.deepEqual(memoryLines(other), [], "a status of another file is not this model's");
  const fine = reduce(opened("/m/s.h5"), { type: "memoryStatus", status: status({ availableMB: 4096, reasons: [] }) });
  assert.deepEqual(memoryLines(fine), [], "a reading that is not short clears the notice");
});

test("a failed load keeps the memory notice beside the failure", () => {
  let s = reduce(opened("/m/s.h5"), { type: "memoryStatus", status: status({ phase: "after" }) });
  s = reduce(s, { type: "fileFailed", artifact: "model", path: "/m/s.h5", error: "the loader process stopped (exit code -1) before it finished reading" });
  assert.equal(memoryLines(s).length, 1);
});
