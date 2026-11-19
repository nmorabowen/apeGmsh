// #2669: the capture and measure harness says what actually failed, and a model
// that cannot give a row is skipped with the reason.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { readRowStage } from "../src/main/modes/readRow.ts";
import { openLadruno } from "../src/reader/ladruno.ts";
import type { ResultsH5 } from "../src/reader/results.ts";
import { capsStillProblem, scanProblem, scanSatisfied, scrubSkipNote } from "../src/renderer/harness/verdicts.ts";
import type { ContourScan } from "../src/state/types.ts";

const scan = (key: string, status: ContourScan["status"] = "done") => ({ key, status, done: 51, total: 51 }) as unknown as ContourScan;
const source = (p: string): string => readFileSync(fileURLToPath(new URL(p, import.meta.url)), "utf8");

test("a cut-face still with the right caps names the contour that failed, not the caps", () => {
  const msg = capsStillProblem("caps-2-contour", { caps: 1, want: 1, settled: true, deformShown: true, contourWanted: true, contourShown: false, contour: "refused: no catalogue entry" });
  assert.equal(msg, "capture caps-2-contour: the contour does not show the step (refused: no catalogue entry)");
});

test("every failed condition of a cut-face still is named", () => {
  const msg = capsStillProblem("c", { caps: 0, want: 2, settled: false, deformShown: false, contourWanted: false, contourShown: false, contour: "off" });
  assert.equal(msg, "capture c: 0 caps drawn, want 2; the caps have not settled; the deformed shape does not show the step");
});

test("a finished scan is never reported as unfinished", () => {
  const want = { element: true, key: null };
  const nodal = scan("r|nodal");
  assert.equal(scanSatisfied(nodal, want), false);
  assert.match(scanProblem(nodal, want), /that finished is not the one this row needs: it is r\|nodal/);
  assert.doesNotMatch(scanProblem(nodal, want), /did not finish/);
  assert.equal(scanSatisfied(scan("r|elem:S"), want), true);
  assert.match(scanProblem(scan("r|elem:S", "running"), want), /did not finish/);
});

test("a node-located mode waits for its own scan key", () => {
  const want = { element: true, key: "r|elem:S|node" };
  assert.equal(scanSatisfied(scan("r|elem:S|cell"), want), false);
  assert.equal(scanSatisfied(scan("r|elem:S|node"), want), true);
});

test("a one-step stage gives no scrub row and says so", () => {
  assert.match(scrubSkipNote("a.ladruno", "stage_0_mode_1", 1)!, /has 1 step, so there is nothing to scrub/);
  assert.equal(scrubSkipNote("a.ladruno", "s", 2), null);
});

test("an eigen-only file has no read row, and says why instead of failing", () => {
  const modes = Array.from({ length: 12 }, (_, k) => ({ id: `stage_0_mode_${k}`, kind: "mode", nodes: { components: ["DISPLACEMENT/Ux"] } }));
  const r = readRowStage("e.ladruno", modes);
  assert.deepEqual(r, { skip: "e.ladruno: no read row, because its 12 stages are all mode shapes" });
  const moved = { id: "s", kind: "transient", nodes: { components: ["DISPLACEMENT/Ux"] } };
  assert.deepEqual(readRowStage("t.ladruno", [...modes, moved]), { stage: moved });
});

test("the capped measure row does not assume six caps, only six slices per step", () => {
  const src = source("../src/renderer/harness/measure.ts");
  assert.doesNotMatch(src, /capCount === 6|capCount !== 6/);
  assert.match(src, /slices !== 6 \* \(T - 1\)/);
});

test("mode stages as the real reader indexes them give no read row", async () => {
  const h5wasm = await import("h5wasm/node");
  await h5wasm.ready;
  const path = fileURLToPath(new URL("../fixtures/ladruno_modes.ladruno", import.meta.url));
  const rf = openLadruno(h5wasm as unknown as ResultsH5, path);
  try {
    const modes = rf.index.stages.filter((s) => s.kind === "mode");
    assert.ok(modes.length > 0);
    const r = readRowStage(path, modes);
    assert.ok("skip" in r);
    assert.match(r.skip, /stages are all mode shapes/);
  } finally {
    rf.close();
  }
});
