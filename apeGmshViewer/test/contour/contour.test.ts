// V4c1 (#1756): the contour's pure parts (fields, labels, colormaps, ranges
// and what the legend says about them), the reducer's guards, and the seams:
// the history-scan cancel, a reload that loses the field, a partial write
// refused per component.

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as h5wasm from "h5wasm/node";
import { ResultsHost } from "../../src/main/results.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { contourLines } from "../../src/panels/legend.ts";
import { CMAP_NAMES, climOf, MAGNITUDE, NO_DATA_RGB, nodalFields, primitiveColours, sampleCmap, vertexValues } from "../../src/state/contour.ts";
import { contourFieldsOf, contourView, noticesOf } from "../../src/state/selectors.ts";
import type { ContourScan } from "../../src/state/types.ts";
import { opened, STAGED_RESULTS, tick, until } from "./harness.ts";

// -- fields and labels ----------------------------------------------------------

test("nodal fields: axis components grouped by field, the magnitude for translational vectors only, names from the file", () => {
  const f = nodalFields(["rotation_z", "displacement_x", "displacement_y", "pore_pressure", "reaction_force_x", "reaction_force_z", "velocity_x"]);
  const shape = f.map((x) => [x.name, x.components.map((c) => `${c.label}=${c.datasets.join("+")}`)]);
  assert.deepEqual(shape, [
    ["rotation", ["z=rotation_z"]],
    ["displacement", ["magnitude=displacement_x+displacement_y", "x=displacement_x", "y=displacement_y"]],
    ["pore_pressure", ["pore_pressure=pore_pressure"]],
    ["reaction_force", ["magnitude=reaction_force_x+reaction_force_z", "x=reaction_force_x", "z=reaction_force_z"]],
    ["velocity", ["x=velocity_x"]],
  ]);
  // Each axis label names the dataset of that axis (an off-by-one would pair x with y's data).
  for (const fld of f) for (const c of fld.components) if (!c.magnitude && c.label.length === 1) assert.equal(c.key, `${fld.name}_${c.label}`);
  assert.equal(f[1]!.components[0]!.key, MAGNITUDE);
});

test("an unnamed component (a name ending in `_`) renders as [k], never as a guessed axis", () => {
  const f = nodalFields(["strain_x", "strain_", "strain_y"]);
  assert.deepEqual(f[0]!.components.map((c) => [c.label, c.key]), [["x", "strain_x"], ["y", "strain_y"], ["[2]", "strain_"]]);
});

// -- colormaps ------------------------------------------------------------------

test("jet is matplotlib's segment data at its breakpoints; the no-data colour is far from every ramp", () => {
  assert.deepEqual(sampleCmap("jet", 0), [0, 0, 0.5]);
  assert.deepEqual(sampleCmap("jet", 1), [0.5, 0, 0]);
  const mid = sampleCmap("jet", 0.5);
  // red (0.35 -> 0.66: 0 -> 1), green (0.375 -> 0.64: 1), blue (0.34 -> 0.65: 1 -> 0)
  assert.ok(Math.abs(mid[0] - 0.15 / 0.31) < 1e-12 && mid[1] === 1 && Math.abs(mid[2] - 0.15 / 0.31) < 1e-12, `jet(0.5) = ${mid}`);
  for (const cmap of CMAP_NAMES) {
    for (let i = 0; i <= 200; i++) {
      const c = sampleCmap(cmap, i / 200);
      const d = Math.hypot(c[0] - NO_DATA_RGB[0], c[1] - NO_DATA_RGB[1], c[2] - NO_DATA_RGB[2]);
      assert.ok(d > 0.15, `${cmap}(${i / 200}) is ${d.toFixed(3)} from the no-data colour`);
    }
  }
});

test("vertex values: a node not in _ids, and a NaN, are no data (NaN), never 0", () => {
  const column = new Map([[10, 0], [11, 1], [12, 2]]);
  const x = Float64Array.of(1, NaN, -3);
  const y = Float64Array.of(0, 2, 4);
  const nodes = Float64Array.of(10, 11, 12, 99, NaN);
  assert.deepEqual(Array.from(vertexValues(nodes, column, [x], false)), [1, NaN, -3, NaN, NaN]);
  assert.deepEqual(Array.from(vertexValues(nodes, column, [x, y], true)), [1, NaN, 5, NaN, NaN]);
  // A primitive with one missing vertex is no data whole; a range with no finite value colours nothing.
  const vals = Float64Array.of(0, 1, 2, 0, NaN, 2);
  const { rgb, noData } = primitiveColours(vals, Int32Array.of(0, 1), 3, { min: 0, max: 2 }, "jet");
  assert.equal(noData, 1);
  for (let v = 3; v < 6; v++) assert.deepEqual(Array.from(rgb.subarray(3 * v, 3 * v + 3)), NO_DATA_RGB.map(Math.fround));
  assert.deepEqual(Array.from(rgb.subarray(0, 3)), sampleCmap("jet", 0).map(Math.fround));
  assert.equal(primitiveColours(vals, Int32Array.of(0), 3, { min: null, max: null }, "jet").noData, 1);
});

// -- the range mode, and what the legend says --------------------------------------

test("the range mode is always stated: history (scanning, done, failed, another key's scan), this step, fixed", () => {
  const step = { min: 1, max: 2 };
  const scan = (p: Partial<ContourScan>): ContourScan => ({ key: "K", status: "running", done: 0, total: 4, min: null, max: null, error: null, ...p });
  const h = { range: "history" as const, fixed: null };
  const at = (c: Parameters<typeof climOf>[0], sc: ContourScan | null, key = "K") => {
    const r = climOf(c, step, sc, key);
    return [r.mode, r.min, r.max, r.label];
  };
  assert.deepEqual(at(h, null), ["step", 1, 2, "range: this step (history scan 0%)"]);
  assert.deepEqual(at(h, scan({ done: 1 })), ["step", 1, 2, "range: this step (history scan 25%)"]);
  assert.deepEqual(at(h, scan({ status: "done", done: 4, min: -1, max: 9 })), ["history", -1, 9, "range: whole history (4 steps)"]);
  assert.deepEqual(at(h, scan({ status: "failed", error: "boom" })), ["step", 1, 2, "range: this step (history scan failed: boom)"]);
  assert.deepEqual(at(h, scan({ status: "done", done: 4, min: -1, max: 9 }), "OTHER"), ["step", 1, 2, "range: this step (history scan 0%)"], "a scan of another key is not this one's");
  assert.deepEqual(at({ range: "step", fixed: null }, scan({ status: "done", min: -1, max: 9 })), ["step", 1, 2, "range: this step"]);
  assert.deepEqual(at({ range: "fixed", fixed: { min: -5, max: 5 } }, null), ["fixed", -5, 5, "range: fixed (user)"]);
});

test("the legend states the field, the component, the range mode, and a step still loading", async () => {
  const h = await opened();
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  const v0 = contourView(h.store.get());
  assert.ok(v0.state === "ready");
  assert.equal(contourLines(h.store.get(), v0).state, "step 1 loading");
  await until(() => h.store.get().contour.scan?.status === "done", "the history scan");
  await until(() => (contourView(h.store.get()) as { stepMeasured?: boolean }).stepMeasured === true, "the step range");
  const v = contourView(h.store.get());
  assert.ok(v.state === "ready");
  const l = contourLines(h.store.get(), v);
  assert.deepEqual([l.title, l.range, l.state], ["displacement · magnitude", "range: whole history (4 steps)", null]);
  h.store.dispatch({ type: "setContourRange", range: "step" });
  assert.equal(contourLines(h.store.get(), contourView(h.store.get()) as Extract<ReturnType<typeof contourView>, { state: "ready" }>).range, "range: this step");
  h.effects.dispose();
});

// -- the reducer's guards ---------------------------------------------------------

test("the reducer refuses a pick the stage does not have, a bad fixed range, an unknown colormap, and drops stale ranges", async () => {
  const h = await opened();
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  const s = h.store.get();
  assert.deepEqual([s.contour.field, s.contour.component], ["displacement", MAGNITUDE], "the default pick: the first field's magnitude");
  assert.throws(() => h.store.dispatch({ type: "setContourField", field: "displacement", component: "displacement_w" }), RangeError);
  assert.throws(() => h.store.dispatch({ type: "setContourField", field: "rotation", component: "rotation_z" }), RangeError);
  assert.throws(() => h.store.dispatch({ type: "setContourRange", range: "fixed", fixed: { min: 2, max: 1 } }), RangeError);
  assert.throws(() => h.store.dispatch({ type: "setContourRange", range: "fixed", fixed: { min: NaN, max: 1 } }), RangeError);
  assert.throws(() => h.store.dispatch({ type: "setContourCmap", cmap: "rainbow" as never }), RangeError);
  const before = h.store.get();
  h.store.dispatch({ type: "contourStepRange", key: "another|step", min: 0, max: 1 });
  h.store.dispatch({ type: "contourScan", scan: { key: "another", status: "done", done: 1, total: 1, min: 0, max: 1, error: null } });
  assert.equal(h.store.get().contour.stepRange, before.contour.stepRange);
  assert.equal(h.store.get().contour.scan?.key ?? null, before.contour.scan?.key ?? null);
  h.effects.dispose();
});

// -- seams ------------------------------------------------------------------------

test("a field change cancels the running history scan; its late answer never lands", async () => {
  const h = await opened(undefined, undefined, { scanOpts: { chunkSteps: 1, paceMs: 40 } });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  await until(() => h.calls.scans.length === 1 && h.host.runningScans().length === 1, "the first scan");
  const first = h.calls.scans[0]!.scanId;
  h.store.dispatch({ type: "setContourField", field: "displacement", component: "displacement_x" });
  await tick();
  assert.deepEqual(h.calls.cancels, [first], "the magnitude scan was cancelled in main");
  await until(() => h.store.get().contour.scan?.status === "done", "the displacement_x scan");
  assert.equal(h.host.runningScans().length, 0, "no scan left running");
  const sc = h.store.get().contour.scan!;
  assert.ok(sc.key.endsWith("|displacement|displacement_x"), sc.key);
  assert.equal(h.calls.scans.length, 2);
  assert.deepEqual(h.calls.scans[1]!.components, ["displacement_x"]);
  h.effects.dispose();
});

test("reload: a re-read cancels the scan; a re-read without the picked component clears the pick with a notice", async () => {
  const h = await opened(undefined, undefined, { scanOpts: { chunkSteps: 1, paceMs: 40 } });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "displacement", component: "displacement_y" });
  await until(() => h.host.runningScans().length === 1, "a running scan");
  const running = h.calls.scans.at(-1)!.scanId;
  // The same file read again (a D1 re-run): the old scan stops; the ranges measured on it are forgotten.
  await h.effects.openResults(STAGED_RESULTS);
  assert.ok(h.calls.cancels.includes(running), "the scan was cancelled when the file was read again");
  // A re-read whose stage lost displacement_y: the pick is cleared, with a typed notice, and not re-picked on the next scrub.
  const s = h.store.get();
  const info = { ...s.results!, stages: s.results!.stages.map((st, i) => (i === 2 ? { ...st, nodal: st.nodal.filter((c) => c !== "displacement_y") } : st)) };
  h.store.dispatch({ type: "resultsLoaded", info, sizeBytes: 0, readMs: 0, warnings: [] });
  const after = h.store.get();
  assert.deepEqual([after.contour.field, after.contour.component, after.contour.scan, after.contour.stepRange], [null, null, null, null]);
  assert.match(after.contour.notice!, /has no displacement displacement_y in stage unprop: the contour pick was cleared/);
  assert.ok(noticesOf(after).includes(after.contour.notice!));
  h.store.dispatch({ type: "setResultStep", step: 1 });
  assert.equal(h.store.get().contour.field, null, "a scrub does not re-pick behind the notice");
  assert.equal(contourView(h.store.get()).state, "unavailable");
  h.store.dispatch({ type: "setContourField", field: "displacement", component: "displacement_x" });
  assert.equal(h.store.get().contour.notice, null);
  h.effects.dispose();
});

test("ResultsHost: a cancel stops a running scan, and so does opening the file again", async () => {
  const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready, { idleMs: 50 });
  await host.open(STAGED_RESULTS);
  const req = { path: STAGED_RESULTS, stage: "stage_2", components: ["displacement_x"], magnitude: false, scanId: 7 };
  const a = host.scan(req, () => {}, { chunkSteps: 1, paceMs: 30 });
  host.cancelScan(7);
  assert.deepEqual(await a, { ok: false, scanId: 7, cancelled: true, error: "cancelled" });
  const b = host.scan({ ...req, scanId: 8 }, () => {}, { chunkSteps: 1, paceMs: 30 });
  await new Promise((r) => setTimeout(r, 5));
  await host.open(STAGED_RESULTS);
  assert.deepEqual(await b, { ok: false, scanId: 8, cancelled: true, error: "cancelled" });
  const c = await host.scan({ ...req, scanId: 9 }, () => {});
  assert.ok(c.ok && c.steps === 4);
  host.close();
});

test("partial write: a component whose value columns disagree with _ids is refused alone, with a banner; the file still opens", async () => {
  await h5wasm.ready;
  const path = join(mkdtempSync(join(tmpdir(), "v4c1-")), "partial.results.h5");
  const f = new h5wasm.File(path, "w");
  f.create_attribute("results_schema_version", "1.0.0");
  const st = f.create_group("stages").create_group("stage_0");
  st.create_attribute("name", "pushover");
  st.create_attribute("kind", "static");
  st.create_dataset({ name: "time", data: new Float64Array([0, 1]) });
  const n = st.create_group("partitions").create_group("partition_0").create_group("nodes");
  n.create_dataset({ name: "_ids", data: new BigInt64Array([1n, 2n, 3n]) });
  n.create_dataset({ name: "displacement_x", data: new Float64Array([0, 1, 2, 3, 4, 5]), shape: [2, 3] });
  n.create_dataset({ name: "displacement_y", data: new Float64Array([0, 1, 2, 3]), shape: [2, 2] });
  f.close();
  const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);
  const a = await host.open(path);
  assert.ok(a.ok, a.ok ? "" : a.error);
  const stage = a.meta.stages[0]!;
  assert.deepEqual(stage.nodal, ["displacement_x"]);
  assert.equal(stage.refusedNodal.length, 1);
  assert.equal(stage.refusedNodal[0]!.component, "displacement_y");
  assert.match(stage.refusedNodal[0]!.reason, /has 2 value columns and _ids has 3 node ids \(a partial write\)/);
  // The picker offers what was read and lists the refusal; the banner says it.
  assert.deepEqual(nodalFields(stage.nodal).map((x) => x.components.map((c) => c.label)), [["x"]]);
  const h = await opened();
  const s = h.store.get();
  const info = { ...s.results!, stages: s.results!.stages.map((x, i) => (i === 2 ? { ...x, refusedNodal: stage.refusedNodal } : x)) };
  h.store.dispatch({ type: "resultsLoaded", info, sizeBytes: 0, readMs: 0, warnings: [] });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  assert.deepEqual(contourFieldsOf(h.store.get())!.refused, stage.refusedNodal);
  assert.ok(noticesOf(h.store.get()).some((t) => t.startsWith("Nodal component displacement_y not read: ")));
  host.close();
  h.effects.dispose();
});
