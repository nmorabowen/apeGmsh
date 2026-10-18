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
import { contourLines, rampCss } from "../../src/panels/legend.ts";
import { contourKey } from "../../src/renderer/contour.ts";
import { CMAPS, CMAP_NAMES, climOf, fieldsOf, MAGNITUDE, NO_DATA_RGB, nodalFields, primitiveColours, sampleCmap, vertexValues } from "../../src/state/contour.ts";
import { contourFieldsOf, contourView, noticesOf, resolveFieldUnits } from "../../src/state/selectors.ts";
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

test("jet is matplotlib's segment data at its breakpoints; diverging wavefield maps centre on white; the no-data colour is far from every ramp", () => {
  assert.deepEqual(CMAP_NAMES, [
    "jet", "turbo", "viridis", "plasma", "inferno", "magma",
    "coolwarm", "RdBu_r", "RdBu", "bwr", "seismic", "BrBG", "Spectral",
  ]);
  assert.deepEqual(Object.keys(CMAPS), [...CMAP_NAMES]);
  assert.deepEqual(sampleCmap("jet", 0), [0, 0, 0.5]);
  assert.deepEqual(sampleCmap("jet", 1), [0.5, 0, 0]);
  const mid = sampleCmap("jet", 0.5);
  // red (0.35 -> 0.66: 0 -> 1), green (0.375 -> 0.64: 1), blue (0.34 -> 0.65: 1 -> 0)
  assert.ok(Math.abs(mid[0] - 0.15 / 0.31) < 1e-12 && mid[1] === 1 && Math.abs(mid[2] - 0.15 / 0.31) < 1e-12, `jet(0.5) = ${mid}`);
  // Symmetric blue-white-red wavefield maps: negative is blue (b > r), zero is neutral white, positive is red (r > b).
  for (const wf of ["RdBu_r", "bwr", "seismic"] as const) {
    const lo = sampleCmap(wf, 0), zero = sampleCmap(wf, 0.5), hi = sampleCmap(wf, 1);
    assert.ok(lo[2] > lo[0] && hi[0] > hi[2], `${wf} endpoints: ${lo} -> ${hi}`);
    assert.ok(zero[0] > 0.95 && zero[1] > 0.95 && zero[2] > 0.95, `${wf}(0.5) = ${zero}`);
  }
  for (const [t0, t1] of [[0, 1], [0.25, 0.75], [0.5, 0.5], [1, 0]] as const) {
    const a = sampleCmap("RdBu", t0), b = sampleCmap("RdBu_r", t1);
    assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-12, `RdBu(${t0}) vs RdBu_r(${t1})`);
  }
  for (const cmap of CMAP_NAMES) {
    const stops = CMAPS[cmap];
    assert.equal(stops[0]![0], 0, `${cmap} starts at 0`);
    assert.equal(stops.at(-1)![0], 1, `${cmap} ends at 1`);
    for (let i = 0; i < stops.length; i++) {
      const [t, r, g, b] = stops[i]!;
      if (i > 0) assert.ok(t > stops[i - 1]![0], `${cmap} stop ${i} strictly increases`);
      assert.ok(r >= 0 && r <= 1 && g >= 0 && g <= 1 && b >= 0 && b <= 1, `${cmap} stop ${i} in [0,1]`);
    }
    const grad = rampCss(cmap);
    assert.match(grad, /^linear-gradient\(to right, rgb\(\d+,\d+,\d+\) 0\.0%(?:, rgb\(\d+,\d+,\d+\) \d+\.\d%)+, rgb\(\d+,\d+,\d+\) 100\.0%\)$/);
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

test("a refused translational axis withdraws the magnitude: it is shown refused, never offered over the surviving axes, never scanned", async () => {
  // The review's reproducer: x and y are (2, 3), z is (2, 2) with 3 _ids, so z is refused.
  await h5wasm.ready;
  const path = join(mkdtempSync(join(tmpdir(), "v4c1-")), "zrefused.results.h5");
  const f = new h5wasm.File(path, "w");
  f.create_attribute("results_schema_version", "1.0.0");
  const st = f.create_group("stages").create_group("stage_0");
  st.create_attribute("name", "pushover");
  st.create_attribute("kind", "static");
  st.create_dataset({ name: "time", data: new Float64Array([0, 1]) });
  const n = st.create_group("partitions").create_group("partition_0").create_group("nodes");
  n.create_dataset({ name: "_ids", data: new BigInt64Array([1n, 2n, 3n]) });
  n.create_dataset({ name: "displacement_x", data: new Float64Array([0, 1, 2, 3, 4, 5]), shape: [2, 3] });
  n.create_dataset({ name: "displacement_y", data: new Float64Array([0, 0, 0, 0, 0, 0]), shape: [2, 3] });
  n.create_dataset({ name: "displacement_z", data: new Float64Array([100, 100, 100, 100]), shape: [2, 2] });
  f.close();
  const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);
  const a = await host.open(path);
  assert.ok(a.ok, a.ok ? "" : a.error);
  const stage = a.meta.stages[0]!;
  assert.deepEqual(stage.nodal, ["displacement_x", "displacement_y"]);
  const [disp] = fieldsOf({ ...stage, refusedNodal: stage.refusedNodal });
  assert.deepEqual(disp!.components.map((c) => c.label), ["x", "y"], "no magnitude over x and y alone");
  assert.match(disp!.magnitudeRefused!, /^displacement_z not read \(.*has 2 value columns and _ids has 3 node ids/);
  assert.equal(nodalFields(stage.nodal).find((x) => x.name === "displacement")!.components[0]!.label, "magnitude", "without the refusal the axes would offer one");
  host.close();
  // In the app: the refused stage shown, colour by contour; the default pick is x, a magnitude pick is refused, and no scan reads a magnitude.
  const h = await opened();
  const s = h.store.get();
  const info = {
    ...s.results!,
    stages: s.results!.stages.map((x, i) => (i === 2 ? { ...x, nodal: x.nodal.filter((c) => c !== "displacement_z"), refusedNodal: [{ component: "displacement_z", reason: "a partial write" }] } : x)),
  };
  h.store.dispatch({ type: "resultsLoaded", info, sizeBytes: 0, readMs: 0, warnings: [] });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  assert.deepEqual([h.store.get().contour.field, h.store.get().contour.component], ["displacement", "displacement_x"]);
  assert.throws(() => h.store.dispatch({ type: "setContourField", field: "displacement", component: MAGNITUDE }), RangeError);
  assert.equal(contourFieldsOf(h.store.get())!.fields[0]!.magnitudeRefused, "displacement_z not read (a partial write)");
  await until(() => h.store.get().contour.scan?.status === "done", "the history scan");
  assert.ok(h.calls.scans.length > 0 && h.calls.scans.every((c) => !c.magnitude), JSON.stringify(h.calls.scans));
  h.effects.dispose();
});

test("a re-read always recolours: the colour key changes although the step, the component and the range do not", async () => {
  const h = await opened();
  const shown = () => {
    const v = contourView(h.store.get());
    return v.state === "ready" && v.stepMeasured && h.effects.currentStep()?.step === 3 ? v : null;
  };
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  h.store.dispatch({ type: "setResultStep", step: 3 });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  // Path 1: a fixed range, unchanged across the re-read.
  h.store.dispatch({ type: "setContourRange", range: "fixed", fixed: { min: 0, max: 1 } });
  await until(() => shown() !== null, "the step, fixed range");
  const before = shown()!;
  await h.effects.openResults(STAGED_RESULTS);
  await until(() => shown() !== null, "the step after the re-read");
  const after = shown()!;
  assert.deepEqual([after.keys.step, after.clim.min, after.clim.max], [before.keys.step, before.clim.min, before.clim.max]);
  assert.notEqual(contourKey(after), contourKey(before), "fixed range: the re-read changes the key");
  // Path 2: history mode at the last step, where the step range equals the history range.
  h.store.dispatch({ type: "setContourRange", range: "history" });
  await until(() => h.store.get().contour.scan?.status === "done" && shown() !== null, "the history range");
  const hist = shown()!;
  const sr = h.store.get().contour.stepRange!;
  assert.deepEqual([hist.clim.mode, hist.clim.min, hist.clim.max], ["history", sr.min, sr.max], "the step range equals the history range here");
  await h.effects.openResults(STAGED_RESULTS);
  await until(() => h.store.get().contour.scan?.status === "done" && shown() !== null, "the history range after the re-read");
  const hist2 = shown()!;
  assert.deepEqual([hist2.keys.step, hist2.clim.min, hist2.clim.max], [hist.keys.step, hist.clim.min, hist.clim.max]);
  assert.notEqual(contourKey(hist2), contourKey(hist), "history range: the re-read changes the key");
  h.effects.dispose();
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

test("baseUnits: setContourUnits converts forces and moments to tf_m_s and labels display units", () => {
  const rf = resolveFieldUnits("reaction_force", "nodal", "kN_m_s", "tf_m_s", "static");
  assert.equal(rf.unit, "tf");
  assert.ok(Math.abs(rf.factor - 1000 / 9806.65) < 1e-15);
  const wa = resolveFieldUnits("wood_armer_neg", "element", "kN_m_s", "tf_m_s", "static");
  assert.equal(wa.unit, "tf·m/m");
  assert.ok(Math.abs(wa.factor - 1000 / 9806.65) < 1e-15);
  const rm = resolveFieldUnits("rib_moment_pos", "element", "kN_m_s", "tf_m_s", "static");
  assert.equal(rm.unit, "tf·m");
  assert.ok(Math.abs(rm.factor - 1000 / 9806.65) < 1e-15);
  const dispMm = resolveFieldUnits("displacement", "nodal", "kN_m_s", "N_mm_s", "static");
  assert.equal(dispMm.unit, "mm");
  assert.equal(dispMm.factor, 1000);
  const modeDisp = resolveFieldUnits("displacement", "nodal", "kN_m_s", "N_mm_s", "mode");
  assert.equal(modeDisp.unit, null);
  assert.equal(modeDisp.factor, 1);
});

test("declared vs undeclared units: declared files auto-select native units; undeclared files require source units", async () => {
  const h = await opened();
  // Undeclared file starts at as_stored with sourceUnits === null.
  assert.equal(h.store.get().contour.declaredUnits, null);
  assert.equal(h.store.get().contour.sourceUnits, null);
  assert.equal(h.store.get().contour.targetUnits, "as_stored");

  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 1 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  await until(() => h.store.get().contour.scan?.status === "done", "the initial history scan");

  // Selecting a target unit system on an undeclared file without sourceUnits prompts for source units.
  h.store.dispatch({ type: "setContourUnits", target: "N_mm_s" });
  assert.equal(h.store.get().contour.sourceUnits, null);
  assert.equal(h.store.get().contour.targetUnits, "N_mm_s");
  const vUndeclared = contourView(h.store.get());
  assert.equal(vUndeclared.state, "ready");
  if (vUndeclared.state === "ready") {
    const lines = contourLines(h.store.get(), vUndeclared);
    assert.equal(lines.title, "displacement · magnitude");
    assert.equal(lines.units, "Units: not declared in file — select the source unit system (from …) to convert.");
  }

  // Once the user specifies sourceUnits, conversion is applied immediately.
  h.store.dispatch({ type: "setContourUnits", source: "kN_m_s" });
  assert.equal(h.store.get().contour.sourceUnits, "kN_m_s");
  if (vUndeclared.state === "ready") {
    const lines = contourLines(h.store.get(), vUndeclared);
    assert.equal(lines.title, "displacement · magnitude [mm]");
    assert.equal(lines.units, "Units: N-mm-tonne-s (converted from kN-m-tonne-s).");
  }

  // When a results file carries declared units (`units: "kN_m_s"`), native units are auto-selected and sourceUnits is locked.
  const infoWithUnits = { ...h.store.get().results!, units: "kN_m_s" as const };
  h.store.dispatch({ type: "resultsLoaded", info: infoWithUnits, sizeBytes: 0, readMs: 0, warnings: [] });
  assert.equal(h.store.get().contour.declaredUnits, "kN_m_s");
  assert.equal(h.store.get().contour.sourceUnits, "kN_m_s");
  assert.equal(h.store.get().contour.targetUnits, "kN_m_s");

  await until(() => h.store.get().contour.scan?.status === "done", "the history scan after resultsLoaded");
  const vDeclared = contourView(h.store.get());
  assert.equal(vDeclared.state, "ready");
  if (vDeclared.state === "ready") {
    const lines = contourLines(h.store.get(), vDeclared);
    assert.equal(lines.title, "displacement · magnitude [m]");
    assert.equal(lines.units, "Units: kN-m-tonne-s (declared in file).");
  }

  // Converting a declared file needs only targetUnits; sourceUnits stays locked to declaredUnits.
  h.store.dispatch({ type: "setContourUnits", target: "N_mm_s", source: "kip_in_s" });
  assert.equal(h.store.get().contour.sourceUnits, "kN_m_s");
  assert.equal(h.store.get().contour.targetUnits, "N_mm_s");
  if (vDeclared.state === "ready") {
    const lines = contourLines(h.store.get(), vDeclared);
    assert.equal(lines.title, "displacement · magnitude [mm]");
    assert.equal(lines.units, "Units: N-mm-tonne-s (converted from declared kN-m-tonne-s).");
  }

  // Transitioning back to an undeclared results file resets declaredUnits/sourceUnits to null and targetUnits to as_stored.
  const infoUndeclared = { ...h.store.get().results!, units: null };
  h.store.dispatch({ type: "resultsLoaded", info: infoUndeclared, sizeBytes: 0, readMs: 0, warnings: [] });
  assert.equal(h.store.get().contour.declaredUnits, null);
  assert.equal(h.store.get().contour.sourceUnits, null);
  assert.equal(h.store.get().contour.targetUnits, "as_stored");

  // Re-load declared units, then open a different results path via fileOpened -> resets contour units to model's units (null).
  h.store.dispatch({ type: "resultsLoaded", info: infoWithUnits, sizeBytes: 0, readMs: 0, warnings: [] });
  assert.equal(h.store.get().contour.declaredUnits, "kN_m_s");
  h.store.dispatch({ type: "fileOpened", artifact: "results", path: "/another/file.ladruno" });
  assert.equal(h.store.get().contour.declaredUnits, null);
  assert.equal(h.store.get().contour.sourceUnits, null);
  assert.equal(h.store.get().contour.targetUnits, "as_stored");

  h.effects.dispose();
});
