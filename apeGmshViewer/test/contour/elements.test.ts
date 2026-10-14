// V4c2a (#1833) element (Gauss) contours, flat per cell, on fixtures written
// by the real NativeWriter (scripts/make_viewer_results_fixtures.py
// --elements), through the app's own renderer-side path (harness.ts). The
// closed form is the generator's, restated here (not copied from the app):
//
//   v[k, e, g] = eid_e + 0.1 g + 0.25 (c + 1)(k + 1)
//
// so a one-point cell is eid + 0.25 (c + 1)(k + 1), and the mean of an
// element's G points adds 0.05 (G - 1). The element of a drawn primitive is
// read from the model file (reader/node.ts blocks), not from the app's join.
//
// Mutations each test is meant to catch (the PR body lists the runs):
//   M1 drawing an unverified file      -> "unverified: refused ..."
//   M2 joining rows by position        -> "gauss_cell: per-cell colours ..."
//   M3 an unknown id mapped silently   -> "an _element_index id the mesh does not have ..."
//   M4 an average that drops a point   -> "element_mean: ..." and the gaussCellValues unit test

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { openModel } from "../../src/reader/node.ts";
import { contourShownKey, elementColours, toLinear } from "../../src/renderer/contour.ts";
import { elementMaskOf, visibleMaps } from "../../src/renderer/visible.ts";
import { gaussCellValues, gaussFields, joinElements, NO_DATA_RGB, rampT, sampleCmap } from "../../src/state/contour.ts";
import { contourFieldsOf, contourView, deformView, noticesOf } from "../../src/state/selectors.ts";
import type { GaussGroup, State } from "../../src/state/types.ts";
import { opened, STAGED, until } from "./harness.ts";

const fx = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const MIXED = fx("elem_mixed.h5");
const MIXED_RESULTS = fx("elem_mixed.results.h5");
const UNSTAMPED = fx("elem_mixed_unstamped.h5");
const UNSTAMPED_RESULTS = fx("elem_mixed_unstamped.results.h5");
const BADID = fx("elem_badid.h5");
const BADID_RESULTS = fx("elem_badid.results.h5");
const SOLID = fx("elem_solid.h5");
const SOLID_RESULTS = fx("elem_solid.results.h5");
const STAGED_ELEMENTS = fx("elem_staged.results.h5");

const PLANE = ["stress_xx", "stress_yy", "stress_xy"];
const close = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));

/** The closed form of one cell: the mean of its G points (G == 1: its one value). */
const cellValue = (eid: number, G: number, c: number, k: number) => eid + 0.05 * (G - 1) + 0.25 * (c + 1) * (k + 1);

/** The model's cells by FEM id and their nodes per element, read from the model file. */
async function cellsOf(path: string): Promise<Map<number, number>> {
  const m = await openModel(path);
  const out = new Map<number, number>();
  for (const b of m.blocks) b.ids.forEach((id) => out.set(id, b.npe));
  return out;
}

type Ready = Extract<ReturnType<typeof contourView>, { state: "element" }>;

/** Colour by an element field at results stage `stage`, step `k`, range `this step`; wait until it is drawn. */
async function pickElement(h: Awaited<ReturnType<typeof opened>>, field: string, component: string, stage: number, k: number): Promise<Ready> {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
  if (k) h.store.dispatch({ type: "setResultStep", step: k });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourRange", range: "step" });
  h.store.dispatch({ type: "setContourField", field, component, source: "element" });
  await until(() => {
    const v = contourView(h.store.get());
    const st = h.effects.currentStep();
    return v.state === "element" && v.stepMeasured && st?.step === k && st.element === component;
  }, `the element field ${component} at step ${k}`);
  return contourView(h.store.get()) as Ready;
}

/** The drawn maps of the state (the viewport's one visibility path, stage activation included). */
function drawnMaps(h: Awaited<ReturnType<typeof opened>>, s: State) {
  const m = s.mesh!;
  return visibleMaps(m.legend, s.visibility.hidden, h.blobs.i32(m.lineGroup), h.blobs.i32(m.triGroup), elementMaskOf(s, m, h.blobs));
}

/** Every drawn triangle's colour against the closed form at its element (by FEM id from the decl path), or no data. */
function checkTriangles(h: Awaited<ReturnType<typeof opened>>, v: Ready, want: (femId: number) => number | null): { coloured: number; noData: number } {
  const s = h.store.get();
  const { lineMap, triMap } = drawnMaps(h, s);
  const cc = elementColours(s, v, s.mesh!, h.blobs, h.effects.currentStep(), lineMap, triMap);
  assert.ok(cc, "colours for the step on screen");
  const triEl = h.blobs.i32(s.mesh!.triElement);
  let coloured = 0, noData = 0;
  for (let d = 0; d < triMap.length; d++) {
    const path = s.mesh!.elements[triEl[triMap[d]!]!]!;
    const femId = Number(/^mesh\/element\/(\d+)$/.exec(path)![1]);
    const w = want(femId);
    const rgb = w === null ? NO_DATA_RGB.map(toLinear) : sampleCmap(v.cmap, rampT(w, v.clim.min!, v.clim.max!)).map(toLinear);
    for (let vert = 0; vert < 3; vert++) {
      for (let c = 0; c < 3; c++) assert.ok(Math.abs(cc.tri[(d * 3 + vert) * 3 + c]! - rgb[c]!) < 1e-6, `triangle ${d} (element ${femId}) vertex ${vert}`);
    }
    if (w === null) noData++;
    else coloured++;
  }
  return { coloured, noData };
}

test("gauss_cell: per-cell colours equal the closed form on a quad plus triangle field, each group's cells from its own group; the quad no group has is no data", async () => {
  const h = await opened(MIXED, MIXED_RESULTS);
  const cells = await cellsOf(MIXED);
  const quads = [...cells].filter(([, n]) => n === 4).map(([id]) => id).sort((a, b) => a - b);
  const tris = [...cells].filter(([, n]) => n === 3).map(([id]) => id);
  assert.equal(quads.length, 8);
  assert.equal(tris.length, 8);
  const missing = quads[0]!;
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  const picker = contourFieldsOf(h.store.get())!;
  assert.deepEqual(picker.elements.map((f) => [f.name, f.components.map((c) => c.label), f.groups, f.refused]), [["stress", ["xx", "xy", "yy"], ["quads_G1", "tris_G1"], null]]);
  assert.equal(picker.elementsVerified, true);
  const k = 1;
  for (const [c, comp] of PLANE.entries()) {
    const v = await pickElement(h, "stress", comp, 0, k);
    assert.equal(v.topology, "gauss_cell");
    assert.match(v.topologyLabel, /^gauss_cell: each element's one Gauss-point value, flat per cell$/);
    assert.deepEqual(v.groups.map((g) => [g.id, g.nGauss]), [["quads_G1", 1], ["tris_G1", 1]]);
    // The step range: the closed form over every recorded element at this step.
    const all = [...quads.slice(1), ...tris].map((id) => cellValue(id, 1, c, k));
    assert.ok(close(v.clim.min!, Math.min(...all)) && close(v.clim.max!, Math.max(...all)), `${comp} step range`);
    const r = checkTriangles(h, v, (id) => (id === missing ? null : cells.has(id) ? cellValue(id, 1, c, k) : null));
    assert.ok(r.coloured > 0 && r.noData === 2, `${comp}: the missing quad's two triangles are no data (${r.noData})`);
  }
  h.effects.dispose();
});

test("element_mean: per-cell colours equal each element's mean of its 8 Gauss values on a solid; the history range is the closed form's", async () => {
  const h = await opened(SOLID, SOLID_RESULTS);
  const cells = await cellsOf(SOLID);
  assert.equal(cells.size, 12);
  const k = 2;
  const v = await pickElement(h, "stress", "stress_yy", 0, k);
  assert.equal(v.topology, "element_mean");
  assert.equal(v.topologyLabel, "element mean of 8 Gauss points, flat per cell");
  assert.ok(!/gauss_cell_averaged/.test(v.topologyLabel), "the per-element mean never carries Qt's gauss_cell_averaged name");
  const r = checkTriangles(h, v, (id) => cellValue(id, 8, 1, k));
  assert.ok(r.coloured > 0 && r.noData === 0);
  // The history scan reads the element field only, reduced per element.
  h.store.dispatch({ type: "setContourRange", range: "history" });
  await until(() => h.store.get().contour.scan?.status === "done", "the element history scan");
  const sc = h.store.get().contour.scan!;
  assert.match(sc.key, /\|elem:stress\|stress_yy$/);
  const ids = [...cells.keys()];
  const all = [0, 1, 2].flatMap((kk) => ids.map((id) => cellValue(id, 8, 1, kk)));
  assert.ok(close(sc.min!, Math.min(...all)) && close(sc.max!, Math.max(...all)), `scan ${sc.min}..${sc.max}`);
  h.effects.dispose();
});

test("unverified: an unstamped copy of the same file is refused with a banner, drawn as no data, and main never serves its element rows or values", async () => {
  const h = await opened(UNSTAMPED, UNSTAMPED_RESULTS);
  const s0 = h.store.get();
  assert.equal(s0.results!.elementIndex?.verified, false);
  assert.ok(s0.results!.stages[0]!.gauss.every((g) => g.elementIds === undefined), "no element ids reach the renderer");
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "stress", component: "stress_xx", source: "element" });
  await new Promise((r) => setTimeout(r, 50));
  const s = h.store.get();
  const v = contourView(s);
  assert.equal(v.state, "refused");
  assert.ok(v.state === "refused");
  assert.equal(v.code, "unverified");
  assert.match(v.reason, /elem_mixed_unstamped\.results\.h5 does not prove which elements its rows are, so its element results are not drawn \(the file has no @element_index_space/);
  assert.ok(noticesOf(s).some((n) => n.startsWith("Element contour refused: elem_mixed_unstamped.results.h5 does not prove which elements its rows are")));
  assert.equal(contourShownKey(v), `no-data:${v.reason}`);
  assert.equal(s.contour.join, null, "an unverified file is never joined");
  assert.notEqual(h.effects.currentStep()?.element, "stress_xx");
  // Main refuses the element step and scan of an unverified file on its own.
  const step = h.host.readStep({ path: UNSTAMPED_RESULTS, stage: "stage_0", step: 0, seq: 1, element: "stress_xx" });
  assert.ok(!step.ok && /element results of .* are refused: the file has no @element_index_space/.test(step.error));
  const scan = await h.host.scan({ path: UNSTAMPED_RESULTS, stage: "stage_0", components: ["stress_xx"], magnitude: false, scanId: 9, source: "element" }, () => undefined);
  assert.ok(!scan.ok && /are refused/.test(scan.error));
  const open = await h.host.open(UNSTAMPED_RESULTS);
  assert.ok(open.ok && Object.keys(open.elementIds).length === 0);
  h.effects.dispose();
});

test("a results info with no elementIndex verdict reads as unverified: its element fields are refused, never joined (fail closed)", async () => {
  const h = await opened(MIXED, MIXED_RESULTS);
  const info = h.store.get().results!;
  assert.equal(info.elementIndex?.verified, true);
  // The same info (element-id blobs included) with the verdict left out.
  const { elementIndex: _dropped, ...noVerdict } = info;
  h.store.dispatch({ type: "fileOpened", artifact: "results", path: info.path });
  h.store.dispatch({ type: "resultsLoaded", info: noVerdict, sizeBytes: 0, readMs: 0, warnings: [] });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "stress", component: "stress_xx", source: "element" });
  assert.equal(contourFieldsOf(h.store.get())!.elementsVerified, false);
  await new Promise((r) => setTimeout(r, 50));
  const v = contourView(h.store.get());
  assert.ok(v.state === "refused" && v.code === "unverified", `state ${v.state}`);
  assert.match(v.reason, /this results info carries no element-index verdict/);
  assert.equal(h.store.get().contour.join, null, "never joined");
  h.effects.dispose();
});

test("an _element_index id the mesh does not have refuses the field, with a typed state naming it; never mapped silently", async () => {
  const h = await opened(BADID, BADID_RESULTS);
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "stress", component: "stress_xx", source: "element" });
  await until(() => contourView(h.store.get()).state === "refused", "the join verdict");
  const v = contourView(h.store.get());
  assert.ok(v.state === "refused");
  assert.equal(v.code, "unknown-id");
  assert.match(v.reason, /_element_index names 1 element id the model has no cell for \(999999\)/);
  assert.ok(noticesOf(h.store.get()).some((n) => /Element contour refused: element field stress: _element_index names 1 element id/.test(n)));
  h.effects.dispose();
});

test("stage activation and the deformed shape compose: only drawn cells are coloured, each from its own row; the contour shows on the deformed mesh", async () => {
  const h = await opened(STAGED, STAGED_ELEMENTS);
  const cells = await cellsOf(STAGED);
  const stage = 0, k = 1; // base: Top is not active yet
  const v = await pickElement(h, "stress", "stress_xx", stage, k);
  const s = h.store.get();
  const { triMap } = drawnMaps(h, s);
  assert.ok(triMap.length < h.blobs.i32(s.mesh!.triElement).length, "stage base hides Top's faces");
  const r = checkTriangles(h, v, (id) => (cells.has(id) ? cellValue(id, 1, 0, k) + stage : null));
  assert.equal(r.noData, 0);
  assert.equal(r.coloured, triMap.length);
  // V4d: the deformed shape on a 2-D model with an element contour.
  const m = await opened(MIXED, MIXED_RESULTS);
  await pickElement(m, "stress", "stress_xx", 0, 2);
  m.store.dispatch({ type: "setDeform", on: true });
  await until(() => deformView(m.store.get()).state === "ready", "the deformed shape");
  assert.equal(contourView(m.store.get()).state, "element");
  h.effects.dispose();
  m.effects.dispose();
});

test("a re-read refreshes what is drawn: the colour key and the join carry the results-read count", async () => {
  const h = await opened(MIXED, MIXED_RESULTS);
  const v1 = await pickElement(h, "stress", "stress_xx", 0, 0);
  assert.ok(await h.effects.openResults(MIXED_RESULTS));
  await until(() => {
    const v = contourView(h.store.get());
    return v.state === "element" && v.stepMeasured && v.reads === v1.reads + 1 && h.effects.currentStep()?.element === "stress_xx";
  }, "the re-read joined and measured again");
  const v2 = contourView(h.store.get()) as Ready;
  assert.notEqual(v2.joinKey, v1.joinKey);
  assert.notEqual(contourShownKey(v2), contourShownKey(v1));
  assert.equal(h.store.get().contour.join?.key, v2.joinKey);
  h.effects.dispose();
});

test("V4c2b: no draw mode reads 'not yet available' any more; each node-located mode draws on this solid with its own topology", async () => {
  const h = await opened(SOLID, SOLID_RESULTS);
  await pickElement(h, "stress", "stress_xx", 0, 0);
  for (const [mode, topology] of [["cell_averaged", "element_mean_averaged"], ["node", "gauss_node"], ["node_discrete", "gauss_node_discrete"]] as const) {
    h.store.dispatch({ type: "setContourGauss", mode });
    await until(() => contourView(h.store.get()).state === "element", mode);
    const v = contourView(h.store.get());
    assert.ok(v.state === "element" && v.topology === topology, mode);
    assert.ok(!/not yet available/.test(v.topologyLabel));
  }
  h.effects.dispose();
});

test("groups that disagree on a field's components refuse it (no component is taken from the other group); a partial write too", async () => {
  const h = await opened(MIXED, MIXED_RESULTS);
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  const info = h.store.get().results!;
  const drop = (g: GaussGroup): GaussGroup => (g.id === "tris_G1" ? { ...g, components: g.components.filter((c) => c !== "stress_xy") } : g);
  for (const [what, edit, pattern] of [
    ["missing", drop, /group tris_G1 has no stress_xy/],
    ["refused", (g: GaussGroup) => (g.id === "tris_G1" ? { ...drop(g), refused: [{ component: "stress_xy", reason: "a partial write" }] } : g), /group tris_G1 refused stress_xy \(a partial write\)/],
  ] as const) {
    h.store.dispatch({ type: "fileOpened", artifact: "results", path: info.path });
    h.store.dispatch({ type: "resultsLoaded", info: { ...info, stages: info.stages.map((st) => ({ ...st, gauss: st.gauss.map(edit) })) }, sizeBytes: 0, readMs: 0, warnings: [] });
    h.store.dispatch({ type: "setContourField", field: "stress", component: "stress_xx", source: "element" });
    const v = contourView(h.store.get());
    assert.ok(v.state === "refused" && v.code === "components", what);
    assert.match(v.reason, pattern);
  }
  h.effects.dispose();
});

test("gaussCellValues: the mean of every Gauss point; any non-finite point makes the element no data, never the mean of the rest", () => {
  const v = gaussCellValues(Float64Array.from([1, 2, 3, 10, 4, 5, NaN, 6]), 2, 4);
  assert.deepEqual(Array.from(v.subarray(0, 1)), [4]);
  assert.ok(Number.isNaN(v[1]!));
  assert.deepEqual(Array.from(gaussCellValues(Float64Array.from([7, Infinity]), 2, 1)).map(Number.isNaN), [false, true]);
  assert.throws(() => gaussCellValues(new Float64Array(0), 0, 0), /0 Gauss points per element/);
  assert.throws(() => gaussCellValues(new Float64Array(3), 2, 2), /3 values for 2 elements x 2 Gauss points/);
});

test("gaussFields: component names from the file, [k] for unnamed ones, local axes from the name; mixed Gauss counts average", () => {
  const g = (id: string, components: string[], nGauss = 1): GaussGroup => ({ id, components, refused: [], nElements: 1, nGauss, localAxes: false, naturalCoords: null });
  const f = gaussFields([g("a", ["stress_xx", "stress_yy", "footfall_ap", "membrane_force_local_xx", "weird_"], 4), g("b", ["stress_xx", "stress_yy"], 1)]);
  assert.deepEqual(
    f.map((x) => [x.name, x.components.map((c) => c.label), x.nGauss, x.refused === null]),
    [
      ["stress", ["xx", "yy"], [1, 4], true],
      ["footfall_ap", ["footfall_ap"], [4], true],
      ["membrane_force_local", ["xx (local)"], [4], true],
      ["weird", ["[0]"], [4], true],
    ],
  );
  assert.equal(f[2]!.components[0]!.local, true);
});

test("joinElements joins by FEM id only: a duplicate row is refused, and an unknown id is named", () => {
  const decls = Object.create(null) as Record<string, { element: { femId: number } }>;
  for (const id of [5, 9]) decls[`mesh/element/${id}`] = { element: { femId: id } };
  const ok = joinElements(decls as never, [{ id: "g", eids: Float64Array.from([9, 5]) }]);
  assert.ok(ok.ok && ok.at.get(9)!.row === 0 && ok.at.get(5)!.row === 1);
  const dup = joinElements(decls as never, [{ id: "a", eids: Float64Array.from([5]) }, { id: "b", eids: Float64Array.from([5]) }]);
  assert.ok(!dup.ok && dup.code === "duplicate-id" && /element 5 has two rows/.test(dup.reason));
  const unk = joinElements(decls as never, [{ id: "a", eids: Float64Array.from([5, 1, 2, 3, 4, 6, 7]) }]);
  assert.ok(!unk.ok && unk.code === "unknown-id" && /names 6 element ids the model has no cell for \(1, 2, 3, 4, 6, \.\.\.\)/.test(unk.reason));
});
