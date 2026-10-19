// V4d (#1765): the deformed shape. Oracles on the committed fixtures, through
// the app's own renderer-side path (test/contour/harness.ts: the real
// Effects over main's readers):
//
// - deformed vertex positions equal x + s u from the generator's closed form,
//   at two steps and two scales (contour_staged, 3-D);
// - a 2-D model (deform_2d, ndm 2) keeps every vertex's z, and its truss
//   chords stay straight segments between the displaced end nodes;
// - coincident nodes deform independently (each vertex through its own node);
// - a drawn node with no displacement is "partial": counted, drawn
//   undeformed, never moved by a displacement of 0; an inactive element's
//   nodes do not count (stage activation), nor do a hidden group's;
// - a refused displacement axis refuses the deformed shape (never drawn from
//   the surviving axes), and so do unnamed components and a missing z in a
//   model not declared 2-D;
// - a results re-read at the same path deforms again (the key carries the
//   results-read count);
// - a pick on the deformed mesh selects the element drawn there.

import assert from "node:assert/strict";
import { copyFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { deformLines } from "../../src/panels/legend.ts";
import { openModel } from "../../src/reader/node.ts";
import { basePositions, deformFull, edgesShown, gatherInto, positionsFor, writePositions } from "../../src/renderer/deform.ts";
import { hitElement } from "../../src/renderer/selection.ts";
import type { ModelFile } from "../../src/model/types.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import * as h5wasm from "h5wasm/node";
import { columnOf, nodalFields } from "../../src/state/contour.ts";
import { displacementAxes, LADRUNO_TRANSLATION, modeNote, spatialNdm, vertexDisplacements } from "../../src/state/deform.ts";
import { loadModel } from "../../src/state/load.ts";
import { deformView, noticesOf } from "../../src/state/selectors.ts";
import type { ResultsStage, SpatialNdm } from "../../src/state/types.ts";
import { closedForm, harness, NODATA, NODATA_RESULTS, opened, STAGED, STAGED_RESULTS, STAGES, until } from "../contour/harness.ts";

const DEFORM_2D = fileURLToPath(new URL("../../fixtures/deform_2d.h5", import.meta.url));
const DEFORM_2D_RESULTS = fileURLToPath(new URL("../../fixtures/deform_2d.results.h5", import.meta.url));

/** The generator's 2-D closed form (scripts/make_viewer_results_fixtures.py `deform_values`), restated. */
const closed2d = (x: number, y: number, k: number): [number, number] => {
  const s = 0.01 * (k + 1);
  return [s * y, s * (0.5 * x + 0.25 * y)];
};

/** Node id -> [x, y, z], read from the model file (not from the app's derivations). */
async function coordsOf(path: string): Promise<Map<number, [number, number, number]>> {
  const m = await openModel(path);
  const out = new Map<number, [number, number, number]>();
  m.nodeIds.forEach((id, i) => out.set(id, [m.nodeCoords[3 * i]!, m.nodeCoords[3 * i + 1]!, m.nodeCoords[3 * i + 2]!]));
  return out;
}

const near = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));

type H = Awaited<ReturnType<typeof opened>>;

/** Show results stage `stage` at step `k`, deformed at `scale`, and wait until the effects computed and counted it. */
async function showDeformed(h: H, stage: number, k: number, scale: number) {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
  if (k !== 0) h.store.dispatch({ type: "setResultStep", step: k });
  h.store.dispatch({ type: "setDeform", on: true });
  h.store.dispatch({ type: "setDeformScale", scale });
  await until(() => {
    const v = deformView(h.store.get());
    return (v.state === "ready" || v.state === "partial") && h.effects.currentDisplacement()?.key === v.keys.disp;
  }, `the deformed shape of stage ${stage} step ${k}`);
  const s = h.store.get();
  const full = positionsFor(s, s.mesh!, h.blobs, h.effects.currentDisplacement(), null)!;
  return { s, v: deformView(s), full };
}

const stage = (over: Partial<ResultsStage>): ResultsStage => ({
  id: "stage_0", name: "pushover", kind: "static", time: [0, 1], nodal: [], refusedNodal: [], gauss: [], unread: [], ...over,
});
const NDM3: SpatialNdm = { ndm: 3, source: "test" };
const NDM2: SpatialNdm = { ndm: 2, source: "test" };
const NONE: SpatialNdm = { ndm: null, reason: "the model declares no ndm (/meta/ndm 0, a broker-only file)" };

test("displacement axes: x, y, z by name; rotations ignored; never an assumed axis", () => {
  const xyz = ["displacement_x", "displacement_y", "displacement_z"];
  const ok = displacementAxes(stage({ nodal: [...xyz, "rotation_x", "rotation_y", "rotation_z"] }), NDM3);
  assert.deepEqual(ok, { ok: true, datasets: xyz, note: null }, "rotations are not part of the deformed shape");
  // The file's order does not matter: the names say the axes.
  assert.deepEqual(displacementAxes(stage({ nodal: ["displacement_z", "displacement_x", "displacement_y"] }), NONE), { ok: true, datasets: xyz, note: null });
  // A translational family that is not the displacement is not deformed by.
  const inc = displacementAxes(stage({ nodal: ["displacement_increment_x", "velocity_x"] }), NDM3);
  assert.equal(inc.ok, false);
  assert.equal(!inc.ok && inc.kind, "absent");
  // Unnamed components (`[k]`): the axis mapping is ambiguous, so refuse, even beside named ones.
  for (const nodal of [["displacement_", "displacement_", "displacement_"], ["displacement_x", "displacement_y", "displacement_"], ["displacement"]]) {
    const r = displacementAxes(stage({ nodal }), NDM3);
    assert.ok(!r.ok && r.kind === "refused" && /name no axis.*ambiguous/.test(r.reason), `${nodal}: ${JSON.stringify(r)}`);
  }
});

test("2-D: a missing displacement_z is 0 only in a model declared ndm 2; otherwise the deformed shape refuses", () => {
  const xy = stage({ nodal: ["displacement_x", "displacement_y"] });
  const two = displacementAxes(xy, NDM2);
  assert.ok(two.ok);
  assert.deepEqual(two.datasets, ["displacement_x", "displacement_y", null]);
  assert.match(two.note!, /2-D model \(ndm 2/);
  for (const ndm of [NDM3, NONE]) {
    const r = displacementAxes(xy, ndm);
    assert.ok(!r.ok && r.kind === "refused" && /no displacement_z and the model is not declared 2-D.*would be a guess/.test(r.reason), JSON.stringify(r));
  }
  const xOnly = displacementAxes(stage({ nodal: ["displacement_x"] }), NDM2);
  assert.ok(!xOnly.ok && xOnly.kind === "refused" && /no displacement_y/.test(xOnly.reason));
  // /meta/ndm is the declared dimension from neutral 2.34.0 on; before, the mesh dimension; 0 declares none.
  assert.deepEqual(spatialNdm({ ndm: 2 }, "2.35.0"), { ndm: 2, source: "/meta/ndm, neutral 2.35.0" });
  assert.deepEqual(spatialNdm({ ndm: 3 }, "3.0.0"), { ndm: 3, source: "/meta/ndm, neutral 3.0.0" });
  assert.equal(spatialNdm({ ndm: 2 }, "2.33.9").ndm, null);
  assert.equal(spatialNdm({ ndm: 0 }, "2.35.0").ndm, null);
  assert.equal(spatialNdm({}, "2.35.0").ndm, null);
});

test("a refused displacement axis refuses the deformed shape: never drawn from the surviving axes, in 3-D or 2-D", () => {
  const refused = [{ component: "displacement_z", reason: "a partial write" }];
  for (const ndm of [NDM3, NDM2]) {
    const r = displacementAxes(stage({ nodal: ["displacement_x", "displacement_y"], refusedNodal: refused }), ndm);
    assert.ok(!r.ok && r.kind === "refused", JSON.stringify(r));
    assert.match(!r.ok ? r.reason : "", /^displacement_z not read \(a partial write\); the deformed shape is never drawn from the axes that survive$/);
  }
});

test("vertex displacement: a node absent from _ids, or NaN on an axis, is missing (flagged, never a displacement of 0)", () => {
  const nodes = Float64Array.from([1, 2, 3, 4, 9]);
  const ids = Float64Array.from([1, 2, 3, 4]);
  const rx = Float64Array.from([0.1, 0.2, NaN, 0.4]);
  const ry = Float64Array.from([1, 2, 3, 4]);
  const rz = Float64Array.from([5, 6, 7, NaN]);
  const d = vertexDisplacements(nodes, columnOf(ids), [rx, ry, rz]);
  assert.deepEqual(Array.from(d.missing), [0, 0, 1, 1, 1], "NaN x (node 3), NaN z (node 4), absent (node 9)");
  assert.equal(d.count, 3);
  assert.deepEqual(Array.from(d.d.subarray(0, 6)), [0.1, 1, 5, 0.2, 2, 6]);
  // In 2-D (no z row) a NaN z cannot occur, and z is 0 by definition.
  const d2 = vertexDisplacements(nodes, columnOf(ids), [rx, ry, null]);
  assert.deepEqual(Array.from(d2.missing), [0, 0, 1, 0, 1]);
  assert.deepEqual(Array.from(d2.d.subarray(9, 12)), [0.4, 4, 0]);
});

test("oracle (3-D): every deformed vertex is x + s u of its node, at two steps and two scales", async () => {
  const h = await opened();
  const coords = await coordsOf(STAGED);
  const st = 2;
  for (const k of [1, 3]) {
    for (const scale of [1, 25]) {
      const { s, v, full } = await showDeformed(h, st, k, scale);
      assert.equal(v.state, "ready");
      assert.equal(full.key, `${h.effects.currentDisplacement()!.key}|scale ${scale}`);
      for (const [pos, vn] of [[full.tri, s.mesh!.vertexNodes.tri], [full.line, s.mesh!.vertexNodes.line], [full.edge, s.mesh!.vertexNodes.edge]] as const) {
        if (vn !== s.mesh!.vertexNodes.line) assert.ok(h.blobs.get(vn).length > 0, "the faces and the outline edges have vertices (this model has no lines)");
        const nodes = h.blobs.get(vn) as Float64Array;
        nodes.forEach((id, i) => {
          const [x, y, z] = coords.get(id)!;
          const u = closedForm(x, z, st, k, STAGES[st]!.T);
          const want = [x, y, z].map((c, a) => Math.fround(Math.fround(c) + scale * u[a]!));
          for (let a = 0; a < 3; a++) assert.ok(near(pos[3 * i + a]!, want[a]!), `k ${k} scale ${scale} vertex ${i} node ${id} axis ${a}: ${pos[3 * i + a]} vs ${want[a]}`);
        });
      }
    }
  }
  // Scale 1 is true scale; the legend states any other scale.
  assert.equal(deformLines(h.store.get()).scale, "scale ×25 (not true scale)");
  h.store.dispatch({ type: "setDeformScale", scale: 1 });
  assert.equal(deformLines(h.store.get()).scale, "true scale (×1)");
  assert.throws(() => h.store.dispatch({ type: "setDeformScale", scale: 0 }), RangeError);
  assert.throws(() => h.store.dispatch({ type: "setDeformScale", scale: Number.NaN }), RangeError);
  h.effects.dispose();
});

test("oracle (2-D, ndm 2): z stays at the node's z, x and y follow the closed form, truss chords stay straight", async () => {
  const h = await opened(DEFORM_2D, DEFORM_2D_RESULTS);
  assert.deepEqual(h.store.get().ndm.ndm, 2, JSON.stringify(h.store.get().ndm));
  const coords = await coordsOf(DEFORM_2D);
  const scale = 4, k = 2;
  const { s, v, full } = await showDeformed(h, 0, k, scale);
  assert.equal(v.state, "ready");
  assert.ok(v.state === "ready" && v.datasets[2] === null && /2-D model/.test(v.note!));
  const base = basePositions(s.mesh!, h.blobs);
  for (const [pos, b, vn] of [[full.tri, base.tri, s.mesh!.vertexNodes.tri], [full.line, base.line, s.mesh!.vertexNodes.line]] as const) {
    const nodes = h.blobs.get(vn) as Float64Array;
    assert.ok(nodes.length > 0);
    nodes.forEach((id, i) => {
      const [x, y] = coords.get(id)!;
      const [ux, uy] = closed2d(x, y, k);
      assert.ok(near(pos[3 * i]!, Math.fround(Math.fround(x) + scale * ux)), `x of node ${id}`);
      assert.ok(near(pos[3 * i + 1]!, Math.fround(Math.fround(y) + scale * uy)), `y of node ${id}`);
      assert.equal(pos[3 * i + 2], b[3 * i + 2], `z of node ${id} is the node's own`);
    });
  }
  // The trusses: one segment per pair of end nodes, as undeformed (nothing inserted, nothing curved).
  assert.equal(full.line.length, base.line.length);
  assert.ok(s.mesh!.counts.lineCells >= 4);
  h.effects.dispose();
});

test("coincident nodes deform independently: each vertex moves by its own node's displacement", () => {
  // A decoupled interface (2 on 5, 3 on 8) and a zero-length spring (13 on 14), as V4c1's coincident fixture.
  const XYZ: Record<number, [number, number, number]> = {
    1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 1, 0], 4: [0, 1, 0], 5: [1, 0, 0], 6: [2, 0, 0], 7: [2, 1, 0], 8: [1, 1, 0], 13: [3, 0, 0], 14: [3, 0, 0],
  };
  const ids = Object.keys(XYZ).map(Number);
  const model: ModelFile = {
    path: "coincident.h5", sizeBytes: 0, neutralVersion: "2.35.0", meta: {},
    nodeIds: new Float64Array(ids), nodeCoords: new Float64Array(ids.flatMap((i) => XYZ[i]!)),
    blocks: [
      { alias: "quad4", code: 3, dim: 2, npe: 4, ids: new Float64Array([1, 2]), connectivity: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]) },
      { alias: "line2", code: 1, dim: 1, npe: 2, ids: new Float64Array([3]), connectivity: new Float64Array([13, 14]) },
    ],
    physicalGroups: [], labels: [], opensees: null, sequence: { status: "absent" }, warnings: [], readMs: 0,
  };
  const blobs = new BlobStore();
  const load = loadModel(model, blobs);
  const u = (id: number): [number, number, number] => [id, 10 * id, -id];
  const col = columnOf(Float64Array.from(ids));
  const rows = [0, 1, 2].map((a) => Float64Array.from(ids, (id) => u(id)[a]!)) as [Float64Array, Float64Array, Float64Array];
  for (const [vn, pos] of [[load.mesh.vertexNodes.tri, load.mesh.triPositions], [load.mesh.vertexNodes.line, load.mesh.linePositions]] as const) {
    const nodes = blobs.get(vn) as Float64Array;
    const d = vertexDisplacements(nodes, col, rows);
    assert.equal(d.count, 0);
    const base = blobs.f32(pos);
    nodes.forEach((id, i) => {
      for (let a = 0; a < 3; a++) assert.equal(d.d[3 * i + a], u(id)[a], `vertex ${i} (node ${id}) axis ${a}`);
      assert.deepEqual([base[3 * i], base[3 * i + 1], base[3 * i + 2]], XYZ[id]!.map(Math.fround));
    });
  }
  for (const [a, b] of [[2, 5], [3, 8], [13, 14]] as const) assert.notDeepEqual(u(a), u(b));
});

test("outline edges follow their own nodes (coincident nodes included), are drawn on the deformed mesh, and move in place", () => {
  // Two quads side by side whose shared edge is a decoupled interface: 2 sits on 5, 3 on 8.
  const XYZ: Record<number, [number, number, number]> = {
    1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 1, 0], 4: [0, 1, 0], 5: [1, 0, 0], 6: [2, 0, 0], 7: [2, 1, 0], 8: [1, 1, 0],
  };
  const ids = Object.keys(XYZ).map(Number);
  const model: ModelFile = {
    path: "edges.h5", sizeBytes: 0, neutralVersion: "2.35.0", meta: {},
    nodeIds: new Float64Array(ids), nodeCoords: new Float64Array(ids.flatMap((i) => XYZ[i]!)),
    blocks: [{ alias: "quad4", code: 3, dim: 2, npe: 4, ids: new Float64Array([1, 2]), connectivity: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]) }],
    physicalGroups: [], labels: [], opensees: null, sequence: { status: "absent" }, warnings: [], readMs: 0,
  };
  const blobs = new BlobStore();
  const mesh = loadModel(model, blobs).mesh;
  const edgeNodes = blobs.get(mesh.vertexNodes.edge) as Float64Array;
  // The perimeter of each quad, in connectivity order, each edge once: both sides of the interface are kept.
  assert.deepEqual(Array.from(edgeNodes), [1, 2, 2, 3, 3, 4, 4, 1, 5, 6, 6, 7, 7, 8, 8, 5]);
  const base = basePositions(mesh, blobs);
  edgeNodes.forEach((id, i) => assert.deepEqual(Array.from(base.edge.subarray(3 * i, 3 * i + 3)), XYZ[id]!.map(Math.fround), `edge vertex ${i} sits on node ${id}`));
  // Each node its own displacement; coincident pairs differ.
  const u = (id: number): [number, number, number] => [0.1 * id, -0.05 * id, 0.02 * id];
  const col = columnOf(Float64Array.from(ids));
  const rows = [0, 1, 2].map((a) => Float64Array.from(ids, (id) => u(id)[a]!)) as [Float64Array, Float64Array, Float64Array];
  const disp = {
    key: "k",
    line: vertexDisplacements(blobs.get(mesh.vertexNodes.line) as Float64Array, col, rows),
    tri: vertexDisplacements(blobs.get(mesh.vertexNodes.tri) as Float64Array, col, rows),
    edge: vertexDisplacements(edgeNodes, col, rows),
  };
  const scale = 3;
  const full = deformFull(base, disp, scale, "k|scale 3");
  edgeNodes.forEach((id, i) => {
    for (let a = 0; a < 3; a++) assert.equal(full.edge[3 * i + a], Math.fround(Math.fround(XYZ[id]![a]!) + scale * u(id)[a]!), `edge vertex ${i} (node ${id}) axis ${a}`);
  });
  // The interface edge 2-3 and its twin 5-8 sit together undeformed and apart deformed.
  assert.notDeepEqual(Array.from(full.edge.subarray(3, 6)), Array.from(full.edge.subarray(45, 48)));
  // Shown on the deformed mesh when the user's toggle is on; hidden only by the toggle.
  assert.equal(edgesShown(true, full), true);
  assert.equal(edgesShown(false, full), false);
  // writePositions moves the drawn edges in place, through the drawn-edge map.
  const map = Int32Array.from([0, 2, 7]);
  const object = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(gatherInto(base.edge, map, 6), 3)));
  writePositions(null, null, full, { lineMap: new Int32Array(0), triMap: new Int32Array(0) }, { object, map });
  assert.deepEqual(Array.from(object.geometry.getAttribute("position").array as Float32Array), Array.from(gatherInto(full.edge, map, 6)));
});

/** The drawn tri vertices of nodes with z > 3 (the Top box's own nodes, which contour_nodata's results do not record). */
async function topOnly(h: H): Promise<{ ids: Set<number>; drawnTotal: number }> {
  const coords = await coordsOf(NODATA);
  const s = h.store.get();
  const tri = h.blobs.get(s.mesh!.vertexNodes.tri) as Float64Array;
  const ids = new Set<number>();
  const all = new Set<number>();
  tri.forEach((id) => {
    all.add(id);
    if (coords.get(id)![2] > 3 + 1e-9) ids.add(id);
  });
  return { ids, drawnTotal: all.size };
}

test("partial: drawn nodes with no displacement are counted, stated in the banner, and drawn undeformed; inactive or hidden ones do not count", async () => {
  const h = await opened(NODATA, NODATA_RESULTS);
  const top = await topOnly(h);
  assert.ok(top.ids.size > 0);
  // Stage erect: Top is active, so its own nodes are drawn and have no displacement.
  const { s, v, full } = await showDeformed(h, 1, 1, 10);
  assert.equal(v.state, "partial");
  assert.ok(v.state === "partial");
  assert.equal(v.missing, top.ids.size, "each drawn node with no displacement, once");
  assert.equal(v.drawn, top.drawnTotal);
  assert.ok(noticesOf(s).includes(
    `Deformed shape: ${top.ids.size} of ${top.drawnTotal} drawn nodes have no displacement at this step (not recorded, or NaN): ` +
      "they are drawn undeformed and marked, never moved by a displacement of 0.",
  ));
  assert.match(deformLines(s).partial!, new RegExp(`^${top.ids.size} of ${top.drawnTotal} drawn nodes have no displacement`));
  const base = basePositions(s.mesh!, h.blobs);
  const tri = h.blobs.get(s.mesh!.vertexNodes.tri) as Float64Array;
  let moved = 0;
  tri.forEach((id, i) => {
    const same = [0, 1, 2].every((a) => full.tri[3 * i + a] === base.tri[3 * i + a]);
    if (top.ids.has(id)) assert.ok(same, `node ${id} (no displacement) is drawn undeformed`);
    else if (!same) moved++;
  });
  assert.ok(moved > 0, "the recorded nodes move");
  // Hide Top: none of its nodes is drawn, so the deformed shape is complete again.
  const topGroup = Object.keys(s.decls).find((p) => p.endsWith("/physical_group/Top"))!;
  h.store.dispatch({ type: "setHidden", decls: [topGroup], hidden: true });
  await until(() => deformView(h.store.get()).state !== "loading", "the count with Top hidden");
  assert.equal(deformView(h.store.get()).state, "ready");
  h.store.dispatch({ type: "setHidden", decls: [topGroup], hidden: false });
  // Stage base: Top is not activated yet, so its nodes are not drawn and nothing is missing.
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  await until(() => {
    const x = deformView(h.store.get());
    return x.state === "ready" || x.state === "partial";
  }, "the count on stage base");
  assert.equal(deformView(h.store.get()).state, "ready", "an inactive element's nodes are not drawn nodes");
  h.effects.dispose();
});

test("a refused axis in the app: the deformed shape is refused with the reason, nothing is computed, the mesh stays undeformed", async () => {
  const h = await opened();
  const s0 = h.store.get();
  const info = {
    ...s0.results!,
    stages: s0.results!.stages.map((x, i) => (i === 2 ? { ...x, nodal: x.nodal.filter((c) => c !== "displacement_z"), refusedNodal: [{ component: "displacement_z", reason: "a partial write" }] } : x)),
  };
  h.store.dispatch({ type: "resultsLoaded", info, sizeBytes: 0, readMs: 0, warnings: [] });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 2, step: 0 } });
  h.store.dispatch({ type: "setDeform", on: true });
  await until(() => h.store.get().resultStep?.status === "ready", "the step");
  const s = h.store.get();
  const v = deformView(s);
  assert.equal(v.state, "refused");
  assert.ok(noticesOf(s).includes("Deformed shape not drawn: displacement_z not read (a partial write); the deformed shape is never drawn from the axes that survive."));
  assert.equal(h.effects.currentDisplacement(), null, "no displacement is computed from x and y");
  assert.equal(positionsFor(s, s.mesh!, h.blobs, h.effects.currentDisplacement(), null)!.key, "base");
  assert.equal(deformLines(s).notes[0]!.refused, true);
  h.effects.dispose();
});

test("reload: a re-read of the same path deforms again (the key carries the results-read count)", async () => {
  const dir = mkdtempSync(join(tmpdir(), "v4d-"));
  const path = join(dir, "same.results.h5");
  copyFileSync(STAGED_RESULTS, path);
  const h = await opened(STAGED, path);
  const first = await showDeformed(h, 1, 1, 10);
  assert.equal(first.v.state, "ready");
  // The same path now holds results that miss the Top box's own nodes.
  copyFileSync(NODATA_RESULTS, path);
  await h.effects.openResults(path);
  await until(() => {
    const v = deformView(h.store.get());
    return (v.state === "ready" || v.state === "partial") && h.effects.currentDisplacement()?.key === v.keys.disp;
  }, "the re-read step");
  const v = deformView(h.store.get());
  assert.equal(v.state, "partial", "the re-read data is what is drawn");
  assert.notEqual(h.effects.currentDisplacement()!.key, first.full.disp!.key);
  h.effects.dispose();
});

test("pick: a click on the deformed mesh selects the element drawn there (faces and truss chords)", async () => {
  const h = await opened(DEFORM_2D, DEFORM_2D_RESULTS);
  const { s, full } = await showDeformed(h, 0, 2, 30);
  const info = s.mesh!;
  const drawn = { lineMap: Int32Array.from({ length: info.lineElement.shape[0]! }, (_, i) => i), triMap: Int32Array.from({ length: info.triElement.shape[0]! }, (_, i) => i) };
  const base = basePositions(info, h.blobs);
  // The objects as the viewport builds them, from the undeformed positions; then moved in place.
  const faces = new THREE.Mesh(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(gatherInto(base.tri, drawn.triMap, 9), 3)), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  const lg = new LineSegmentsGeometry();
  lg.setPositions(gatherInto(base.line, drawn.lineMap, 6));
  const lm = new LineMaterial({ linewidth: 3, worldUnits: false });
  lm.resolution.set(800, 800);
  const lines = new LineSegments2(lg, lm);
  const camera = new THREE.OrthographicCamera(-4, 4, 4, -4, 0.1, 100);
  camera.position.set(1.5, 1.5, 10);
  camera.lookAt(1.5, 1.5, 0);
  camera.updateMatrixWorld();
  faces.updateMatrixWorld();
  lines.updateMatrixWorld();
  const lineElement = h.blobs.i32(info.lineElement), triElement = h.blobs.i32(info.triElement);
  const ray = new THREE.Raycaster();
  ray.params.Line2 = { threshold: 7 };
  const at = (p: THREE.Vector3) => {
    const ndc = p.clone().project(camera);
    ray.setFromCamera(new THREE.Vector2(ndc.x, ndc.y), camera);
    return hitElement(ray, lines, faces, drawn, info.elements, lineElement, triElement, info.radius);
  };
  // Targets: the deformed centroid of the last face triangle, and the deformed midpoint of the last truss segment.
  const t = drawn.triMap.length - 1;
  const centroid = new THREE.Vector3(...[0, 1, 2].map((a) => (full.tri[9 * t + a]! + full.tri[9 * t + 3 + a]! + full.tri[9 * t + 6 + a]!) / 3));
  const g = drawn.lineMap.length - 1;
  const mid = new THREE.Vector3(...[0, 1, 2].map((a) => (full.line[6 * g + a]! + full.line[6 * g + 3 + a]!) / 2));
  const triWant = info.elements[triElement[t]!]!, lineWant = info.elements[lineElement[g]!]!;
  // Undeformed, those points are not on those elements: the test can tell the two shapes apart.
  assert.notEqual(info.elements[at(centroid)?.element ?? -1], triWant);
  assert.notEqual(info.elements[at(mid)?.element ?? -1], lineWant);
  writePositions(faces, lines, full, drawn);
  assert.equal(info.elements[at(centroid)!.element], triWant, "the face drawn at the deformed centroid");
  assert.equal(info.elements[at(mid)!.element], lineWant, "the truss chord drawn at the deformed midpoint");
  h.effects.dispose();
});

// -- V4L-b2 (#2128): a .ladruno's translation, by the fork schema's names (§7.1.1) --
//
// Fork `ladruno_schema_v1.md` §7.1.1 (fork 3b17c32f): translation and rotation are
// separate ON_NODES groups. `DISPLACEMENT` (a mode stage: `MODES_OF_VIBRATION(U)`)
// is `Ux` / `Ux,Uy` / `Ux,Uy,Uz` at ndm 1 / 2 / 3; `ROTATION` is `Rx,Ry,Rz` / `Rz` /
// `Rx,Ry,Rz`. Deform reads the translation group only.

const L_FIX = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));
const FRAME3D_MODEL = L_FIX("ladruno_frame3d.h5");
const FRAME3D = L_FIX("ladruno_frame3d.ladruno");
const MODES = L_FIX("ladruno_modes.ladruno");
const PORTAL3D = L_FIX("ladruno_portal3d.ladruno");
const D3 = ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "DISPLACEMENT/Uz"];

test("ladruno: DISPLACEMENT/Ux,Uy,Uz are x, y, z; ROTATION is never read; the group names the schema section", () => {
  assert.deepEqual(LADRUNO_TRANSLATION, { static: "DISPLACEMENT", transient: "DISPLACEMENT", mode: "MODES_OF_VIBRATION(U)" });
  const ok = displacementAxes(stage({ nodal: [...D3, "ROTATION/Rx", "ROTATION/Ry", "ROTATION/Rz"] }), NDM3);
  assert.deepEqual(ok, { ok: true, datasets: D3, note: null });
  // The file's column order does not matter: the labels say the axes.
  assert.deepEqual(displacementAxes(stage({ nodal: ["ROTATION/Rz", "DISPLACEMENT/Uz", "DISPLACEMENT/Ux", "DISPLACEMENT/Uy"] }), NDM3), ok);
  // 2-D (ndm 2): `Ux,Uy`, z is 0 only in a model declared 2-D.
  const two = displacementAxes(stage({ nodal: ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "ROTATION/Rz"] }), NDM2);
  assert.ok(two.ok);
  assert.deepEqual(two.datasets, ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", null]);
  for (const ndm of [NDM3, NONE]) {
    const r = displacementAxes(stage({ nodal: ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy"] }), ndm);
    assert.ok(!r.ok && r.kind === "refused" && /records no DISPLACEMENT\/Uz and the model is not declared 2-D/.test(r.reason), JSON.stringify(r));
  }
  // A stage that records only rotations has no displacement (never "rotation as translation").
  const rotOnly = displacementAxes(stage({ nodal: ["ROTATION/Rx", "ROTATION/Ry", "ROTATION/Rz"] }), NDM3);
  assert.ok(!rotOnly.ok && rotOnly.kind === "absent", JSON.stringify(rotOnly));
  // The group the reader refused (a partial write) refuses the deformed shape by name.
  const refused = displacementAxes(stage({ nodal: ["ROTATION/Rz"], refusedNodal: [{ component: "DISPLACEMENT", reason: "a partial write" }] }), NDM2);
  assert.ok(!refused.ok && refused.kind === "refused" && /^DISPLACEMENT not read \(a partial write\)/.test(refused.reason), JSON.stringify(refused));
});

test("ladruno: a label outside §7.1.1 refuses by name (the axis is never guessed), including a combined 6-component group", () => {
  const uq = displacementAxes(stage({ nodal: ["DISPLACEMENT/Ux", "DISPLACEMENT/Uq", "DISPLACEMENT/Uz"] }), NDM3);
  assert.ok(!uq.ok && uq.kind === "refused", JSON.stringify(uq));
  assert.match(!uq.ok ? uq.reason : "", /^DISPLACEMENT\/Uq names no axis of the fork schema's DISPLACEMENT labels \(Ux, Uy, Uz, ladruno_schema_v1\.md §7\.1\.1\)/);
  // `X`/`Y`/`Z` are not the schema's labels either (the pre-#2128 code accepted them).
  const xyz = displacementAxes(stage({ nodal: ["DISPLACEMENT/X", "DISPLACEMENT/Y", "DISPLACEMENT/Z"] }), NDM3);
  assert.ok(!xyz.ok && /DISPLACEMENT\/X, DISPLACEMENT\/Y, DISPLACEMENT\/Z name no axis/.test(xyz.reason), JSON.stringify(xyz));
  // The pre-#2128 hand-built frame3d layout: rotations inside DISPLACEMENT are refused, never dropped silently.
  const six = displacementAxes(stage({ nodal: [...D3, "DISPLACEMENT/Rx", "DISPLACEMENT/Ry", "DISPLACEMENT/Rz"] }), NDM3);
  assert.ok(!six.ok && /DISPLACEMENT\/Rx, DISPLACEMENT\/Ry, DISPLACEMENT\/Rz name no axis/.test(six.reason), JSON.stringify(six));
});

test("trap 1 (§7.1.1): a 2-D u-p node's ROTATION/Rz is pore pressure; deform never reads it and contour never calls it rotation z", () => {
  // ndm 2, ndf 3 on a u-p element: DISPLACEMENT = Ux,Uy; ROTATION/Rz holds the pore pressure.
  const up = stage({ nodal: ["DISPLACEMENT/Ux", "DISPLACEMENT/Uy", "ROTATION/Rz"] });
  const d = displacementAxes(up, NDM2);
  assert.ok(d.ok);
  assert.ok(!d.datasets.includes("ROTATION/Rz"), "the pore pressure is never a displacement axis");
  // Contour: the lone Rz is its own component (labelled Rz), never the z of a rotation vector.
  const rot = nodalFields(up.nodal).find((f) => f.name === "ROTATION")!;
  assert.deepEqual(rot.components.map((c) => `${c.label}=${c.key}`), ["Rz=ROTATION/Rz"]);
  // A mode stage's rotational group follows the same rule.
  const modal = nodalFields(["MODES_OF_VIBRATION(U)/Ux", "MODES_OF_VIBRATION(U)/Uy", "MODES_OF_VIBRATION(R)/Rz"]);
  assert.deepEqual(modal.find((f) => f.name === "MODES_OF_VIBRATION(R)")!.components.map((c) => `${c.label}=${c.key}`), ["Rz=MODES_OF_VIBRATION(R)/Rz"]);
  // ...and the mode's translation offers its magnitude, as a native mode stage's displacement does.
  assert.deepEqual(modal.find((f) => f.name === "MODES_OF_VIBRATION(U)")!.components.map((c) => c.label), ["magnitude", "x", "y"]);
  // At ndm 3 the three rotation labels are a rotation vector (§7.1.1 copies DOF 3..5 there).
  const r3 = nodalFields(["ROTATION/Rx", "ROTATION/Ry", "ROTATION/Rz"]).find((f) => f.name === "ROTATION")!;
  assert.deepEqual(r3.components.map((c) => c.label), ["x", "y", "z"]);
});

test("trap 2 (§7.1.1): a 1-D model's ROTATION is three zero columns; it never stands in for the missing axes", () => {
  const oneD = stage({ nodal: ["DISPLACEMENT/Ux", "ROTATION/Rx", "ROTATION/Ry", "ROTATION/Rz"] });
  for (const ndm of [NDM2, NDM3, NONE]) {
    const r = displacementAxes(oneD, ndm);
    assert.ok(!r.ok && r.kind === "refused", JSON.stringify(r));
    // The refusal names the translation the file lacks, never a rotation column.
    assert.match(!r.ok ? r.reason : "", /records no DISPLACEMENT\/Uy: the deformed shape needs every translational axis$/);
    assert.doesNotMatch(!r.ok ? r.reason : "", /ROTATION/);
  }
});

test("ladruno mode stage: MODES_OF_VIBRATION(U) is the translation, (R) is never read, and the legend names the eigenvalue and period", () => {
  const lam = 381.96601125;
  const mode = { index: 1, eigenvalue: lam, frequencyHz: Math.sqrt(lam) / (2 * Math.PI), periodS: (2 * Math.PI) / Math.sqrt(lam) };
  const st = stage({ id: "stage_0_mode_1", name: "mode_1", kind: "mode", time: [0], nodal: ["MODES_OF_VIBRATION(U)/Ux", "MODES_OF_VIBRATION(U)/Uy", "MODES_OF_VIBRATION(R)/Rz"], mode });
  const r = displacementAxes(st, NDM2);
  assert.ok(r.ok);
  assert.deepEqual(r.datasets, ["MODES_OF_VIBRATION(U)/Ux", "MODES_OF_VIBRATION(U)/Uy", null]);
  assert.equal(r.note, `${modeNote(mode)}; 2-D model (ndm 2, test): z is not a degree of freedom, so it stays at the node's z`);
  assert.equal(modeNote(mode), "mode 1: eigenvalue λ = 381.97, f = 3.1105 Hz, T = 0.32149 s; a mode shape has no physical amplitude, so the scale is relative");
  assert.match(modeNote({ index: 3, eigenvalue: -2, frequencyHz: 0, periodS: 0 }), /^mode 3: eigenvalue λ = -2 \(not positive: no frequency or period\)/);
  // A mode stage reads its own group only: a DISPLACEMENT component on it is not its translation.
  const wrong = displacementAxes(stage({ kind: "mode", nodal: D3 }), NDM3);
  assert.ok(!wrong.ok, JSON.stringify(wrong));
  // A refused modal group refuses the mode shape by name.
  const refusedU = displacementAxes(stage({ kind: "mode", nodal: ["MODES_OF_VIBRATION(R)/Rz"], refusedNodal: [{ component: "MODES_OF_VIBRATION(U)", reason: "a partial write" }] }), NDM2);
  assert.ok(!refusedU.ok && /^MODES_OF_VIBRATION\(U\) not read/.test(refusedU.reason), JSON.stringify(refusedU));
  // A native mode stage is unchanged: displacement_* and no note of its own.
  assert.deepEqual(displacementAxes(stage({ kind: "mode", nodal: ["displacement_x", "displacement_y", "displacement_z"] }), NDM3), { ok: true, datasets: ["displacement_x", "displacement_y", "displacement_z"], note: null });
});

/** One dataset of a `.ladruno`, read with h5wasm directly (not through the app's reader). */
async function ladrunoArray(path: string, ds: string): Promise<{ value: Float64Array; shape: number[] }> {
  await h5wasm.ready;
  const f = new h5wasm.File(path, "r");
  try {
    const d = f.get(ds) as h5wasm.Dataset;
    return { value: Float64Array.from(d.value as ArrayLike<number>), shape: [...(d.shape ?? [])] };
  } finally {
    f.close();
  }
}

test("oracle (.ladruno, paired frame3d): every deformed vertex is MODEL/NODES/COORDINATES + s × DISPLACEMENT/DATA of its node", async () => {
  const h = await opened(FRAME3D_MODEL, FRAME3D);
  const stageIdx = 1; // "lateral" = MODEL_STAGE[2]
  const k = 2;
  const scale = 10;
  const root = "MODEL_STAGE[2]";
  const coords = await ladrunoArray(FRAME3D, `${root}/MODEL/NODES/COORDINATES`);
  const ids = await ladrunoArray(FRAME3D, `${root}/RESULTS/ON_NODES/DISPLACEMENT/ID`);
  const data = await ladrunoArray(FRAME3D, `${root}/RESULTS/ON_NODES/DISPLACEMENT/DATA`);
  const [, n, c] = data.shape as [number, number, number];
  assert.equal(c, 3, "§7.1.1: DISPLACEMENT has three columns at ndm 3");
  assert.equal(coords.shape[0], n);
  // The hand-built fixture's NODES has no ID: its rows are the DISPLACEMENT/ID order, which the model.h5 confirms.
  const model = await coordsOf(FRAME3D_MODEL);
  const want = new Map<number, number[]>();
  let moved = 0;
  for (let j = 0; j < n; j++) {
    const id = ids.value[j]!;
    const x = [0, 1, 2].map((a) => coords.value[3 * j + a]!);
    assert.deepEqual(model.get(id), x, `node ${id}: the model.h5 and the .ladruno agree on its coordinates`);
    const w = x.map((xa, a) => Math.fround(Math.fround(xa) + scale * data.value[(k * n + j) * 3 + a]!));
    if (w[0] !== Math.fround(x[0]!)) moved++;
    want.set(id, w);
  }
  assert.ok(moved > 0, "the lateral stage sways: the oracle is not the undeformed mesh");
  const { s, v, full } = await showDeformed(h, stageIdx, k, scale);
  assert.equal(v.state, "ready");
  assert.deepEqual(v.state === "ready" ? v.datasets : null, D3, "deform reads DISPLACEMENT, never ROTATION");
  for (const [pos, vn] of [[full.tri, s.mesh!.vertexNodes.tri], [full.line, s.mesh!.vertexNodes.line]] as const) {
    const nodes = h.blobs.get(vn) as Float64Array;
    assert.ok(nodes.length > 0, "slabs (faces) and beams (lines) both have vertices");
    nodes.forEach((id, i) => {
      const w = want.get(id)!;
      for (let a = 0; a < 3; a++) assert.ok(near(pos[3 * i + a]!, w[a]!), `vertex ${i} node ${id} axis ${a}: ${pos[3 * i + a]} vs ${w[a]}`);
    });
  }
  h.effects.dispose();
});

test("mutation (.ladruno): DISPLACEMENT relabelled Ux,Uq,Uz refuses the deformed shape by the label's name", async () => {
  await h5wasm.ready;
  const p = join(mkdtempSync(join(tmpdir(), "v4lb2-")), "ladruno_frame3d.ladruno");
  copyFileSync(FRAME3D, p);
  const f = new h5wasm.File(p, "a");
  try {
    for (const g of ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]) {
      const grp = f.get(`${g}/RESULTS/ON_NODES/DISPLACEMENT`) as h5wasm.Group;
      grp.delete_attribute("COMPONENTS");
      grp.create_attribute("COMPONENTS", "Ux,Uq,Uz");
    }
  } finally {
    f.close();
  }
  const h = await opened(FRAME3D_MODEL, p);
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 1, step: 0 } });
  h.store.dispatch({ type: "setDeform", on: true });
  await until(() => h.store.get().resultStep?.status === "ready", "the step");
  const s = h.store.get();
  const v = deformView(s);
  assert.equal(v.state, "refused");
  assert.match(v.state === "refused" ? v.reason : "", /^DISPLACEMENT\/Uq names no axis of the fork schema's DISPLACEMENT labels/);
  assert.ok(noticesOf(s).some((x) => x.startsWith("Deformed shape not drawn: DISPLACEMENT/Uq names no axis")), noticesOf(s).join("\n"));
  assert.equal(h.effects.currentDisplacement(), null, "nothing is computed from Ux and Uz");
  h.effects.dispose();
});

test("mode shape (fork-written ladruno_modes): the eigenvector rows deform the nodes; closed form Ux3/Ux2 = 2 - λ/k; the legend names λ", async () => {
  const { host } = harness();
  const a = await host.open(MODES);
  assert.ok(a.ok, a.ok ? "" : a.error);
  const k = 1000;
  for (const [m, sign] of [[1, -1], [2, 1]] as const) {
    const meta = a.meta.stages.find((x) => x.id === `stage_0_mode_${m}`)!;
    const lambda = (k * (3 + sign * Math.sqrt(5))) / 2;
    assert.ok(Math.abs(meta.mode!.eigenvalue - lambda) < 1e-6 * lambda, `mode ${m}: λ ${meta.mode!.eigenvalue} vs ${lambda}`);
    const axes = displacementAxes({ ...meta }, NDM2);
    assert.ok(axes.ok, JSON.stringify(axes));
    assert.deepEqual(axes.datasets, ["MODES_OF_VIBRATION(U)/Ux", "MODES_OF_VIBRATION(U)/Uy", null]);
    assert.ok(axes.note!.startsWith(`mode ${m}: eigenvalue λ = ${Number(lambda.toPrecision(5))}, f = `), axes.note!);
    const r = await host.readStep({ path: MODES, stage: meta.id, step: 0, seq: m });
    assert.ok(r.ok);
    const ids: Float64Array = a.nodeIds[meta.id]!;
    const d = vertexDisplacements(ids, columnOf(ids), [r.nodal["MODES_OF_VIBRATION(U)/Ux"]!, r.nodal["MODES_OF_VIBRATION(U)/Uy"]!, null]);
    assert.equal(d.count, 0);
    const ux = (node: number): number => d.d[3 * Array.from(ids).indexOf(node)]!;
    assert.equal(ux(1), 0, "node 1 is fixed");
    assert.ok(Math.abs(ux(3) / ux(2) - (2 - lambda / k)) < 1e-9, `mode ${m}: Ux3/Ux2 = ${ux(3) / ux(2)}, closed form ${2 - lambda / k}`);
  }
  host.close();
});

test("fork-written 3-D portal (ladruno_portal3d): DISPLACEMENT and ROTATION are separate groups; deform reads DISPLACEMENT, equal to the fork's nodeDisp", async () => {
  const { host } = harness();
  const a = await host.open(PORTAL3D);
  assert.ok(a.ok, a.ok ? "" : a.error);
  const meta = a.meta.stages[0]!;
  assert.deepEqual(meta.nodal, [...D3, "ROTATION/Rx", "ROTATION/Ry", "ROTATION/Rz"], "§7.1.1 at ndm 3, as the fork writes it");
  const axes = displacementAxes({ ...meta }, NDM3);
  assert.ok(axes.ok);
  assert.deepEqual(axes.datasets, D3);
  const r = await host.readStep({ path: PORTAL3D, stage: meta.id, step: meta.time.length - 1, seq: 1 });
  assert.ok(r.ok);
  // Cross-engine oracle: ops.nodeDisp(2) and (3) printed by the same fork run that wrote the file
  // (make_ladruno_fixtures.py `_PORTAL3D_RUN`, after the second LoadControl step).
  const ids = Array.from(a.nodeIds[meta.id]!);
  const nodeDisp: Record<number, number[]> = {
    2: [0.0008771164181733662, 0.0002456163480553718, -2.9071217963256807e-6],
    3: [0.0008671566596220584, 0.000654383651944626, -1.2092878203674294e-5],
  };
  for (const [node, u] of Object.entries(nodeDisp)) {
    D3.forEach((c, i) => {
      const got = r.nodal[c]![ids.indexOf(Number(node))]!;
      assert.ok(Math.abs(got - u[i]!) <= 1e-15, `node ${node} ${c}: ${got} vs nodeDisp ${u[i]}`);
    });
  }
  // Its mode stages read MODES_OF_VIBRATION(U) and carry the file's LAMBDA.
  const modes = a.meta.stages.filter((x) => x.kind === "mode");
  assert.deepEqual(modes.map((x) => x.mode!.index), [1, 2]);
  for (const mst of modes) {
    const ax = displacementAxes({ ...mst }, NDM3);
    assert.ok(ax.ok && ax.datasets.every((ds) => ds!.startsWith("MODES_OF_VIBRATION(U)/")), JSON.stringify(ax));
  }
  host.close();
});
