// V4c2b (#1875) element contours carried to the nodes: gauss_cell_averaged,
// gauss_node and gauss_node_discrete, through the app's own renderer-side path
// (harness.ts), against PYTHON as the oracle. The fixtures and
// `fixtures/extrap.oracle.json` are written by
// `scripts/make_viewer_results_fixtures.py --extrapolation`: apeGmsh's own
// `extrapolate_gauss_slab_to_nodes` (gauss_node),
// `extrapolate_gauss_slab_per_element` (gauss_node_discrete), and the averaged
// path on each element's mean as its one point (cell_averaged), on the slab
// read back from each written file. The app's values must equal them to 1e-9
// relative. The element of a drawn primitive is read from its decl path.
//
// Mutations each test is meant to catch (the PR body lists the runs):
//   X1 the extrapolation matrix transposed   -> "gauss_node ... equal Python" (hex G8, quad G4)
//   X2 the averaging divisor dropped         -> "gauss_node ... equal Python", "cell_averaged ..."
//   X3 an unsupported type drawn as a guess  -> "an element type with no shape functions reads unsupported ..."

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { elementColours, toLinear } from "../../src/renderer/contour.ts";
import { elementMaskOf, visibleMaps } from "../../src/renderer/visible.ts";
import { joinElements, NO_DATA_RGB, rampT, sampleCmap } from "../../src/state/contour.ts";
import {
  applyPlan,
  buildPlan,
  checkPlan,
  cornerVertexValues,
  extrapolationMatrix,
  extrapolationVerdict,
  pinv,
  planFor,
  shapeOf,
  type ExtrapPlan,
  type PlanValues,
} from "../../src/state/extrapolation.ts";
import { contourView } from "../../src/state/selectors.ts";
import type { GaussMode, State } from "../../src/state/types.ts";
import { opened, until } from "./harness.ts";

const fx = (name: string) => fileURLToPath(new URL(`../../fixtures/${name}`, import.meta.url));

interface Averaged {
  node_ids: number[];
  values: number[][];
}
interface Discrete {
  element_ids: number[];
  corner_node_ids: number[][];
  values: number[][][];
}
type Oracle = Record<string, Record<string, { gauss_node: Averaged; gauss_node_discrete: Discrete; cell_averaged: Averaged }>>;
const ORACLE = JSON.parse(readFileSync(fx("extrap.oracle.json"), "utf8")) as Oracle;
const STEPS = 3;

const rel = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));

type ElementView = Extract<ReturnType<typeof contourView>, { state: "element" }>;
type H = Awaited<ReturnType<typeof opened>>;

/** Colour by element field `component` in `mode` at step `k`, range `this step`; wait until it is measured. */
async function pick(h: H, component: string, mode: GaussMode, k: number): Promise<ElementView> {
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  if (k) h.store.dispatch({ type: "setResultStep", step: k });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourRange", range: "step" });
  h.store.dispatch({ type: "setContourField", field: "stress", component, source: "element" });
  h.store.dispatch({ type: "setContourGauss", mode });
  await until(() => {
    const v = contourView(h.store.get());
    const st = h.effects.currentStep();
    return v.state === "element" && v.stepMeasured && st?.step === k && st.element === component;
  }, `${component} ${mode} at step ${k}`);
  return contourView(h.store.get()) as ElementView;
}

/** The app's plan and values for the view on screen (the effects' and the viewport's own functions). */
function appValues(h: H, v: ElementView): { plan: ExtrapPlan; r: PlanValues } {
  const s = h.store.get();
  const plan = planFor(v as Parameters<typeof planFor>[0], s, h.blobs);
  const step = h.effects.currentStep()!;
  return { plan, r: applyPlan(plan, v.groups.map((g) => step.gauss[g.id]!)) };
}

/** The app's averaged nodal values, by node id. */
function appNodal(plan: ExtrapPlan, r: PlanValues): Map<number, number> {
  const out = new Map<number, number>();
  plan.nodeIds.forEach((id, i) => out.set(id, r.nodal![i]!));
  return out;
}

/** The app's per-element corner values: element id -> [corner node ids, values]. */
function appCorners(h: H, v: ElementView, plan: ExtrapPlan, r: PlanValues): Map<number, { nodes: number[]; values: number[] }> {
  const out = new Map<number, { nodes: number[]; values: number[] }>();
  plan.groups.forEach((g, gi) => {
    const eids = h.blobs.get(v.groups[gi]!.eids) as Float64Array;
    for (let e = 0; e < g.nElements; e++) {
      const nodes: number[] = [];
      const values: number[] = [];
      for (let c = g.start[e]!; c < g.start[e + 1]!; c++) {
        nodes.push(plan.nodeIds[plan.cornerSlot[c]!]!);
        values.push(r.corner[c]!);
      }
      out.set(eids[e]!, { nodes, values });
    }
  });
  return out;
}

function checkAveraged(oracle: Averaged, k: number, nodal: Map<number, number>, what: string): void {
  assert.deepEqual([...nodal.keys()].sort((a, b) => a - b), oracle.node_ids, `${what}: the nodes that received a value are Python's`);
  oracle.node_ids.forEach((id, j) => {
    const want = oracle.values[k]![j]!;
    assert.ok(rel(nodal.get(id)!, want), `${what}: node ${id} ${nodal.get(id)} vs Python ${want}`);
  });
}

function checkDiscrete(oracle: Discrete, k: number, corners: Map<number, { nodes: number[]; values: number[] }>, what: string): void {
  assert.deepEqual([...corners.keys()].sort((a, b) => a - b), oracle.element_ids, `${what}: the elements are Python's`);
  oracle.element_ids.forEach((id, e) => {
    const got = corners.get(id)!;
    assert.deepEqual(got.nodes, oracle.corner_node_ids[e], `${what}: element ${id} corners in Python's order`);
    oracle.values[e]![k]!.forEach((want, c) => assert.ok(rel(got.values[c]!, want), `${what}: element ${id} corner ${c} ${got.values[c]} vs Python ${want}`));
  });
}

/** Every drawn triangle vertex in the colour of `want(element, node)` (null: no data), the vertex's own element by decl path. */
function checkColours(h: H, v: ElementView, want: (femId: number, node: number) => number | null): { coloured: number; noData: number } {
  const s = h.store.get();
  const m = s.mesh!;
  const { lineMap, triMap } = visibleMaps(m.legend, s.visibility.hidden, h.blobs.i32(m.lineGroup), h.blobs.i32(m.triGroup), elementMaskOf(s, m, h.blobs));
  const cc = elementColours(s, v, m, h.blobs, h.effects.currentStep(), lineMap, triMap);
  assert.ok(cc, "colours for the step on screen");
  const triEl = h.blobs.i32(m.triElement);
  const triNodes = h.blobs.get(m.vertexNodes.tri) as Float64Array;
  let coloured = 0, noData = 0;
  for (let d = 0; d < triMap.length; d++) {
    const p = triMap[d]!;
    const femId = Number(/^mesh\/element\/(\d+)$/.exec(m.elements[triEl[p]!]!)![1]);
    const ws = [0, 1, 2].map((vert) => want(femId, triNodes[p * 3 + vert]!));
    const none = ws.some((w) => w === null);
    for (let vert = 0; vert < 3; vert++) {
      const rgb = none ? NO_DATA_RGB.map(toLinear) : sampleCmap(v.cmap, rampT(ws[vert]!, v.clim.min!, v.clim.max!)).map(toLinear);
      for (let c = 0; c < 3; c++) assert.ok(Math.abs(cc.tri[(d * 3 + vert) * 3 + c]! - rgb[c]!) < 1e-6, `triangle ${d} (element ${femId}) vertex ${vert}`);
    }
    if (none) noData++;
    else coloured++;
  }
  return { coloured, noData };
}

/** The triangles of elements no group has: the plane's left-out quad (two triangles); none elsewhere. */
const NO_DATA_TRIS: Record<string, number> = { extrap_solid: 0, extrap_plane: 2, extrap_tet: 0 };

const range = (xs: number[]) => ({ min: Math.min(...xs), max: Math.max(...xs) });

for (const [name, comps, groupsWant] of [
  ["extrap_solid", ["stress_xx", "stress_yy"], [["hex_G8", 8]]],
  ["extrap_plane", ["stress_xx", "stress_xy"], [["quads_G4", 4], ["tris_G1", 1]]],
  ["extrap_tet", ["stress_xx"], [["tets_G1", 1]]],
] as const) {
  test(`${name}: gauss_node and gauss_node_discrete equal Python's extrapolation at every step, values, step ranges and colours`, async () => {
    const h = await opened(fx(`${name}.h5`), fx(`${name}.results.h5`));
    for (const comp of comps) {
      const o = ORACLE[name]![comp]!;
      for (let k = 0; k < STEPS; k++) {
        const vn = await pick(h, comp, "node", k);
        assert.equal(vn.topology, "gauss_node");
        assert.match(vn.topologyLabel, /^gauss_node: .* extrapolated to the corners \(least squares\), averaged at the nodes across elements$/);
        assert.deepEqual(vn.groups.map((g) => [g.id, g.nGauss]), groupsWant);
        const a = appValues(h, vn);
        const nodal = appNodal(a.plan, a.r);
        checkAveraged(o.gauss_node, k, nodal, `${name} ${comp} gauss_node step ${k}`);
        // The step range is of the values drawn, which overshoot the Gauss points (never their range).
        const rn = range(o.gauss_node.values[k]!);
        assert.ok(rel(vn.clim.min!, rn.min) && rel(vn.clim.max!, rn.max), `${comp} gauss_node step range ${vn.clim.min}..${vn.clim.max} vs ${rn.min}..${rn.max}`);
        const elems = new Set(o.gauss_node_discrete.element_ids);
        const cn = checkColours(h, vn, (e, n) => (elems.has(e) ? o.gauss_node.values[k]![o.gauss_node.node_ids.indexOf(n)]! : null));
        // No data: the quad no group has (extrap_plane) is drawn whole as no data, never from its neighbours' nodes.
        assert.ok(cn.coloured > 0 && cn.noData === NO_DATA_TRIS[name], `${name} gauss_node: ${cn.noData} no-data triangles`);

        const vd = await pick(h, comp, "node_discrete", k);
        assert.equal(vd.topology, "gauss_node_discrete");
        const d = appValues(h, vd);
        assert.equal(d.r.nodal, null, "the discrete topology averages nothing");
        checkDiscrete(o.gauss_node_discrete, k, appCorners(h, vd, d.plan, d.r), `${name} ${comp} gauss_node_discrete step ${k}`);
        const rd = range(o.gauss_node_discrete.values.flatMap((e) => e[k]!));
        assert.ok(rel(vd.clim.min!, rd.min) && rel(vd.clim.max!, rd.max), `${comp} gauss_node_discrete step range`);
        const byEl = new Map(o.gauss_node_discrete.element_ids.map((id, e) => [id, e]));
        const cd = checkColours(h, vd, (e, n) => {
          const i = byEl.get(e);
          return i === undefined ? null : o.gauss_node_discrete.values[i]![k]![o.gauss_node_discrete.corner_node_ids[i]!.indexOf(n)]!;
        });
        assert.equal(cd.noData, NO_DATA_TRIS[name], `${name} gauss_node_discrete no-data triangles`);
      }
    }
    h.effects.dispose();
  });

  test(`${name}: cell_averaged equals Python's averaged path on each element's mean, values and colours`, async () => {
    const h = await opened(fx(`${name}.h5`), fx(`${name}.results.h5`));
    const comp = comps[0];
    const o = ORACLE[name]![comp]!;
    const k = 2;
    const v = await pick(h, comp, "cell_averaged", k);
    const oneGauss = groupsWant.every(([, g]) => g === 1);
    assert.equal(v.topology, oneGauss ? "gauss_cell_averaged" : "element_mean_averaged");
    // A field with several Gauss points never carries Qt's gauss_cell_averaged name for its element means.
    if (!oneGauss) assert.match(v.topologyLabel, /^element mean of .* Gauss points, averaged at the nodes across elements$/);
    const a = appValues(h, v);
    checkAveraged(o.cell_averaged, k, appNodal(a.plan, a.r), `${name} cell_averaged`);
    const elems = new Set(o.gauss_node_discrete.element_ids);
    const cc = checkColours(h, v, (e, n) => (elems.has(e) ? o.cell_averaged.values[k]![o.cell_averaged.node_ids.indexOf(n)]! : null));
    assert.equal(cc.noData, NO_DATA_TRIS[name], `${name} cell_averaged no-data triangles`);
    h.effects.dispose();
  });
}

test("the history scan of a node-located topology is the range of Python's values over every step (gauss_node, then gauss_node_discrete)", async () => {
  const h = await opened(fx("extrap_solid.h5"), fx("extrap_solid.results.h5"));
  const o = ORACLE["extrap_solid"]!["stress_yy"]!;
  for (const [mode, want] of [
    ["node", range(o.gauss_node.values.flat())],
    ["node_discrete", range(o.gauss_node_discrete.values.flat(2))],
  ] as const) {
    await pick(h, "stress_yy", mode, 1);
    h.store.dispatch({ type: "setContourRange", range: "history" });
    await until(() => h.store.get().contour.scan?.status === "done" && h.store.get().contour.scan!.key.includes(`@${mode}|`), `the ${mode} history scan`);
    const sc = h.store.get().contour.scan!;
    assert.match(sc.key, new RegExp(`\\|elem:stress@${mode}\\|stress_yy$`));
    assert.ok(rel(sc.min!, want.min) && rel(sc.max!, want.max), `${mode} scan ${sc.min}..${sc.max} vs Python ${want.min}..${want.max}`);
    const v = contourView(h.store.get()) as ElementView;
    assert.equal(v.clim.mode, "history");
  }
  h.effects.dispose();
});

test("an element with a non-finite Gauss value has no data, adds nothing to its nodes, and the divisor counts only the elements that gave a value", () => {
  // Two quads (one point each) sharing nodes 2 and 3: [1 2 3 4] and [2 5 6 3].
  const cells = new Map<number, { code: number; nodes: number[] }>([
    [10, { code: 3, nodes: [1, 2, 3, 4] }],
    [11, { code: 3, nodes: [2, 5, 6, 3] }],
  ]);
  const plan = buildPlan("gauss_cell_averaged", [{ id: "g", nElements: 2, nGauss: 1, eids: Float64Array.of(10, 11), naturalCoords: [[0, 0]] }], (id) => cells.get(id) ?? null);
  const both = applyPlan(plan, [Float64Array.of(1, 3)]);
  const at = (r: PlanValues, node: number) => r.nodal![plan.nodeIds.indexOf(node)]!;
  assert.equal(at(both, 2), 2);
  assert.equal(at(both, 1), 1);
  const one = applyPlan(plan, [Float64Array.of(1, NaN)]);
  assert.equal(at(one, 2), 1, "the shared node is element 10's value alone (divided by 1, not 2)");
  assert.ok(Number.isNaN(at(one, 5)), "a node only the no-data element touches has no value");
  assert.ok([4, 5, 6, 7].every((c) => Number.isNaN(one.corner[c]!)), "the no-data element's corners are NaN");
  // A vertex of the no-data element at a shared node is no data, never the neighbour's average.
  const vals = cornerVertexValues(plan, one, Int32Array.of(1, 4));
  assert.equal(vals[0], 1);
  assert.ok(Number.isNaN(vals[1]!));
});

test("an element type with no shape functions reads unsupported (never Python's element-mean fallback); natural coordinates too short block only the extrapolating topologies", async () => {
  // Pure: gmsh 7 (pyramid) and 8 (line3) have no corner shape functions here, as Python's catalog.
  for (const code of [7, 8, 99]) {
    assert.equal(shapeOf(code), null);
    const v = extrapolationVerdict([{ id: "g", nElements: 1, nGauss: 4, eids: Float64Array.of(1), naturalCoords: [[0, 0, 0]] }], () => ({ code, nodes: [1, 2, 3, 4, 5] }));
    assert.match(v.corners!, new RegExp(`gmsh type ${code}, which has no corner shape functions`));
  }
  const short = extrapolationVerdict([{ id: "g", nElements: 1, nGauss: 8, eids: Float64Array.of(1), naturalCoords: Array.from({ length: 8 }, () => [0.5, 0.5]) }], () => ({ code: 5, nodes: [1, 2, 3, 4, 5, 6, 7, 8] }));
  assert.equal(short.corners, null);
  assert.match(short.matrix!, /fewer than gmsh type 5's 3/);
  assert.throws(() => buildPlan("gauss_node", [{ id: "g", nElements: 1, nGauss: 8, eids: Float64Array.of(1), naturalCoords: [[0.5, 0.5]] }], () => ({ code: 5, nodes: [1, 2, 3, 4, 5, 6, 7, 8] })));

  // Through the app: the solid's block declared as an unknown type is refused "unsupported" in each node mode; per cell still draws.
  const h = await opened(fx("extrap_solid.h5"), fx("extrap_solid.results.h5"));
  const s0 = h.store.get();
  (s0 as { blocks: State["blocks"] }).blocks = s0.blocks.map((b) => ({ ...b, code: 7 }));
  for (const mode of ["cell_averaged", "node", "node_discrete"] as const) {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
    h.store.dispatch({ type: "setColourBy", by: "contour" });
    h.store.dispatch({ type: "setContourField", field: "stress", component: "stress_xx", source: "element" });
    h.store.dispatch({ type: "setContourGauss", mode });
    await until(() => contourView(h.store.get()).state === "refused", `${mode} refused`);
    const v = contourView(h.store.get());
    assert.ok(v.state === "refused" && v.code === "unsupported", mode);
    assert.match(v.reason, /is unsupported: element \d+ is gmsh type 7, which has no corner shape functions/);
  }
  h.store.dispatch({ type: "setContourGauss", mode: "cell" });
  assert.equal(contourView(h.store.get()).state, "element");

  // A verdict with only a matrix problem: cell_averaged draws, the extrapolating modes are unsupported.
  const s1 = h.store.get();
  const j = s1.contour.join!;
  assert.ok(j.ok);
  const withMatrix = (mode: GaussMode): State => ({ ...s1, contour: { ...s1.contour, gauss: mode, join: { ...j, extrapolation: { corners: null, matrix: "group hex_G8: bad" } } } });
  assert.equal(contourView(withMatrix("cell_averaged")).state !== "refused", true);
  for (const mode of ["node", "node_discrete"] as const) {
    const v = contourView(withMatrix(mode));
    assert.ok(v.state === "refused" && v.code === "unsupported" && /group hex_G8: bad/.test(v.reason), mode);
  }
  h.effects.dispose();
});

test("review F1: one non-finite _natural_coords value in a verified quad group reads unsupported in node_discrete and gauss_node, never a silent 0, and never widens the range", async () => {
  // Pure: a NaN coordinate (or an all-zero / non-finite result) gives no matrix.
  const g = 1 / Math.sqrt(3);
  const quad = [[-g, -g], [g, -g], [g, g], [-g, g]];
  assert.ok(extrapolationMatrix(3, quad));
  assert.equal(extrapolationMatrix(3, [[NaN, -g], ...quad.slice(1)]), null);
  assert.equal(extrapolationMatrix(3, [[Infinity, -g], ...quad.slice(1)]), null);

  // Through the app (the reviewer's repro): extrap_plane's quads_G4 with _natural_coords[0][0] = NaN.
  const h = await opened(fx("extrap_plane.h5"), fx("extrap_plane.results.h5"));
  const grp = h.store.get().results!.stages[0]!.gauss.find((x) => x.id === "quads_G4")!;
  grp.naturalCoords![0]![0] = NaN;
  for (const mode of ["node_discrete", "node"] as const) {
    h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
    h.store.dispatch({ type: "setResultStep", step: 2 });
    h.store.dispatch({ type: "setColourBy", by: "contour" });
    h.store.dispatch({ type: "setContourRange", range: "step" });
    h.store.dispatch({ type: "setContourField", field: "stress", component: "stress_xx", source: "element" });
    h.store.dispatch({ type: "setContourGauss", mode });
    await until(() => contourView(h.store.get()).state === "refused", `${mode} refused`);
    const v = contourView(h.store.get());
    assert.ok(v.state === "refused" && v.code === "unsupported", mode);
    assert.match(v.reason, /group quads_G4: its _natural_coords hold a non-finite value/);
    // No range is measured for this mode's step (a 0 corner would have widened it).
    await new Promise((r) => setTimeout(r, 50));
    const sr = h.store.get().contour.stepRange;
    assert.ok(sr === null || !sr.key.includes(`@${mode}|`), `${mode}: no step range measured (${JSON.stringify(sr)})`);
  }
  // The one-point modes need no matrix: cell_averaged still draws, from finite values only.
  h.store.dispatch({ type: "setContourGauss", mode: "cell_averaged" });
  await until(() => contourView(h.store.get()).state === "element", "cell_averaged draws");
  h.effects.dispose();
});

test("review F2: pinv cuts singular values at numpy's default as Python calls it (rcond 1e-15 x sigma_max)", () => {
  // numpy 2.3.5: np.linalg.pinv([[1, 0], [0, 9.5e-16]]) == [[1, 0], [0, 0]].
  assert.deepEqual(pinv([[1, 0], [0, 9.5e-16]]), [[1, 0], [0, 0]]);
  // Just above the cutoff it is kept.
  assert.ok(Math.abs(pinv([[1, 0], [0, 2e-15]])[1]![1]! - 5e14) < 1);
});

test("pinv: the hex8 2x2x2 matrix inverts the shape functions (A M = I), and a rank-deficient matrix gets numpy's minimum-norm answer", () => {
  const g = 1 / Math.sqrt(3);
  const nat = [-1, 1].flatMap((c) => [-1, 1].flatMap((b) => [-1, 1].map((a) => [a * g, b * g, c * g])));
  const M = extrapolationMatrix(5, nat)!;
  const A = nat.map((p) => shapeOf(5)!.N(p));
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      let s = 0;
      for (let k = 0; k < 8; k++) s += A[i]![k]! * M[k * 8 + j]!;
      assert.ok(Math.abs(s - (i === j ? 1 : 0)) < 1e-12, `(A M)[${i}][${j}] = ${s}`);
    }
  }
  // [[1, 1], [1, 1]]: pinv = [[0.25, 0.25], [0.25, 0.25]]; a 1 x 3 row: its transpose over its squared norm.
  assert.deepEqual(pinv([[1, 1], [1, 1]]).flat().map((x) => Math.round(x * 1e12) / 1e12), [0.25, 0.25, 0.25, 0.25]);
  pinv([[1, 2, 2]]).forEach((r, i) => assert.ok(Math.abs(r[0]! - [1, 2, 2][i]! / 9) < 1e-15));
});

test("checkPlan accepts a plan the app builds and refuses a malformed one; main's scan refuses a plan whose groups are not the field's", async () => {
  const h = await opened(fx("extrap_plane.h5"), fx("extrap_plane.results.h5"));
  const v = await pick(h, "stress_xx", "node", 0);
  const plan = planFor(v as Parameters<typeof planFor>[0], h.store.get(), h.blobs);
  assert.ok(checkPlan(plan));
  assert.ok(checkPlan(structuredClone(plan)), "the plan survives the IPC's structured clone");
  assert.equal(checkPlan({ ...plan, cornerSlot: Int32Array.of(-1) }), false);
  assert.equal(checkPlan({ ...plan, topology: "gauss_cell" }), false);
  assert.equal(checkPlan({ ...plan, groups: [{ ...plan.groups[0]!, mats: [new Float64Array(3)] }, plan.groups[1]!] }), false);
  const s = h.store.get();
  const st = s.results!.stages[0]!;
  const bad = await h.host.scan({ path: s.results!.path, stage: st.id, components: ["stress_xx"], magnitude: false, scanId: 990, source: "element", plan: { ...plan, groups: [plan.groups[1]!, plan.groups[0]!] } }, () => undefined);
  assert.ok(!bad.ok && /are not stage .* groups with stress_xx/.test(bad.error), JSON.stringify(bad));
  h.effects.dispose();
});

test("the join of a verified field carries its extrapolation verdict (the effects check the cells, never the selectors guess)", async () => {
  const h = await opened(fx("extrap_plane.h5"), fx("extrap_plane.results.h5"));
  await pick(h, "stress_xx", "node", 0);
  const j = h.store.get().contour.join!;
  assert.ok(j.ok);
  assert.deepEqual(j.extrapolation, { corners: null, matrix: null });
  // joinElements itself is unchanged: the verdict is beside it.
  const s = h.store.get();
  const st = s.results!.stages[0]!;
  const jj = joinElements(s.decls, st.gauss.map((g) => ({ id: g.id, eids: h.blobs.get(g.elementIds!) as Float64Array })));
  assert.ok(jj.ok && jj.rows === j.rows);
  h.effects.dispose();
});
