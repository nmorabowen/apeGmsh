// V4c1 (#1756) oracles on the committed contour fixtures, through the app's
// own renderer-side path (test/contour/harness.ts): the per-vertex values and
// colours equal the generator's closed form at a chosen step, the history
// range equals the min and max of the closed form over every node and step,
// and a node the field does not record is "no data", never a value.

import assert from "node:assert/strict";
import { test } from "node:test";
import { openModel } from "../../src/reader/node.ts";
import { contourColours, toLinear } from "../../src/renderer/contour.ts";
import { columnOf, MAGNITUDE, NO_DATA_RGB, rampT, sampleCmap, vertexValues } from "../../src/state/contour.ts";
import { contourFieldsOf, contourView } from "../../src/state/selectors.ts";
import { closedForm, NODATA, NODATA_RESULTS, opened, STAGED, STAGES, until } from "./harness.ts";

/** Node id -> [x, y, z], read from the model file (not from the app's derivations). */
async function coordsOf(path: string): Promise<Map<number, [number, number, number]>> {
  const m = await openModel(path);
  const out = new Map<number, [number, number, number]>();
  m.nodeIds.forEach((id, i) => out.set(id, [m.nodeCoords[3 * i]!, m.nodeCoords[3 * i + 1]!, m.nodeCoords[3 * i + 2]!]));
  return out;
}

/** The expected value of a component key at a node, from the closed form. */
function expected(key: string, xyz: readonly [number, number, number], stage: number, k: number): number {
  const [ux, uy, uz] = closedForm(xyz[0], xyz[2], stage, k, STAGES[stage]!.T);
  if (key === MAGNITUDE) return Math.hypot(ux, uy, uz);
  return { displacement_x: ux, displacement_y: uy, displacement_z: uz }[key]!;
}

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));

test("every drawn vertex is tied to exactly one node, at that node's coordinates", async () => {
  const h = await opened();
  const s = h.store.get();
  const vn = s.mesh!.vertexNodes!;
  assert.equal(vn.unresolved, 0);
  const coords = await coordsOf(STAGED);
  for (const [pos, nodes] of [
    [h.blobs.f32(s.mesh!.triPositions), h.blobs.get(vn.tri)],
    [h.blobs.f32(s.mesh!.linePositions), h.blobs.get(vn.line)],
  ] as const) {
    assert.equal(nodes.length * 3, pos.length);
    nodes.forEach((id, v) => {
      const c = coords.get(id)!;
      assert.ok(c, `vertex ${v}: node ${id} is a model node`);
      assert.deepEqual([pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]], c.map(Math.fround), `vertex ${v} sits on node ${id}`);
    });
  }
  h.effects.dispose();
});

test("per-vertex values and colours at a chosen step equal the closed form: the magnitude and each axis component", async () => {
  const h = await opened();
  const coords = await coordsOf(STAGED);
  const stage = 2, k = 2;
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
  h.store.dispatch({ type: "setResultStep", step: k });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourRange", range: "step" });
  const field = contourFieldsOf(h.store.get())!.fields;
  assert.deepEqual(field.map((f) => [f.name, f.components.map((c) => c.label)]), [["displacement", ["magnitude", "x", "y", "z"]]]);
  for (const comp of field[0]!.components) {
    h.store.dispatch({ type: "setContourField", field: "displacement", component: comp.key });
    await until(() => {
      const v = contourView(h.store.get());
      return v.state === "ready" && v.stepMeasured && h.effects.currentStep()?.step === k;
    }, `the step range of ${comp.label}`);
    const s = h.store.get();
    const v = contourView(s);
    assert.ok(v.state === "ready");
    const st = s.results!.stages[stage]!;
    const step = h.effects.currentStep()!;
    const triNodes = h.blobs.get(s.mesh!.vertexNodes!.tri) as Float64Array;
    const values = vertexValues(triNodes, columnOf(h.blobs.get(st.nodeIds!) as Float64Array), v.component.datasets.map((d) => step.nodal[d]!), v.component.magnitude);
    triNodes.forEach((id, i) => assert.ok(close(values[i]!, expected(comp.key, coords.get(id)!, stage, k)), `${comp.label} at vertex ${i} (node ${id})`));
    // The step range: the closed form's min and max over every recorded node at this step.
    const all = [...coords.values()].map((c) => expected(comp.key, c, stage, k));
    assert.ok(close(v.clim.min!, Math.min(...all)) && close(v.clim.max!, Math.max(...all)), `${comp.label} step range`);
    assert.equal(v.clim.label, "range: this step");
    // The colours the viewport uploads: jet at the closed-form value, in linear space.
    const n = triNodes.length / 3;
    const triMap = Int32Array.from({ length: n }, (_, i) => i);
    const cc = contourColours(s, v, s.mesh!, h.blobs, step, new Int32Array(0), triMap)!;
    for (const i of [0, Math.floor(n / 2), n - 1]) {
      const want = sampleCmap("jet", rampT(expected(comp.key, coords.get(triNodes[3 * i]!)!, stage, k), v.clim.min!, v.clim.max!)).map(toLinear);
      for (let c = 0; c < 3; c++) assert.ok(Math.abs(cc.tri[9 * i + c]! - want[c]!) < 1e-6, `${comp.label} colour of triangle ${i}`);
    }
  }
  h.effects.dispose();
});

test("the history range equals the closed form's min and max over every node and step (magnitude and a component)", async () => {
  const h = await opened();
  const coords = [...(await coordsOf(STAGED)).values()];
  const stage = 2;
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  for (const key of [MAGNITUDE, "displacement_z"]) {
    h.store.dispatch({ type: "setContourField", field: "displacement", component: key });
    await until(() => h.store.get().contour.scan?.status === "done" && contourView(h.store.get()).state === "ready" && (contourView(h.store.get()) as { keys: { scan: string } }).keys.scan === h.store.get().contour.scan!.key, `the scan of ${key}`);
    const all: number[] = [];
    for (let k = 0; k < STAGES[stage]!.T; k++) for (const c of coords) all.push(expected(key, c, stage, k));
    const sc = h.store.get().contour.scan!;
    assert.ok(close(sc.min!, Math.min(...all)) && close(sc.max!, Math.max(...all)), `${key}: scan ${sc.min}..${sc.max}`);
    const v = contourView(h.store.get());
    assert.ok(v.state === "ready");
    assert.deepEqual([v.clim.min, v.clim.max, v.clim.mode, v.clim.label], [sc.min, sc.max, "history", "range: whole history (4 steps)"]);
  }
  h.effects.dispose();
});

test("no data: a node the field does not record is NaN, never 0, and every primitive touching it is drawn whole in the no-data colour", async () => {
  const h = await opened(NODATA, NODATA_RESULTS);
  const coords = await coordsOf(NODATA);
  const stage = 2, k = 3;
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
  h.store.dispatch({ type: "setResultStep", step: k });
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  await until(() => {
    const v = contourView(h.store.get());
    return v.state === "ready" && v.stepMeasured && h.effects.currentStep()?.step === k;
  }, "the step range");
  const s = h.store.get();
  const v = contourView(s);
  assert.ok(v.state === "ready");
  const st = s.results!.stages[stage]!;
  const ids = h.blobs.get(st.nodeIds!) as Float64Array;
  assert.equal(ids.length, [...coords.values()].filter((c) => c[2] <= 3 + 1e-9).length, "the file records the nodes with z <= 3 only");
  const triNodes = h.blobs.get(s.mesh!.vertexNodes!.tri) as Float64Array;
  const step = h.effects.currentStep()!;
  const values = vertexValues(triNodes, columnOf(ids), v.component.datasets.map((d) => step.nodal[d]!), true);
  let absent = 0;
  triNodes.forEach((id, i) => {
    if (coords.get(id)![2] > 3 + 1e-9) {
      absent++;
      assert.ok(Number.isNaN(values[i]!), `vertex ${i} (node ${id}, not recorded) is no data, not ${values[i]}`);
    } else assert.ok(close(values[i]!, expected(MAGNITUDE, coords.get(id)!, stage, k)));
  });
  assert.ok(absent > 0, "the drawn mesh has vertices on unrecorded nodes");
  const n = triNodes.length / 3;
  const cc = contourColours(s, v, s.mesh!, h.blobs, step, new Int32Array(0), Int32Array.from({ length: n }, (_, i) => i))!;
  const noData = NO_DATA_RGB.map(toLinear);
  let grey = 0;
  for (let t = 0; t < n; t++) {
    const touches = [0, 1, 2].some((j) => Number.isNaN(values[3 * t + j]!));
    for (let j = 0; j < 3; j++) {
      const rgb = [0, 1, 2].map((c) => cc.tri[9 * t + 3 * j + c]!);
      const isGrey = rgb.every((x, c) => Math.abs(x - noData[c]!) < 1e-7);
      assert.equal(isGrey, touches, `triangle ${t} vertex ${j}: no data ${touches}, drawn grey ${isGrey}`);
    }
    if (touches) grey++;
  }
  assert.ok(grey > 0);
  h.effects.dispose();
});
