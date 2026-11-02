// Gauss point placement: the corner weights at each point's natural
// coordinates, the positions and displacements they give, and the refusals.

import assert from "node:assert/strict";
import { test } from "node:test";
import { columnOf } from "../src/state/contour.ts";
import { gaussBasePositions, gaussDisplacements, GaussLayoutBuilder } from "../src/state/gaussPoints.ts";
import { Store } from "../src/state/store.ts";

// A 2 x 1 quad (gmsh 3) on nodes 1-4, a beam (gmsh 1) on nodes 4-5, and a 10-node tet (gmsh 11, placed on its corners).
const ids = Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
const xyz = Float64Array.from([0, 0, 0, 2, 0, 0, 2, 1, 0, 0, 1, 0, 0, 1, 4, 0, 0, 1, 1, 0, 1, 0, 1, 1]);
const cells: Record<string, { code: number; nodes: number[] }> = {
  "mesh/element/10": { code: 3, nodes: [1, 2, 3, 4] },
  "mesh/element/20": { code: 1, nodes: [4, 5] },
  "mesh/element/30": { code: 11, nodes: [1, 6, 7, 8, 0, 0, 0, 0, 0, 0] },
  "mesh/element/40": { code: 99, nodes: [1] },
};
const builder = (hidden: string[] = []) =>
  new GaussLayoutBuilder(
    (p) => cells[p] ?? null,
    (p) => !hidden.includes(p),
  );

test("a quad's 2x2 points sit at the shape-function weighted corners", () => {
  const b = builder();
  const g = 1 / Math.sqrt(3);
  b.add("mesh/element/10", [[-g, -g], [g, -g], [g, g], [-g, g]]);
  const layout = b.build();
  assert.equal(layout.count, 4);
  const p = gaussBasePositions(layout, ids, xyz);
  const close = (a: number, x: number) => assert.ok(Math.abs(a - x) < 1e-6, `${a} vs ${x}`);
  close(p[0]!, 1 - g);
  close(p[1]!, 0.5 - g / 2);
  close(p[6]!, 1 + g);
  close(p[7]!, 0.5 + g / 2);
  assert.equal(p[2], 0);
});

test("a beam station at xi along its two nodes; a quadratic tet is placed on its linear corners", () => {
  const b = builder();
  b.add("mesh/element/20", [[-1], [0], [1]]);
  b.add("mesh/element/30", [[0.25, 0.25, 0.25]]);
  const p = gaussBasePositions(b.build(), ids, xyz);
  assert.deepEqual([...p.slice(0, 9)], [0, 1, 0, 0, 1, 2, 0, 1, 4]);
  assert.deepEqual([...p.slice(9)], [0.25, 0.25, 0.75], "the centroid of nodes 1, 6, 7, 8");
});

test("hidden elements are left out; unplaceable ones are counted by reason, never guessed", () => {
  const b = builder(["mesh/element/20"]);
  b.add("mesh/element/20", [[0]]);
  b.add("mesh/element/40", [[0]]);
  b.add("mesh/element/77", [[0]]);
  b.add("mesh/element/10", [[0]]);
  b.refuse("no Gauss point coordinates in the file", 5);
  const layout = b.build();
  assert.equal(layout.count, 0);
  assert.deepEqual(
    layout.refused.map((r) => [r.reason, r.elements]),
    [
      ["no shape functions for gmsh type 99", 1],
      ["not an element of the model's mesh", 1],
      ["natural coordinates with fewer than 2 components", 1],
      ["no Gauss point coordinates in the file", 5],
    ],
  );
});

test("a point moves by the same weights over its corners' displacements, and stays put when a corner has none", () => {
  const b = builder();
  b.add("mesh/element/20", [[0]]);
  b.add("mesh/element/10", [[0, 0]]);
  const layout = b.build();
  const column = columnOf(ids);
  const ux = Float64Array.from([0, 0, 0, 2, 4, 0, 0, 0]);
  const zero = new Float64Array(8);
  assert.deepEqual([...gaussDisplacements(layout, column, [ux, zero, null])], [3, 0, 0, 0.5, 0, 0]);
  const withNaN = Float64Array.from([NaN, 0, 0, 2, 4, 0, 0, 0]);
  assert.deepEqual([...gaussDisplacements(layout, column, [withNaN, zero, null])], [3, 0, 0, 0, 0, 0], "node 1 has none: the quad's point stays");
});

test("setGaussPoints and setGaussSize follow the node markers' size rules, separately from them", () => {
  const store = new Store();
  store.dispatch({ type: "setGaussPoints", on: true });
  store.dispatch({ type: "setGaussSize", size: 4, auto: true });
  store.dispatch({ type: "setNodeSize", size: 12 });
  assert.deepEqual(store.get().gaussPoints, { on: true, size: 4, auto: true, measured: null });
  assert.deepEqual(store.get().nodes, { on: false, size: 12, auto: false });
  store.dispatch({ type: "setGaussPoints", on: false });
  store.dispatch({ type: "setGaussPoints", on: true });
  assert.equal(store.get().gaussPoints.size, null, "automatic: chosen again");
});

test("a staged model: elements the stage does not have get no point, interior solid elements included", async () => {
  const { gaussLayoutOf } = await import("../src/state/gaussPoints.ts");
  const { activationNow } = await import("../src/state/selectors.ts");
  const { opened, STAGED } = await import("./contour/harness.ts");
  const { fileURLToPath } = await import("node:url");
  const h = await opened(STAGED, fileURLToPath(new URL("../fixtures/elem_staged.results.h5", import.meta.url)));
  try {
    const stages = h.store.get().results!.stages;
    let checked = 0;
    for (let stage = 0; stage < stages.length; stage++) {
      h.store.dispatch({ type: "setPhase", at: { kind: "results", stage, step: 0 } });
      const s = h.store.get();
      const a = activationNow(s);
      if (!a || a.activation.state !== "filtered" || !s.results!.stages[stage]!.gauss[0]?.elementIds) continue;
      const tags = a.activation.inactiveTags;
      const g = s.results!.stages[stage]!.gauss[0]!;
      const eids = h.blobs.get(g.elementIds!) as Float64Array;
      const active = [...eids].filter((id) => !(s.decls[`mesh/element/${id}`]?.element?.metas ?? []).some((m) => tags.has(m.tag))).length;
      if (active === eids.length) continue;
      const layout = gaussLayoutOf(s, h.blobs, null)!;
      assert.equal(layout.count, active * g.nGauss, `stage ${stage}: one point per Gauss point of each active element only`);
      checked++;
    }
    assert.ok(checked > 0, "the fixture has a filtered stage whose Gauss groups include inactive elements");
  } finally {
    h.effects.dispose();
  }
});

test("orientStations: xi runs from the element's node I; a cell stored J to I takes -xi; contradicting nodes refuse", async () => {
  const { orientStations } = await import("../src/state/gaussPoints.ts");
  assert.deepEqual(orientStations([4, 5], [4, 5], [-1, 0.4]), [[-1], [0.4]]);
  assert.deepEqual(orientStations([4, 5], [5, 4], [-1, 0.4]), [[1], [-0.4]]);
  assert.equal(orientStations([4, 5], [4, 6], [0]), null);
  assert.equal(orientStations([4, 5], [4, 4], [0]), null);
  assert.deepEqual(orientStations([4, 5], null, [0.5]), [[0.5]], "no recorded nodes: the cell's order");
});

test("a beam's points shift to the clear span between its joint offsets; the shift does not deform", () => {
  const b = builder();
  b.add("mesh/element/20", [[-1], [0], [1]], { dI: [0, 0, 0.5], dJ: [0, 0, -1] });
  const layout = b.build();
  const p = gaussBasePositions(layout, ids, xyz);
  assert.deepEqual([...p], [0, 1, 0.5, 0, 1, 1.75, 0, 1, 3], "node 4 at z 0 + 0.5, node 5 at z 4 - 1, mid-span between them");
  const zero = new Float64Array(8);
  assert.deepEqual([...gaussDisplacements(layout, columnOf(ids), [zero, zero, null])], new Array(9).fill(0));
});

test("dropUnplaced: a point whose corner is not in the node table is dropped and counted, never drawn at NaN", async () => {
  const { dropUnplaced } = await import("../src/state/gaussPoints.ts");
  const b = new GaussLayoutBuilder((p) => (p === "e/1" ? { code: 1, nodes: [4, 99] } : cells[p] ?? null), () => true);
  b.add("mesh/element/20", [[0]]);
  b.add("e/1", [[0]]);
  const layout = b.build();
  const out = dropUnplaced(layout, gaussBasePositions(layout, ids, xyz));
  assert.equal(out.layout.count, 1);
  assert.deepEqual([...out.base], [0, 1, 2]);
  assert.match(out.layout.notes.at(-1)!, /^1 point not placed: a corner node is not in the model's node table\.$/);
});

test("gaussMeasured: the Display row lists what the viewport could not place, by reason, for the stage shown", async () => {
  const { reduce, initialState } = await import("../src/state/store.ts");
  const { gaussNotesOf } = await import("../src/state/gaussNotes.ts");
  let s = reduce(initialState, { type: "setGaussPoints", on: true });
  s = reduce(s, { type: "gaussMeasured", stage: "st", count: 0, refused: [{ reason: "no shape functions for gmsh type 8", elements: 3 }], notes: ["Beam stations not read: boom"] });
  assert.equal(s.gaussPoints.measured?.count, 0);
  const staged = { ...s, phase: { ...s.phase, at: { kind: "results" as const, stage: 0, step: 0 } }, results: { stages: [{ id: "st", gauss: [], lineStations: [] }] } } as unknown as typeof s;
  assert.deepEqual(gaussNotesOf(staged), [
    "No Gauss point or beam station placed on this stage.",
    "3 elements not placed: no shape functions for gmsh type 8.",
    "Beam stations not read: boom",
  ]);
  assert.throws(() => reduce(s, { type: "gaussMeasured", stage: "st", count: -1, refused: [], notes: [] }), RangeError);
});
