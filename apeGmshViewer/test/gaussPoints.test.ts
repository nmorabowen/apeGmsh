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
  assert.deepEqual(store.get().gaussPoints, { on: true, size: 4, auto: true });
  assert.deepEqual(store.get().nodes, { on: false, size: 12, auto: false });
  store.dispatch({ type: "setGaussPoints", on: false });
  store.dispatch({ type: "setGaussPoints", on: true });
  assert.equal(store.get().gaussPoints.size, null, "automatic: chosen again");
});
