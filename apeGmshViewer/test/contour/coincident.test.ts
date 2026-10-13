// V4c1 (#1756, amendment): coincident nodes. apeGmsh models routinely carry
// distinct nodes at one point (a decoupled interface, an RBE master, a tie,
// a zero-length spring), so the node of a drawn vertex must come from the
// connectivity the vertex was built from, never from a match by position.
//
// The fixture is a synthetic model.h5 as the reader returns it (a
// ModelFile), small enough to state by hand:
//
//   A  quad  1 2 3 4        the unit square at x 0..1
//   B  quad  5 6 7 8        x 1..2; node 5 sits on node 2, node 8 on node 3 (a decoupled interface)
//   I  quad  2 5 8 3        a zero-thickness interface element on that edge: two coincident pairs in one face
//   C  quad  9 10 11 12     a collapsed quad: node 11 sits on node 10, in the same drawn face
//   S  line  13 14          a zero-length spring: both ends at (3, 0, 0)
//
// Every node has its own value, 10 id + 0.5, so coincident nodes differ.

import assert from "node:assert/strict";
import { test } from "node:test";
import type { ModelFile } from "../../src/model/types.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { columnOf, primitiveColours, vertexValues } from "../../src/state/contour.ts";
import { loadModel } from "../../src/state/load.ts";

const XYZ: Record<number, [number, number, number]> = {
  1: [0, 0, 0], 2: [1, 0, 0], 3: [1, 1, 0], 4: [0, 1, 0],
  5: [1, 0, 0], 6: [2, 0, 0], 7: [2, 1, 0], 8: [1, 1, 0],
  9: [0, 2, 0], 10: [1, 2, 0], 11: [1, 2, 0], 12: [0, 3, 0],
  13: [3, 0, 0], 14: [3, 0, 0],
};
const QUADS = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 5, 8, 3], [9, 10, 11, 12]];
const value = (id: number) => 10 * id + 0.5;

function coincidentModel(): ModelFile {
  const ids = Object.keys(XYZ).map(Number);
  return {
    path: "coincident.h5",
    sizeBytes: 0,
    neutralVersion: "2.35.0",
    meta: {},
    nodeIds: new Float64Array(ids),
    nodeCoords: new Float64Array(ids.flatMap((i) => XYZ[i]!)),
    blocks: [
      { alias: "quad4", code: 3, dim: 2, npe: 4, ids: new Float64Array([1, 2, 3, 4]), connectivity: new Float64Array(QUADS.flat()) },
      { alias: "line2", code: 1, dim: 1, npe: 2, ids: new Float64Array([5]), connectivity: new Float64Array([13, 14]) },
    ],
    physicalGroups: [],
    labels: [],
    opensees: null,
    sequence: { status: "absent" },
    warnings: [],
    readMs: 0,
  };
}

test("coincident nodes: each drawn vertex names the node of its own connectivity slot, and gets that node's value, with no grey", () => {
  const blobs = new BlobStore();
  const load = loadModel(coincidentModel(), blobs);
  const triNodes = blobs.get(load.mesh.vertexNodes.tri) as Float64Array;
  const lineNodes = blobs.get(load.mesh.vertexNodes.line) as Float64Array;
  // The connectivity, fanned as mesh/build.ts draws a quad: (c0 c1 c2), (c0 c2 c3).
  assert.deepEqual(Array.from(triNodes), QUADS.flatMap(([a, b, c, d]) => [a!, b!, c!, a!, c!, d!]));
  assert.deepEqual(Array.from(lineNodes), [13, 14]);
  // Each vertex sits where its node is (the map and the positions agree).
  const pos = blobs.f32(load.mesh.triPositions);
  triNodes.forEach((id, v) => assert.deepEqual([pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]], XYZ[id]!.map(Math.fround)));
  // The values: every vertex its own node's, coincident pairs differ, nothing is no data.
  const ids = Float64Array.from(Object.keys(XYZ).map(Number));
  const column = columnOf(ids);
  const rows = [Float64Array.from(ids, value)];
  const tv = vertexValues(triNodes, column, rows, false);
  const lv = vertexValues(lineNodes, column, rows, false);
  triNodes.forEach((id, v) => assert.equal(tv[v], value(id), `triangle vertex ${v} (node ${id})`));
  assert.deepEqual(Array.from(lv), [value(13), value(14)]);
  for (const [a, b] of [[2, 5], [3, 8], [10, 11], [13, 14]] as const) {
    assert.deepEqual(XYZ[a], XYZ[b]);
    assert.notEqual(value(a), value(b));
  }
  const all = (n: number) => Int32Array.from({ length: n }, (_, i) => i);
  const clim = { min: value(1), max: value(14) };
  assert.equal(primitiveColours(tv, all(triNodes.length / 3), 3, clim, "jet").noData, 0, "no triangle is grey");
  assert.equal(primitiveColours(lv, all(1), 2, clim, "jet").noData, 0, "the zero-length spring is not grey");
});
