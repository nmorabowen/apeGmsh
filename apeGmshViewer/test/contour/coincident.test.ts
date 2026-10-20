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
import { buildMesh, colourGroups } from "../../src/mesh/build.ts";
import type { ModelFile } from "../../src/model/types.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { columnOf, joinElements, primitiveColours, vertexValues } from "../../src/state/contour.ts";
import { vertexDisplacements } from "../../src/state/deform.ts";
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

// #2359: V8 caps a Map and a Set at 2^24 entries, and a stage or a model can
// hold more nodes or elements. `underCap` stands in a Map and a Set that throw
// past CAP entries, so a model of a few hundred nodes proves that the contour
// lookup and the mesh build hold no collection sized by the node or element
// count. With columnOf, joinElements or buildMesh back on a Map, this throws.

const CAP = 64;

function underCap<T>(run: () => T): T {
  const [RealMap, RealSet] = [globalThis.Map, globalThis.Set];
  class CappedMap<K, V> extends RealMap<K, V> {
    override set(k: K, v: V): this {
      if (!this.has(k) && this.size >= CAP) throw new RangeError(`a Map past the ${CAP}-entry stand-in cap`);
      return super.set(k, v);
    }
  }
  class CappedSet<V> extends RealSet<V> {
    override add(v: V): this {
      if (!this.has(v) && this.size >= CAP) throw new RangeError(`a Set past the ${CAP}-entry stand-in cap`);
      return super.add(v);
    }
  }
  globalThis.Map = CappedMap as MapConstructor;
  globalThis.Set = CappedSet as SetConstructor;
  try {
    assert.throws(() => new Map(Array.from({ length: CAP + 1 }, (_, i) => [i, i])), RangeError, "the stand-in Map bites");
    assert.throws(() => new Set(Array.from({ length: CAP + 1 }, (_, i) => i)), RangeError, "the stand-in Set bites");
    return run();
  } finally {
    globalThis.Map = RealMap;
    globalThis.Set = RealSet;
  }
}

// A 5 x 5 x 4 grid of unit hexes (180 nodes, 100 cells), node index i at
// (ix, iy, iz), its id 5000 - 3 i (descending, so no lookup takes the sorted
// fast path), cell id 1 + its index. Groups: "Small" (the 25 cells of the
// bottom layer) and "All" (every cell), so a bottom cell is in both. Five
// unit lines along the x axis, and one line on a node tag the model lacks.
const [NX, NY, NZ] = [5, 5, 4];
const nodeIndex = (ix: number, iy: number, iz: number) => ix + (NX + 1) * (iy + (NY + 1) * iz);
const N_NODES = (NX + 1) * (NY + 1) * (NZ + 1);
const nodeId = (i: number) => 5000 - 3 * i;
const MISSING = 999_999;

function gridModel(): ModelFile {
  const coords: number[] = [];
  for (let iz = 0; iz <= NZ; iz++) for (let iy = 0; iy <= NY; iy++) for (let ix = 0; ix <= NX; ix++) coords.push(ix, iy, iz);
  const hexes: number[] = [];
  for (let iz = 0; iz < NZ; iz++) {
    for (let iy = 0; iy < NY; iy++) {
      for (let ix = 0; ix < NX; ix++) {
        const c = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
        for (const [dx, dy, dz] of c) hexes.push(nodeId(nodeIndex(ix + dx!, iy + dy!, iz + dz!)));
      }
    }
  }
  const nHex = NX * NY * NZ;
  const lines = [...Array.from({ length: NX }, (_, ix) => [nodeId(nodeIndex(ix, 0, 0)), nodeId(nodeIndex(ix + 1, 0, 0))]), [MISSING, MISSING]];
  const group = (name: string, ids: number[]) => ({ name, dim: 3, tag: name.length, path: `/physical_groups/${name}`, elementIds: Float64Array.from(ids) });
  return {
    path: "grid.h5",
    sizeBytes: 0,
    neutralVersion: "2.35.0",
    meta: {},
    nodeIds: Float64Array.from({ length: N_NODES }, (_, i) => nodeId(i)),
    nodeCoords: Float64Array.from(coords),
    blocks: [
      { alias: "hexa8", code: 5, dim: 3, npe: 8, ids: Float64Array.from({ length: nHex }, (_, e) => 1 + e), connectivity: Float64Array.from(hexes) },
      { alias: "line2", code: 1, dim: 1, npe: 2, ids: Float64Array.from({ length: lines.length }, (_, e) => 1001 + e), connectivity: Float64Array.from(lines.flat()) },
    ],
    physicalGroups: [group("All", Array.from({ length: nHex }, (_, e) => 1 + e)), group("Small", Array.from({ length: NX * NY }, (_, e) => 1 + e))],
    labels: [],
    opensees: null,
    sequence: { status: "absent" },
    warnings: [],
    readMs: 0,
  };
}

test("#2359: past a stand-in Map/Set cap, buildMesh draws the grid's closed form and colourGroups keeps the first group", () => {
  const model = gridModel();
  const free = buildMesh(model);
  const capped = underCap(() => {
    const groups = colourGroups(model);
    return { mesh: buildMesh(model), order: groups.order, byElement: Array.from({ length: 100 }, (_, e) => groups.byElement.get(1 + e)), size: groups.byElement.size };
  });
  const m = capped.mesh;
  assert.deepEqual(m, free, "the build under the cap equals the build without it");
  // The boundary of an a x b x c box of cells: 2(ab + bc + ca) quads, drawn as two triangles each;
  // its outline has V + F - 2 edges (Euler on the sphere), V the boundary's nodes.
  const F = 2 * (NX * NY + NY * NZ + NZ * NX);
  const V = N_NODES - (NX - 1) * (NY - 1) * (NZ - 1);
  assert.equal(m.triRefs.length, 2 * F);
  assert.equal(m.edgePositions.length, 6 * (V + F - 2));
  assert.deepEqual(m.counts, { lineCells: NX, faceCells: 0, solidCells: NX * NY * NZ, opsOnly: 0, points: 0 });
  assert.deepEqual(m.warnings, ["1 node tags referenced by cells are not in /nodes/ids; those cells are not drawn"]);
  // Every drawn vertex sits on the node its id names (id 5000 - 3 i is node index i), on the box's boundary.
  const at = (id: number) => model.nodeCoords.subarray(3 * ((5000 - id) / 3), 3 * ((5000 - id) / 3) + 3);
  for (const [pos, nodes] of [[m.triPositions, m.triNodes], [m.linePositions, m.lineNodes], [m.edgePositions, m.edgeNodes]] as const) {
    assert.equal(pos.length, 3 * nodes.length);
    nodes.forEach((id, v) => {
      const xyz = Array.from(at(id));
      assert.deepEqual(Array.from(pos.subarray(3 * v, 3 * v + 3)), xyz, `vertex ${v} sits on node ${id}`);
      if (pos !== m.linePositions) assert.ok(xyz[0] === 0 || xyz[0] === NX || xyz[1] === 0 || xyz[1] === NY || xyz[2] === 0 || xyz[2] === NZ, `vertex ${v} is on the boundary`);
    });
  }
  // A cell in both groups keeps the smaller (the first after the sort): the bottom layer is "Small".
  assert.deepEqual(capped.order, ["Small", "All"]);
  assert.equal(capped.size, 100);
  assert.deepEqual(capped.byElement, Array.from({ length: 100 }, (_, e) => (e < NX * NY ? 0 : 1)));
  assert.deepEqual(m.legend, [{ name: "All", elements: 75 }, { name: "Small", elements: 25 }, { name: "(no physical group)", elements: NX }]);
});

test("#2359: past a stand-in Map/Set cap, the contour, the deformed shape and the element join keep the first column of a repeated id", () => {
  // The field records every node, descending, and then node index 0 again: the first column (0) wins.
  const ids = Float64Array.from([...Array.from({ length: N_NODES }, (_, i) => nodeId(i)), nodeId(0)]);
  const row = Float64Array.from(ids, (_, j) => j + 0.25);
  const vertexNodes = Float64Array.from([nodeId(0), nodeId(N_NODES - 1), 7, nodeId(17), nodeId(0)]);
  const decls = Object.create(null) as Record<string, { element: { femId: number } }>;
  for (let e = 1; e <= 100; e++) decls[`mesh/element/${e}`] = { element: { femId: e } };
  const rows = (lo: number, hi: number) => Array.from({ length: hi - lo + 1 }, (_, k) => lo + k);
  const got = underCap(() => {
    const column = columnOf(ids);
    const join = joinElements(decls as never, [{ id: "a", eids: Float64Array.from(rows(1, 60)) }, { id: "b", eids: Float64Array.from(rows(61, 100)) }]);
    const dup = joinElements(decls as never, [{ id: "a", eids: Float64Array.from(rows(1, 60)) }, { id: "b", eids: Float64Array.from([...rows(61, 100), 30, 70]) }]);
    return {
      size: column.size,
      values: vertexValues(vertexNodes, column, [row], false),
      magnitude: vertexValues(vertexNodes, column, [row, row], true),
      d: vertexDisplacements(vertexNodes, column, [row, Float64Array.from(row, (x) => 2 * x), Float64Array.from(row, (x) => 3 * x)]),
      join: join.ok ? [join.rows, join.at.size, join.at.get(1), join.at.get(60), join.at.get(61), join.at.get(100), join.at.get(101)] : join,
      dup,
    };
  });
  assert.equal(got.size, N_NODES);
  const want = [0.25, N_NODES - 1 + 0.25, NaN, 17.25, 0.25];
  assert.deepEqual(Array.from(got.values), want);
  assert.deepEqual(Array.from(got.magnitude), want.map((x) => Math.sqrt(x * x + x * x)));
  assert.deepEqual(Array.from(got.d.missing), [0, 0, 1, 0, 0]);
  assert.equal(got.d.count, 1);
  assert.deepEqual(Array.from(got.d.d.subarray(0, 3)), [0.25, 0.5, 0.75]);
  assert.deepEqual(Array.from(got.d.d.subarray(9, 12)), [17.25, 34.5, 51.75]);
  assert.deepEqual(got.join, [100, 100, { group: 0, row: 0 }, { group: 0, row: 59 }, { group: 1, row: 0 }, { group: 1, row: 39 }, undefined]);
  // A scan in row order meets 30 again (group b row 40) before 70: that is the refusal.
  assert.deepEqual(got.dup, { ok: false, code: "duplicate-id", reason: "element 30 has two rows (group a row 29, group b row 40): its value would be a guess" });
});
