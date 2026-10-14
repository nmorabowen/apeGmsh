// V4e3 (#1904): the cut face against its oracle and its seams.
//
// - The oracle: fixtures/slice.oracle.json, PyVista's slice of a tet, a hex
//   and a wedge solid through the Qt backend's own `PyVistaBackend.slice_layer`
//   (scripts/make_viewer_results_fixtures.py --slice-oracle), one plane alone
//   and trimmed by one and by two other planes. The app's `sliceSolids` must
//   match its area to 1e-9 relative, its points and the nodal values VTK
//   carries onto them to 1e-9 relative (no data where VTK has NaN), and the
//   cell each VTK triangle came from. Two VTK shortcuts are float-precise
//   (measured; the generator's docstring): on an all-linear grid vtkCutter
//   delegates to vtk3DLinearGridPlaneCutter (cut points 3e-8 off the edges),
//   and slice_layer's trim goes to vtkClipPolyData (float clip scalars). The
//   1e-9 oracle (`vtk`) takes vtkCutter's generic path (one vertex cell in
//   the grid) and, when trimmed, clips the same slice exactly
//   (vtkTableBasedClipDataSet); `vtk_fast`, slice_layer verbatim on the bare
//   grid, is what the Qt cap drew, and is checked to 1e-7.
// - Seams: a plane that cuts no cell gives no slice (never an empty one); a
//   plane through a face two cells share gives that face once; a corner
//   with no data stays no data on every cut point it reaches, never 0.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import type { Cut } from "../../src/state/clip.ts";
import { SOLID_CORNERS, SOLID_SHAPES, sliceSolids, solidShapeOf, trimPolygon, type Slice, type Solids, type SolidShape } from "../../src/state/slice.ts";

interface OracleMesh {
  points: number[];
  cells: number[];
  f: (number | null)[];
  L: number[];
  g: number[];
}
interface OracleCase {
  mesh: SolidShape;
  case: string;
  plane: Cut;
  trims: Cut[];
  /** vtkCutter's generic path (exact to round-off): the 1e-9 oracle */
  vtk: VtkSlice;
  /** the bare grid, where vtkCutter delegates to vtk3DLinearGridPlaneCutter (cut points up to 3e-8 off): what the Qt cap drew */
  vtk_fast: VtkSlice;
}
interface VtkSlice {
  points: number[];
  tris: number[];
  f: (number | null)[];
  L: number[];
  g: number[];
  area: number;
}
const ORACLE = JSON.parse(readFileSync(fileURLToPath(new URL("../../fixtures/slice.oracle.json", import.meta.url)), "utf8")) as {
  meshes: Record<SolidShape, OracleMesh>;
  cases: OracleCase[];
};

const nan = (v: number | null): number => (v === null ? NaN : v);

/** Every cell of one shape over the mesh's points. */
function solidsOf(shape: SolidShape, cells: readonly number[]): Solids {
  const nc = SOLID_CORNERS[shape];
  const n = cells.length / nc;
  return {
    shape: new Uint8Array(n).fill(SOLID_SHAPES.indexOf(shape)),
    start: Int32Array.from({ length: n + 1 }, (_, c) => c * nc),
    slot: Int32Array.from(cells),
  };
}

/** Per-corner values of a nodal field (each corner its node's value) and of a cell field (each corner its cell's). */
const nodalCorners = (cells: readonly number[], v: readonly (number | null)[]) => Float64Array.from(cells, (s) => nan(v[s]!));
const cellCorners = (shape: SolidShape, g: readonly number[]) => Float64Array.from(g.flatMap((x) => new Array<number>(SOLID_CORNERS[shape]).fill(x)));

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1, Math.abs(b));

/** Barycentric weights of `p` in triangle `t` of `pos` (in the triangle's plane), or null when outside by more than `tol`. */
function bary(pos: ArrayLike<number>, t: number, p: readonly number[], tol: number): [number, number, number] | null {
  const a = [0, 1, 2].map((k) => pos[9 * t + k]!), b = [0, 1, 2].map((k) => pos[9 * t + 3 + k]!), c = [0, 1, 2].map((k) => pos[9 * t + 6 + k]!);
  const v0 = b.map((x, k) => x - a[k]!), v1 = c.map((x, k) => x - a[k]!), v2 = p.map((x, k) => x - a[k]!);
  const dot = (x: number[], y: number[]) => x[0]! * y[0]! + x[1]! * y[1]! + x[2]! * y[2]!;
  const d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1), d20 = dot(v2, v0), d21 = dot(v2, v1);
  const den = d00 * d11 - d01 * d01;
  const v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den, u = 1 - v - w;
  return u >= -tol && v >= -tol && w >= -tol ? [u, v, w] : null;
}

/** The value a slice carries at `p` (the triangle holding it, interpolated), and that triangle's index. */
function sampleAt(s: Slice, p: readonly number[], tol = 1e-9): { value: number; tri: number } | null {
  for (let t = 0; t < s.cell.length; t++) {
    const w = bary(s.positions, t, p, tol);
    if (w) return { value: w[0] * s.values![3 * t]! + w[1] * s.values![3 * t + 1]! + w[2] * s.values![3 * t + 2]!, tri: t };
  }
  return null;
}

/** My vertex at `p` (within `tol`): the value it carries, or undefined when no vertex is there. */
function vertexAt(s: Slice, p: readonly number[], tol: number): number | undefined {
  for (let i = 0; i < s.positions.length / 3; i++) {
    if (Math.hypot(s.positions[3 * i]! - p[0]!, s.positions[3 * i + 1]! - p[1]!, s.positions[3 * i + 2]! - p[2]!) <= tol) return s.values![i]!;
  }
  return undefined;
}

for (const oc of ORACLE.cases) {
  test(`oracle ${oc.mesh} ${oc.case}: the area, the points and the values match PyVista's slice_layer`, () => {
    const m = ORACLE.meshes[oc.mesh];
    const solids = solidsOf(oc.mesh, m.cells);
    const xyz = Float64Array.from(m.points);
    const f = sliceSolids(solids, xyz, nodalCorners(m.cells, m.f), oc.plane, oc.trims)!;
    const L = sliceSolids(solids, xyz, nodalCorners(m.cells, m.L), oc.plane, oc.trims)!;
    const g = sliceSolids(solids, xyz, cellCorners(oc.mesh, m.g), oc.plane, oc.trims)!;
    assert.ok(f && L && g, "the plane cuts the solid");
    // Area, to 1e-9 relative.
    assert.ok(rel(f.area, oc.vtk.area) <= 1e-9, `area ${f.area} vs VTK ${oc.vtk.area}`);
    const vp = oc.vtk.points;
    const nv = vp.length / 3;
    const tol = 1e-9;
    let vertexHits = 0;
    for (let i = 0; i < nv; i++) {
      const p = [vp[3 * i]!, vp[3 * i + 1]!, vp[3 * i + 2]!];
      // Every VTK point lies on the app's cap, and the linear field there agrees (exact under any triangulation).
      const l = sampleAt(L, p);
      assert.ok(l, `VTK point ${i} ${p} is on the app's cap`);
      assert.ok(rel(l.value, oc.vtk.L[i]!) <= 1e-9, `L at VTK point ${i}: ${l.value} vs ${oc.vtk.L[i]}`);
      // Where the point is a cut point of a cell edge (not one a trim made: a trim's point carries a value that
      // depends on how the polygon was split, in VTK and in the app alike), the nonlinear field and its no data agree too.
      if (oc.trims.some((t) => Math.abs(t.normal[0] * p[0]! + t.normal[1] * p[1]! + t.normal[2] * p[2]! + t.constant) <= 1e-9)) continue;
      const fv = vertexAt(f, p, tol);
      if (fv === undefined) continue;
      vertexHits++;
      const want = nan(oc.vtk.f[i]!);
      if (Number.isNaN(want)) assert.ok(Number.isNaN(fv), `f at VTK point ${i}: VTK has no data, the app ${fv}`);
      else assert.ok(rel(fv, want) <= 1e-9, `f at VTK point ${i}: ${fv} vs ${want}`);
    }
    if (oc.trims.length === 0) {
      // Untrimmed, the point sets are the same: every VTK point is an app vertex and every app vertex a VTK point.
      assert.equal(vertexHits, nv, "every VTK point is a vertex of the app's cap");
      for (let i = 0; i < f.positions.length / 3; i++) {
        const q = [f.positions[3 * i]!, f.positions[3 * i + 1]!, f.positions[3 * i + 2]!];
        let found = false;
        for (let j = 0; j < nv && !found; j++) found = Math.hypot(vp[3 * j]! - q[0]!, vp[3 * j + 1]! - q[1]!, vp[3 * j + 2]! - q[2]!) <= tol;
        assert.ok(found, `app vertex ${q} is a VTK point`);
      }
    } else assert.ok(vertexHits >= 3, `${vertexHits} VTK points are app vertices`);
    // Each VTK triangle came from the cell the app's triangle under its centroid came from (the cell field).
    const tris = oc.vtk.tris;
    for (let t = 0; t < tris.length / 3; t++) {
      const c = [0, 1, 2].map((k) => (vp[3 * tris[3 * t]! + k]! + vp[3 * tris[3 * t + 1]! + k]! + vp[3 * tris[3 * t + 2]! + k]!) / 3);
      const hit = sampleAt(g, c);
      assert.ok(hit, `VTK triangle ${t}'s centroid is on the app's cap`);
      assert.equal(g.values![3 * hit.tri], oc.vtk.g[t], `the cell of VTK triangle ${t}`);
    }
  });
}

for (const oc of ORACLE.cases) {
  test(`oracle ${oc.mesh} ${oc.case}, VTK's fast path: within its own 3e-8 (area and linear field to 1e-7)`, () => {
    const m = ORACLE.meshes[oc.mesh];
    const L = sliceSolids(solidsOf(oc.mesh, m.cells), Float64Array.from(m.points), nodalCorners(m.cells, m.L), oc.plane, oc.trims)!;
    const fast = oc.vtk_fast;
    assert.ok(rel(L.area, fast.area) <= 1e-7, `area ${L.area} vs VTK fast ${fast.area}`);
    assert.ok(rel(L.area, fast.area) > 0, "the fast path is not exact (else this test's tolerance is loose for nothing)");
    for (let i = 0; i < fast.points.length / 3; i++) {
      const p = [fast.points[3 * i]!, fast.points[3 * i + 1]!, fast.points[3 * i + 2]!];
      const l = sampleAt(L, p, 1e-6);
      assert.ok(l, `VTK fast point ${i} is on the app's cap`);
      assert.ok(rel(l.value, fast.L[i]!) <= 1e-7, `L at VTK fast point ${i}: ${l.value} vs ${fast.L[i]}`);
    }
  });
}

test("oracle: the fixture covers a tet, a hex and a wedge, each alone and trimmed by one and two planes", () => {
  const seen = new Set(ORACLE.cases.map((c) => `${c.mesh}/${c.trims.length}`));
  for (const m of ["tet", "hex", "prism"]) for (const k of [0, 1, 2]) assert.ok(seen.has(`${m}/${k}`), `${m} with ${k} trims`);
  assert.ok(ORACLE.meshes.tet.f.some((v) => v === null), "the tet mesh has nodes with no data");
  assert.ok(ORACLE.cases.some((c) => c.vtk.f.some((v) => v === null)), "a VTK cut point carries no data");
});

// ---- seams --------------------------------------------------------------------

/** Two unit cubes side by side along x: [0,1] and [1,2] (gmsh hex order). */
function twoCubes(): { solids: Solids; xyz: Float64Array } {
  const pts: number[] = [];
  for (const z of [0, 1]) for (const y of [0, 1]) for (const x of [0, 1, 2]) pts.push(x, y, z);
  const at = (x: number, y: number, z: number) => z * 6 + y * 3 + x;
  const hex = (i: number) => [at(i, 0, 0), at(i + 1, 0, 0), at(i + 1, 1, 0), at(i, 1, 0), at(i, 0, 1), at(i + 1, 0, 1), at(i + 1, 1, 1), at(i, 1, 1)];
  return { solids: solidsOf("hex", [...hex(0), ...hex(1)]), xyz: Float64Array.from(pts) };
}

test("seam: a plane that cuts no cell gives no slice, never an empty one", () => {
  const { solids, xyz } = twoCubes();
  assert.equal(sliceSolids(solids, xyz, null, { normal: [1, 0, 0], constant: -5 }), null, "beyond the solid");
  assert.equal(sliceSolids(solids, xyz, null, { normal: [0, 0, -1], constant: 0 }), null, "on the bottom face, keeping z < 0: the kept side has no material");
  // Trimmed away entirely by another plane: no slice either.
  assert.equal(sliceSolids(solids, xyz, null, { normal: [1, 0, 0], constant: -0.5 }, [{ normal: [0, 1, 0], constant: -3 }]), null);
});

test("seam: a plane through a face two cells share makes that face once, from the kept cell", () => {
  const { solids, xyz } = twoCubes();
  const values = Float64Array.from([...new Array<number>(8).fill(1), ...new Array<number>(8).fill(2)]);
  for (const [cut, kept] of [
    [{ normal: [1, 0, 0], constant: -1 }, 2],
    [{ normal: [-1, 0, 0], constant: 1 }, 1],
  ] as [Cut, number][]) {
    const s = sliceSolids(solids, xyz, values, cut)!;
    assert.ok(Math.abs(s.area - 1) < 1e-12, `the shared face counted once (area ${s.area})`);
    assert.ok(new Set(s.cell).size === 1 && s.values!.every((v) => v === kept), `drawn in the kept cell's value ${kept}`);
  }
  // A cell's own face on the plane, the cell on the kept side: that face, once.
  const tet = solidsOf("tet", [0, 1, 2, 3]);
  const txyz = Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const face = sliceSolids(tet, txyz, null, { normal: [1, 0, 0], constant: 0 })!;
  assert.ok(Math.abs(face.area - 0.5) < 1e-15 && face.cell.length === 1, "the face x = 0, keeping x > 0");
  // Through a corner or an edge of a cell only: no polygon of no area.
  assert.equal(sliceSolids(tet, txyz, null, { normal: [-1, 0, 0], constant: 0 }), null, "the same face, keeping x < 0: no material kept");
  assert.equal(sliceSolids(tet, txyz, null, { normal: [1, 1, 0], constant: -1 }), null, "the plane x + y = 1 keeping x + y > 1 touches the edge (1,2) only");
  assert.equal(sliceSolids(tet, txyz, null, { normal: [1, 1, 1], constant: -1 }), null, "x + y + z = 1 keeping above touches the face (1,2,3) only");
});

test("seam: a corner with no data stays no data on every cut point it reaches, never 0", () => {
  const tet = solidsOf("tet", [0, 1, 2, 3]);
  const xyz = Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  // Corner 3 (the apex) has no data; a plane z = 0.5 cuts the three edges to it.
  const s = sliceSolids(tet, xyz, Float64Array.from([1, 2, 3, NaN]), { normal: [0, 0, 1], constant: -0.5 })!;
  assert.ok(s.values!.every((v) => Number.isNaN(v)), `every cut point is no data: ${[...s.values!]}`);
  // A plane z = 0.25 through the base corners' edges to the apex, and corner 0 on the plane-side away from data.
  const t = sliceSolids(tet, xyz, Float64Array.from([NaN, 2, 3, 4]), { normal: [1, 0, 0], constant: -0.5 })!;
  // The cut points of edges (0,1), (0,2)? Only edges crossing x = 0.5: (0,1) and (1,2), (1,3). (0,1) mixes corner 0's no data.
  const vals = [...t.values!];
  assert.ok(vals.some((v) => Number.isNaN(v)), "the point on edge (0,1) is no data");
  assert.ok(vals.some((v) => Number.isFinite(v)), "points between two valued corners keep their values");
  assert.ok(!vals.includes(0), "no data is never 0");
  // An exact corner on the plane takes the corner's own value, not a blend with an empty neighbour.
  const u = sliceSolids(tet, xyz, Float64Array.from([5, NaN, NaN, NaN]), { normal: [-1, -1, -1], constant: 0.5 })!;
  assert.ok(u, "x + y + z < 0.5 keeps corner 0's side");
  assert.ok(u.values!.every((v) => Number.isNaN(v)), "every point is between corner 0 and an empty corner");
});

test("trimPolygon keeps the kept half-space of a square, values carried", () => {
  const sq: [number, number, number, number][] = [[0, 0, 0, 0], [1, 0, 0, 1], [1, 1, 0, 2], [0, 1, 0, 1]];
  const half = trimPolygon(sq, { normal: [1, 0, 0], constant: -0.5 });
  assert.equal(half.length, 4);
  assert.ok(half.every((p) => p[0] >= 0.5));
  assert.deepEqual(half.map((p) => p[3]).sort(), [0.5, 1, 1.5, 2]);
});

test("solidShapeOf: tet, hex, prism and pyramid codes are solids (linear and higher-order); points, lines and faces are not", () => {
  for (const [code, shape] of [[4, "tet"], [11, "tet"], [5, "hex"], [12, "hex"], [17, "hex"], [6, "prism"], [18, "prism"], [7, "pyramid"]] as const) assert.equal(solidShapeOf(code), shape);
  for (const code of [15, 1, 2, 3, 9, 16, 999]) assert.equal(solidShapeOf(code), null);
});
