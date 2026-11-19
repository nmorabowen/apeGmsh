import assert from "node:assert/strict";
import { test } from "node:test";
import type { OpsObject, SectionProfile } from "../src/model/types.ts";
import { equivalentRectProfile } from "../src/reader/equivalentSection.ts";
import { checkJntOffset, SchemaError, sectionProfileOf } from "../src/reader/read.ts";
import { insertionPointProfileFromFields } from "../src/state/frameExtrusion.ts";

// Review fixes for the frame-extrusion slice: every case here is one where the
// viewer used to draw something the model does not say.

function extent(p: SectionProfile | null | undefined): [number, number, number, number] {
  assert.ok(p, "expected a profile");
  const pts = p.loops.flat();
  const ys = pts.map((q) => q[0]);
  const zs = pts.map((q) => q[1]);
  const r = (v: number): number => Math.round(v * 1e9) / 1e9;
  return [r(Math.min(...ys)), r(Math.max(...ys)), r(Math.min(...zs)), r(Math.max(...zs))];
}

const area = (p: SectionProfile): number => {
  const loop = p.loops[0]!;
  let a2 = 0;
  for (let i = 0; i < loop.length; i++) {
    const [y0, z0] = loop[i]!;
    const [y1, z1] = loop[(i + 1) % loop.length]!;
    a2 += y0 * z1 - y1 * z0;
  }
  return Math.abs(a2) / 2;
};

test("InsertionPoint bbox follows joint_offset.insertion_point_offsets: mirror_2 flips 3, mirror_3 flips 2, axis_2=z maps 3 to -y", () => {
  // Expected extents are hand-derived from src/apeGmsh/opensees/joint_offset.py
  // (L332-335 for the mirrors, L383-384: e2 = ez, e3 = e1 x ez = -y for axis_2="z").
  const bbox = [-0.2, 0.6, -0.1, 0.3];
  const none = new Map();
  assert.deepEqual(extent(insertionPointProfileFromFields({ bbox }, none)), [-0.2, 0.6, -0.1, 0.3]);
  assert.deepEqual(extent(insertionPointProfileFromFields({ bbox, mirror_2: true }, none)), [-0.2, 0.6, -0.3, 0.1]);
  assert.deepEqual(extent(insertionPointProfileFromFields({ bbox, mirror_3: true }, none)), [-0.6, 0.2, -0.1, 0.3]);
  assert.deepEqual(extent(insertionPointProfileFromFields({ bbox, axis_2: "z" }, none)), [-0.3, 0.1, -0.2, 0.6]);
  assert.deepEqual(extent(insertionPointProfileFromFields({ bbox, axis_2: "z", mirror_2: true }, none)), [-0.1, 0.3, -0.2, 0.6]);
});

test("InsertionPoint depth without width draws nothing (Python reads width as 0), and a square is never axisymmetric", () => {
  const none = new Map();
  assert.equal(insertionPointProfileFromFields({ depth: 0.5 }, none), null);
  const sq = insertionPointProfileFromFields({ depth: 0.4, width: 0.4 }, none);
  assert.ok(sq);
  assert.equal(sq.axisymmetric, undefined);
});

test("Tier-3 equivalent rectangle keeps A, is exact for a true rectangle, and is never axisymmetric", () => {
  // 0.6 (y, depth) x 0.3 (z): A = 0.18, Iz = b h^3/12 = 0.0054, Iy = h b^3/12 = 0.00135
  const rect = equivalentRectProfile(0.18, 0.0054, 0.00135)!;
  assert.deepEqual(extent(rect), [-0.3, 0.3, -0.15, 0.15]);
  // HSS10x10x1/2: A = 17.2, I = 256. The old inversion drew 13.4 x 13.4 (area 179.6).
  const hss = equivalentRectProfile(17.2, 256, 256)!;
  assert.ok(Math.abs(area(hss) - 17.2) < 1e-9);
  assert.equal(hss.axisymmetric, undefined);
  // 2-D keeps A and Iz
  const r2 = equivalentRectProfile(0.18, 0.0054)!;
  assert.deepEqual(extent(r2), [-0.3, 0.3, -0.15, 0.15]);
  for (const bad of [Infinity, Number.NaN, 0, -1, "0.1"]) {
    assert.equal(equivalentRectProfile(bad, 0.0054, 0.00135), null);
    assert.equal(equivalentRectProfile(0.18, bad), null);
    assert.equal(equivalentRectProfile(0.18, 0.0054, bad), null);
  }
});

function fiber(rows: unknown[][]): OpsObject {
  return {
    family: "section",
    path: "/opensees/sections/Fiber_1",
    groupName: "Fiber_1",
    type: "Fiber",
    tag: 1,
    params: [],
    attrs: {},
    tables: {
      patches: {
        path: "/opensees/sections/Fiber_1/patches",
        columns: ["kind", "material_ref", "ny", "nz", "coords"],
        rows: rows.map((c) => [c[0], "/opensees/materials/Steel01_1", 8, 2, c[1]]),
      },
    } as OpsObject["tables"],
  };
}

test("fiber patches the reader cannot draw exactly give no profile instead of a partial one", () => {
  // two 180 deg sectors of one ring: a full ring
  const ring = sectionProfileOf(fiber([["circ", [0, 0, 0, 0.2, 0, 180]], ["circ", [0, 0, 0, 0.2, 180, 360]]]))!;
  assert.equal(ring.source, "patches");
  assert.equal(ring.axisymmetric, true);
  assert.deepEqual(extent(ring), [-0.2, 0.2, -0.2, 0.2]);
  // two sectors that do not close the ring
  assert.equal(sectionProfileOf(fiber([["circ", [0, 0, 0, 0.2, 0, 90]], ["circ", [0, 0, 0, 0.2, 180, 270]]])), undefined);
  // a circ beside a rect
  assert.equal(sectionProfileOf(fiber([["circ", [0, 0, 0, 0.2]], ["rect", [-0.5, -0.1, 0.5, 0.1]]])), undefined);
  // two non-axis-aligned quads
  const q1 = [0, 0, 1, 0.2, 1, 1, 0, 0.8];
  const q2 = [2, 0, 3, 0.2, 3, 1, 2, 0.8];
  assert.equal(sectionProfileOf(fiber([["quad", q1], ["quad", q2]])), undefined);
  // a rect beside a skew quad
  assert.equal(sectionProfileOf(fiber([["rect", [-0.5, -0.1, 0.5, 0.1]], ["quad", q1]])), undefined);
  // an unknown patch kind
  assert.equal(sectionProfileOf(fiber([["rect", [-0.5, -0.1, 0.5, 0.1]], ["spline", [0, 0]]])), undefined);
  // a single rect still draws
  assert.deepEqual(extent(sectionProfileOf(fiber([["rect", [-0.5, -0.1, 0.5, 0.1]]]))), [-0.5, 0.5, -0.1, 0.1]);
});

function transf(jnt: unknown[][], tags: number[]): OpsObject {
  return {
    family: "geomTransf",
    path: "/opensees/transforms/Linear_1",
    groupName: "Linear_1",
    type: "Linear",
    tag: 1,
    params: [0, 0, 1],
    attrs: {},
    tables: {
      jnt_offset: { path: "/opensees/transforms/Linear_1/jnt_offset", columns: ["0", "1", "2", "3", "4", "5"].slice(0, jnt[0]?.length ?? 6), rows: jnt },
      per_element_emitted_tag: { path: "/opensees/transforms/Linear_1/per_element_emitted_tag", columns: ["0"], rows: tags.map((t) => [t]) },
    } as OpsObject["tables"],
  };
}

test("a jnt_offset the bridge could not have written is refused by path, as the Python reader does", () => {
  checkJntOffset(transf([[0.1, 0, 0, -0.1, 0, 0]], [1]));
  checkJntOffset(transf([[0.1, 0, -0.1, 0]], [1]));
  assert.throws(() => checkJntOffset(transf([[0.1, 0, 0, -0.1, 0]], [1])), SchemaError);
  assert.throws(() => checkJntOffset(transf([[0.1, 0, 0, -0.1, 0, Number.NaN]], [1])), SchemaError);
  assert.throws(() => checkJntOffset(transf([[0.1, 0, 0, -0.1, 0, 0]], [1, 2])), /jnt_offset has 1 rows for 2 emitted tags/);
});
