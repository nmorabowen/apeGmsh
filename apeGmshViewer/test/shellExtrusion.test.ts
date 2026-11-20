// Shell and membrane faces drawn extruded by their section thickness: where
// the thickness is read (state/shellThickness.ts: a shell's section, a
// membrane's `-thick` / `-stiff`, a `pg=` membrane's decl fields) and the
// slab built from the drawn faces (renderer/shellSlab.ts): centred on the
// mid-surface, closed by walls only where the surface ends, and following
// the faces when they move (the deformed shape).

import assert from "node:assert/strict";
import { test } from "node:test";
import { buildMesh } from "../src/mesh/build.ts";
import type { DeclTable, Definitions, ElementBlock, ElementMeta, Group, ModelFile, Param } from "../src/model/types.ts";
import { cornerNormals, slabAttribute, slabLinePositions, slabPositions, slabTopology } from "../src/renderer/shellSlab.ts";
import { membraneThickness, sectionThickness, shellThicknessOf } from "../src/state/shellThickness.ts";
import type { Decl, DeclPath } from "../src/state/types.ts";

function sec(tag: number, type: string, params: Param[]): Decl {
  return { path: `opensees/section/${type}_${tag}`, kind: "section", type, name: "", nameSource: "", h5: "", tag, params, refs: {}, links: {}, problems: [], fields: [] } as Decl;
}
const byTag = (...ds: Decl[]) => new Map(ds.map((d) => [d.tag!, d]));
const declsOf = (...ds: Decl[]): Record<DeclPath, Decl> => Object.fromEntries(ds.map((d) => [d.path, d]));

test("a section's thickness: plate h, the sum of a layered shell, a modifier's inner section", () => {
  const plate = sec(1, "ElasticMembranePlateSection", [3e7, 0.2, 0.25, 2.4]);
  const layered = sec(2, "LayeredShell", [3, 10, 0.05, 11, 0.1, 10, 0.05]);
  const fiber = sec(3, "PlateFiber", [7, 0.3]);
  const mod = sec(4, "LadrunoShellModifier", [1, "-m11", 0.5]);
  const fibreBeam = sec(5, "Fiber", []);
  const m = byTag(plate, layered, fiber, mod, fibreBeam);
  assert.equal(sectionThickness(1, m), 0.25);
  assert.ok(Math.abs(sectionThickness(2, m)! - 0.2) < 1e-12);
  assert.equal(sectionThickness(3, m), 0.3);
  assert.equal(sectionThickness(4, m), 0.25, "the modifier draws its inner section's thickness");
  assert.equal(sectionThickness(5, m), null, "a beam section has no thickness");
  assert.equal(sectionThickness(9, m), null, "an unknown tag is refused, not guessed");
  assert.equal(sectionThickness(1, byTag(sec(1, "ElasticMembranePlateSection", [3e7, 0.2, 0, 0]))), null, "a zero thickness draws no slab");
  assert.equal(sectionThickness(2, byTag(sec(2, "LayeredShell", [3, 10, 0.05]))), null, "a truncated layer list is refused");
});

test("a membrane row's thickness: -thick t, or the t of -stiff E nu t", () => {
  assert.equal(membraneThickness(["-nodes", 1, 2, 3, "-method", "fan", "-thick", 0.15, "-mat", 4]), 0.15);
  assert.equal(membraneThickness(["-nodes", 1, 2, 3, "-stiff", 3e7, 0.2, 0.12, "-rho", 2.4]), 0.12);
  assert.equal(membraneThickness(["-nodes", 1, 2, 3, "-method", "tributary"]), null, "a load distributor with no stiffness has no thickness");
});

/** Two quads in z = 0 (nodes 1..6), and a membrane square in z = 3 (nodes 11..14). */
const NODES: Record<number, [number, number, number]> = {
  1: [0, 0, 0], 2: [1, 0, 0], 3: [2, 0, 0], 4: [0, 1, 0], 5: [1, 1, 0], 6: [2, 1, 0],
  11: [0, 0, 3], 12: [1, 0, 3], 13: [1, 1, 3], 14: [0, 1, 3],
};

function model(elementMeta: ElementMeta[], extra: Partial<ModelFile> = {}, records: DeclTable["records"] = []): ModelFile {
  const ids = Object.keys(NODES).map(Number);
  const quads: ElementBlock = { alias: "quad4", code: 3, dim: 2, npe: 4, ids: Float64Array.from([1, 2]), connectivity: Float64Array.from([1, 2, 5, 4, 2, 3, 6, 5]) };
  return {
    path: "synthetic.h5",
    sizeBytes: 0,
    neutralVersion: "2.33.0",
    meta: {},
    nodeIds: Float64Array.from(ids),
    nodeCoords: Float64Array.from(ids.flatMap((id) => NODES[id]!)),
    blocks: [quads],
    physicalGroups: [],
    labels: [],
    opensees: { version: "2.38.0", objects: [], elementMeta, names: [], definitions: { decls: { hasParams: true, records, tags: [], rows: {} } } as unknown as Definitions },
    sequence: { status: "absent" },
    warnings: [],
    readMs: 0,
    ...extra,
  };
}

function meta(type: string, rows: Param[][], femEids: number[]): ElementMeta {
  return { type, path: `/opensees/element_meta/${type}`, ids: Float64Array.from(rows, (_, i) => 100 + i), femEids: Float64Array.from(femEids), args: rows, inlineConnectivity: null };
}

test("each drawn triangle takes its element's thickness: a shell by its section, a minted membrane by its row", () => {
  const plate = sec(7, "ElasticMembranePlateSection", [3e7, 0.2, 0.25, 0]);
  const m = model([
    meta("ShellMITC4", [[7], [7, "-corotational"]], [1, -1]), // the second row is not on a cell
    meta("LadrunoMembrane", [["-nodes", 11, 12, 13, 14, "-method", "fan", "-stiff", 3e7, 0.2, 0.15]], [-1]),
  ]);
  const mesh = buildMesh(m);
  const { triThickness, count } = shellThicknessOf(m, mesh.triRefs, declsOf(plate));
  const got = mesh.triRefs.map((r, i) => [r.kind === "fem" ? `cell${m.blocks[0]!.ids[r.row]}` : "membrane", triThickness[i]!] as const);
  for (const [what, t] of got) {
    if (what === "cell1") assert.equal(t, 0.25);
    else if (what === "cell2") assert.equal(t, 0, "a mesh cell with no shell on it stays a face");
    else assert.ok(Math.abs(t - 0.15) < 1e-7);
  }
  assert.equal(count, got.filter(([, t]) => t > 0).length);
  assert.ok(got.some(([w]) => w === "membrane"));
});

test("a pg= membrane's mesh cells take the thickness of its declaration", () => {
  const pg: Group = { name: "Slab", dim: 2, tag: 1, path: "/physical_groups/Slab", elementIds: Float64Array.from([2]) };
  const records = [{ row: 0, key: "opensees/element/slab", family: "element", name: "slab", synth: false, params: [{ row: 0, type: "LadrunoMembrane", params: { pg: "Slab", thickness: 0.18 }, paramsNames: null, refs: { transf: [], integration: [], section: [] } }] }];
  const m = model([], { physicalGroups: [pg] }, records);
  // No shell type in /element_meta: the membrane's own row is what tells the reader there is one.
  m.opensees!.elementMeta.push(meta("LadrunoMembrane", [["-nodes", 2, 3, 6, 5, "-thick", 0.18, "-mat", 1]], [-1]));
  const mesh = buildMesh(m);
  const { triThickness } = shellThicknessOf(m, mesh.triRefs, {});
  mesh.triRefs.forEach((r, i) => {
    if (r.kind !== "fem") return;
    assert.ok(Math.abs(triThickness[i]! - (m.blocks[0]!.ids[r.row] === 2 ? 0.18 : 0)) < 1e-7);
  });
});

test("a model with no shell or membrane reads no thickness and allocates nothing more", () => {
  const m = model([meta("FourNodeQuad", [[1, "PlaneStress", 1]], [1])]);
  const r = shellThicknessOf(m, buildMesh(m).triRefs, {});
  assert.equal(r.count, 0);
});

// ---- the slab ----

/** A unit square in z = 0 as two triangles (corners 1-2-3, 1-3-4). */
const SQUARE = Float32Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]);
const SQUARE_TAGS = Float64Array.from([1, 2, 3, 1, 3, 4]);

function zRange(slab: Float32Array): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let i = 2; i < slab.length; i += 3) (lo = Math.min(lo, slab[i]!)), (hi = Math.max(hi, slab[i]!));
  return [lo, hi];
}

test("a flat panel becomes a slab centred on its mid-surface, walled on its boundary only", () => {
  const topo = slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0.2, 0.2]))!;
  assert.equal(topo.faces, 2);
  assert.equal(topo.walls, 4, "the four sides; the shared diagonal has none");
  const slab = slabPositions(topo, SQUARE);
  assert.equal(slab.length, 3 * (6 * 2 + 6 * 4));
  const [lo, hi] = zRange(slab);
  assert.ok(Math.abs(lo + 0.1) < 1e-6 && Math.abs(hi - 0.1) < 1e-6, `z in [${lo}, ${hi}]`);
  assert.equal(slabLinePositions(topo, SQUARE).length, 6 * 2 * 4, "with no outline: a top and a bottom rim per wall");
});

test("the mesh outline is drawn on the slab's top and bottom faces, and a rim never doubles it", () => {
  // The outline of the square (its four sides) plus a free-standing edge on no extruded triangle.
  const outline = Float64Array.from([1, 2, 2, 3, 3, 4, 4, 1, 7, 8]);
  const topo = slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0.2, 0.2]), outline)!;
  const lines = slabLinePositions(topo, SQUARE);
  assert.equal(lines.length / 6, 2 * 4, "each side once on top and once below; no rim on top of it, and the stray edge is skipped");
  const zs = new Set<number>();
  for (let i = 2; i < lines.length; i += 3) zs.add(Math.round(lines[i]! * 1e6) / 1e6);
  assert.deepEqual([...zs].sort(), [-0.1, 0.1]);
  // An interior outline edge (the diagonal, as a quad's would be) is drawn too, with no wall under it.
  const inner = slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0.2, 0.2]), Float64Array.from([1, 3]))!;
  assert.equal(slabLinePositions(inner, SQUARE).length / 6, 2 + 2 * 4, "the diagonal top and bottom, and the four walls' rims");
});

test("neighbours that wind opposite ways still meet without a wall or a gap", () => {
  const flipped = Float32Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]);
  const topo = slabTopology(flipped, Float64Array.from([1, 2, 3, 1, 4, 3]), Float32Array.from([0.2, 0.2]))!;
  assert.equal(topo.walls, 4);
  const [lo, hi] = zRange(slabPositions(topo, flipped));
  assert.ok(Math.abs(lo + 0.1) < 1e-6 && Math.abs(hi - 0.1) < 1e-6);
});

test("a change of thickness and a panel at an angle each close with a wall", () => {
  const step = slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0.2, 0.4]))!;
  assert.equal(step.walls, 6, "both sides of the shared diagonal get one");
  // A wall panel standing on edge 1-2 (y = 0), in the plane x-z.
  const tee = Float32Array.from([...SQUARE, 0, 0, 0, 1, 0, 0, 1, 0, 1]);
  const t = slabTopology(tee, Float64Array.from([...SQUARE_TAGS, 1, 2, 9]), Float32Array.from([0.2, 0.2, 0.1]))!;
  assert.equal(t.walls, 4 + 3, "the floor keeps its four sides; the standing panel is walled on all three");
  const n = cornerNormals(t, tee);
  assert.ok(Math.abs(Math.abs(n[3 * 6 + 1]!) - 1) < 1e-6, "the standing panel offsets along y, not the floor's z");
  assert.ok(Math.abs(Math.abs(n[2]!) - 1) < 1e-6, "the floor's corner at the junction keeps its own normal");
});

test("triangles with no thickness are not extruded, and none at all is no slab", () => {
  assert.equal(slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0, 0])), null);
  const half = slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0.2, 0]))!;
  assert.equal(half.faces, 1);
  assert.equal(half.walls, 3, "the extruded half is closed on the diagonal too");
  assert.throws(() => slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0.2])), RangeError);
});

test("the slab follows the faces it was built from: a rotated (deformed) panel takes its slab with it", () => {
  const topo = slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0.2, 0.2]))!;
  // Rotate the square 90 degrees about x: (x, y, z) -> (x, -z, y), so it lies in the x-z plane.
  const moved = new Float32Array(SQUARE.length);
  for (let i = 0; i < SQUARE.length; i += 3) (moved[i] = SQUARE[i]!), (moved[i + 1] = -SQUARE[i + 2]!), (moved[i + 2] = SQUARE[i + 1]!);
  const out = slabPositions(topo, moved, new Float32Array(3 * topo.src.length));
  let ylo = Infinity, yhi = -Infinity;
  for (let i = 1; i < out.length; i += 3) (ylo = Math.min(ylo, out[i]!)), (yhi = Math.max(yhi, out[i]!));
  assert.ok(Math.abs(ylo + 0.1) < 1e-6 && Math.abs(yhi - 0.1) < 1e-6, "the thickness now spans y");
  const [zlo, zhi] = zRange(out);
  assert.ok(zlo > -1e-6 && zhi < 1 + 1e-6);
});

test("each slab vertex copies its corner's colour or contour ramp coordinate", () => {
  const topo = slabTopology(SQUARE, SQUARE_TAGS, Float32Array.from([0.2, 0.2]))!;
  const u = Float32Array.from([0, 0.1, 0.2, 0.3, 0.4, 0.5]); // per drawn corner
  const su = slabAttribute(topo, u, 1);
  assert.equal(su.length, topo.src.length);
  topo.src.forEach((c, v) => assert.equal(su[v], u[c]));
});
