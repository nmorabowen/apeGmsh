// VF3-c (#2961): a model.h5 rebuilt from a `.ladruno` keys its element
// blocks by -class_tag, and the mesh draws each block of a catalogued class
// as the gmsh shape its (dim, npe) resolve to. The oracles are Python's own:
// the resolution table is read from `_gauss_extrapolation.py`, the catalog
// classes are `_catalog_parent_dims()` as the generator wrote them, and the
// fixture counts are what Python's Qt viewer scene (`build_fem_scene`) draws
// from the same files (fixtures/file_coded/README.md holds the generator).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { buildMesh } from "../src/mesh/build.ts";
import { CATALOG_PARENT_DIM, GMSH_CODE_BY_DIM_NPE, gmshCodeOfFileCoded } from "../src/mesh/fileCoded.ts";
import type { ElementBlock, ModelFile } from "../src/model/types.ts";
import { openModel } from "../src/reader/node.ts";

const here = (p: string): string => fileURLToPath(new URL(p, import.meta.url));

interface Golden {
  file: string;
  blocks: Record<string, { code: number; dim: number; npe: number; gmshName: string; cells: number; gmshCode: number | null }>;
  pythonCells: Record<string, number>;
  pythonSkipped: number[];
  pythonSurfaceTriangles: number;
}
const GOLDEN = JSON.parse(readFileSync(here("./fixtures/file_coded/file_coded.golden.json"), "utf8")) as Record<"tri" | "tet", Golden>;

/** The `(dim, npe): code` entries of Python's table body, commented lines skipped. */
function parsePyTable(body: string): Record<string, number> {
  const table: Record<string, number> = {};
  for (const line of body.split("\n")) {
    if (line.trim().startsWith("#")) continue;
    const m = line.match(/^\s*\((\d+),\s*(\d+)\):\s*(\d+)/);
    if (m) table[`${m[1]},${m[2]}`] = Number(m[3]);
  }
  return table;
}

test("the (dim, npe) table is Python's _GMSH_CODE_BY_PARENT_DIM_NPE, entry for entry", () => {
  const py = readFileSync(here("../../src/apeGmsh/results/_gauss_extrapolation.py"), "utf8");
  const body = py.match(/_GMSH_CODE_BY_PARENT_DIM_NPE:[^=]*=\s*\{([\s\S]*?)\n\}/);
  assert.ok(body, "the Python table is no longer where this test reads it");
  const pyTable = parsePyTable(body[1]!);
  assert.ok(Object.keys(pyTable).length >= 10, "parsed too few Python entries to be the table");
  assert.deepEqual({ ...GMSH_CODE_BY_DIM_NPE }, pyTable);
});

test("the table parser skips a commented-out entry", () => {
  assert.deepEqual(parsePyTable("    (2, 3): 2,     # Tri3\n    # (3, 6): 6,   # Wedge6, not verified\n    (3, 4): 4,"), { "2,3": 2, "3,4": 4 });
});

test("the catalogued classes are Python's _catalog_parent_dims(), entry for entry", () => {
  const golden = JSON.parse(readFileSync(here("./fixtures/file_coded/catalog_parent_dims.golden.json"), "utf8")) as Record<string, number>;
  assert.ok(Object.keys(golden).length >= 40, "the golden is too short to be the catalog");
  assert.deepEqual({ ...CATALOG_PARENT_DIM }, golden);
});

for (const stem of ["tri", "tet"] as const) {
  test(`${stem}: a .ladruno-rebuilt model draws every file-coded cell Python's viewer draws`, async () => {
    const g = GOLDEN[stem];
    const model = await openModel(here(`./fixtures/file_coded/${g.file}`));
    assert.deepEqual(model.warnings, []);
    // The fixture is what the slice is about: every block is file-coded, of a catalogued class.
    assert.deepEqual(
      Object.fromEntries(model.blocks.map((b) => [b.alias, { code: b.code, dim: b.dim, npe: b.npe, gmshName: b.gmshName, cells: b.ids.length }])),
      Object.fromEntries(Object.entries(g.blocks).map(([k, b]) => [k, { code: b.code, dim: b.dim, npe: b.npe, gmshName: b.gmshName, cells: b.cells }])),
    );
    for (const b of model.blocks) {
      assert.ok(b.code < 0, `${b.alias} is file-coded`);
      assert.equal(CATALOG_PARENT_DIM[b.gmshName!], b.dim, `${b.alias} is catalogued with its dim`);
    }
    for (const [alias, b] of Object.entries(g.blocks)) assert.equal(gmshCodeOfFileCoded(b.dim, b.npe), b.gmshCode, alias);

    const mesh = buildMesh(model);
    assert.deepEqual(mesh.warnings, []);
    assert.deepEqual(g.pythonSkipped, []);
    const py = g.pythonCells;
    assert.equal(mesh.counts.faceCells, (py["tri"] ?? 0) + (py["quad"] ?? 0));
    assert.equal(mesh.counts.solidCells, (py["tet"] ?? 0) + (py["hex"] ?? 0));
    assert.equal(mesh.counts.lineCells, py["line"] ?? 0);
    // The outer surface, triangulated, as pyvista's extract_surface().triangulate() counts it.
    assert.equal(mesh.triRefs.length, g.pythonSurfaceTriangles);
    assert.equal(mesh.triPositions.length, 9 * g.pythonSurfaceTriangles);
  });
}

test("a file-coded cell draws its corner nodes, the leading entries of its row, as Python does", async () => {
  const model = await openModel(here("./fixtures/file_coded/tri_model.h5"));
  const mesh = buildMesh(model);
  for (const alias of ["beziertri6", "ladrunolst", "sixnodetri", "tri31"]) {
    const bi = model.blocks.findIndex((b) => b.alias === alias);
    const b = model.blocks[bi]!;
    const k = mesh.triRefs.findIndex((r) => r.kind === "fem" && r.blockIndex === bi);
    assert.ok(k >= 0, `${alias} is drawn`);
    assert.deepEqual(Array.from(mesh.triNodes.subarray(3 * k, 3 * k + 3)), Array.from(b.connectivity.subarray(0, 3)), alias);
  }
  const qi = model.blocks.findIndex((b) => b.alias === "fournodequad");
  const quadTris = mesh.triRefs.flatMap((r, k) => (r.kind === "fem" && r.blockIndex === qi ? [k] : []));
  assert.equal(quadTris.length, 2);
  const corners = new Set(quadTris.flatMap((k) => Array.from(mesh.triNodes.subarray(3 * k, 3 * k + 3))));
  assert.deepEqual([...corners].sort((x, y) => x - y), Array.from(model.blocks[qi]!.connectivity.subarray(0, 4)).sort((x, y) => x - y));
});

/** A one-block-per-entry model over the six nodes of a unit wedge. */
function synthetic(blocks: Omit<ElementBlock, "ids" | "connectivity">[]): ModelFile {
  const ids = [1, 2, 3, 4, 5, 6];
  const xyz = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1];
  return {
    path: "synthetic.h5", sizeBytes: 0, neutralVersion: "2.35.0", meta: {},
    nodeIds: new Float64Array(ids), nodeCoords: new Float64Array(xyz),
    blocks: blocks.map((b, i) => ({ ...b, ids: new Float64Array([i + 1]), connectivity: new Float64Array(ids.slice(0, b.npe)) })),
    physicalGroups: [], labels: [], opensees: null, sequence: { status: "absent" }, warnings: [], readMs: 0,
  };
}

test("an uncatalogued class (its dim may be Python's name guess) is not drawn, and warns naming class, code, dim and npe", () => {
  // MVLEM_3D: 4 nodes, not in the response catalog, so element_dim guesses its dim (3 in a 3-D model): a Tet4 if drawn.
  const mesh = buildMesh(synthetic([{ alias: "mvlem", code: -212, dim: 3, npe: 4, gmshName: "MVLEM_3D" }]));
  assert.equal(mesh.counts.solidCells, 0);
  assert.equal(mesh.triRefs.length, 0);
  assert.deepEqual(mesh.warnings, [
    "/elements/mvlem: file-coded class MVLEM_3D (code -212, dim 3, npe 4) is not in Python's response catalog, so its dim may be a class-name guess; not drawn (1 cells)",
  ]);
});

test("a file-coded block with no gmsh_name, or a dim the catalog contradicts, is not drawn and warns", () => {
  const mesh = buildMesh(synthetic([
    { alias: "noname", code: -179, dim: 3, npe: 4 },
    { alias: "nullname", code: -179, dim: 3, npe: 4, gmshName: null },
    { alias: "wrongdim", code: -31, dim: 3, npe: 4, gmshName: "FourNodeQuad" },
    { alias: "tet4", code: -179, dim: 3, npe: 4, gmshName: "FourNodeTetrahedron" },
  ]));
  assert.equal(mesh.counts.solidCells, 1);
  assert.deepEqual(mesh.warnings, [
    "/elements/noname: file-coded type code -179 (dim 3, npe 4) has no gmsh_name, so its class, and whether its dim is known, cannot be told; not drawn (1 cells)",
    "/elements/nullname: file-coded type code -179 (dim 3, npe 4) has no gmsh_name, so its class, and whether its dim is known, cannot be told; not drawn (1 cells)",
    "/elements/wrongdim: file-coded class FourNodeQuad (code -31, dim 3, npe 4) contradicts the response catalog's parent dim 2; not drawn (1 cells)",
  ]);
});

test("shapes Python leaves unresolved (wedge, pyramid, a (dim, npe) with no entry) stay undrawn, each with a warning", () => {
  const mesh = buildMesh(synthetic([
    { alias: "wedge", code: -56, dim: 3, npe: 6, gmshName: "Brick" },
    { alias: "pyramid", code: -56, dim: 3, npe: 5, gmshName: "Brick" },
    { alias: "linequad", code: -19, dim: 1, npe: 4, gmshName: "ZeroLength" },
    { alias: "tri", code: -33, dim: 2, npe: 3, gmshName: "Tri31" },
    { alias: "gmsh_unknown", code: 999, dim: 3, npe: 4 },
  ]));
  assert.equal(mesh.counts.faceCells, 1);
  assert.equal(mesh.counts.solidCells, 0);
  assert.equal(mesh.triRefs.length, 1);
  assert.deepEqual(mesh.warnings, [
    "/elements/wedge: file-coded class Brick (code -56, dim 3, npe 6) resolves to no gmsh shape; not drawn (1 cells)",
    "/elements/pyramid: file-coded class Brick (code -56, dim 3, npe 5) resolves to no gmsh shape; not drawn (1 cells)",
    "/elements/linequad: file-coded class ZeroLength (code -19, dim 1, npe 4) resolves to no gmsh shape; not drawn (1 cells)",
    "/elements/gmsh_unknown: gmsh type code 999 is not drawn (1 cells)",
  ]);
});
