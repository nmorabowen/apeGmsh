// VF3-c (#2961): a model.h5 rebuilt from a `.ladruno` keys its element
// blocks by -class_tag, and the mesh draws each as the gmsh shape its
// (dim, npe) resolve to. The oracles are Python's own: the resolution table
// is read from `_gauss_extrapolation.py`, and the fixture counts are what
// Python's Qt viewer scene (`build_fem_scene`) draws from the same files
// (fixtures/file_coded/README.md holds the generator).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { buildMesh } from "../src/mesh/build.ts";
import { GMSH_CODE_BY_DIM_NPE, gmshCodeOfFileCoded } from "../src/mesh/fileCoded.ts";
import type { ModelFile } from "../src/model/types.ts";
import { openModel } from "../src/reader/node.ts";

const here = (p: string): string => fileURLToPath(new URL(p, import.meta.url));

interface Golden {
  file: string;
  blocks: Record<string, { code: number; dim: number; npe: number; cells: number; gmshCode: number | null }>;
  pythonCells: Record<string, number>;
  pythonSkipped: number[];
  pythonSurfaceTriangles: number;
}
const GOLDEN = JSON.parse(readFileSync(here("./fixtures/file_coded/file_coded.golden.json"), "utf8")) as Record<"tri" | "tet", Golden>;

test("the (dim, npe) table is Python's _GMSH_CODE_BY_PARENT_DIM_NPE, entry for entry", () => {
  const py = readFileSync(here("../../src/apeGmsh/results/_gauss_extrapolation.py"), "utf8");
  const body = /_GMSH_CODE_BY_PARENT_DIM_NPE:[^=]*=\s*\{([\s\S]*?)\n\}/.exec(py);
  assert.ok(body, "the Python table is no longer where this test reads it");
  const pyTable: Record<string, number> = {};
  for (const m of body[1]!.matchAll(/\((\d+),\s*(\d+)\):\s*(\d+)/g)) pyTable[`${m[1]},${m[2]}`] = Number(m[3]);
  assert.ok(Object.keys(pyTable).length >= 10, "parsed too few Python entries to be the table");
  assert.deepEqual({ ...GMSH_CODE_BY_DIM_NPE }, pyTable);
});

for (const stem of ["tri", "tet"] as const) {
  test(`${stem}: a .ladruno-rebuilt model draws every file-coded cell Python's viewer draws`, async () => {
    const g = GOLDEN[stem];
    const model = await openModel(here(`./fixtures/file_coded/${g.file}`));
    assert.deepEqual(model.warnings, []);
    // The fixture is what the slice is about: every block is file-coded.
    assert.deepEqual(
      Object.fromEntries(model.blocks.map((b) => [b.alias, { code: b.code, dim: b.dim, npe: b.npe, cells: b.ids.length }])),
      Object.fromEntries(Object.entries(g.blocks).map(([k, b]) => [k, { code: b.code, dim: b.dim, npe: b.npe, cells: b.cells }])),
    );
    for (const b of model.blocks) assert.ok(b.code < 0, `${b.alias} is file-coded`);
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

test("shapes Python leaves unresolved (wedge, pyramid, a contradicting dim) stay undrawn, each with a warning", () => {
  // Six nodes of a unit wedge; the pyramid and the 1-D "quad" reuse them.
  const ids = [1, 2, 3, 4, 5, 6];
  const xyz = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1];
  const model: ModelFile = {
    path: "synthetic.h5", sizeBytes: 0, neutralVersion: "2.35.0", meta: {},
    nodeIds: new Float64Array(ids), nodeCoords: new Float64Array(xyz),
    blocks: [
      { alias: "wedge", code: -1001, dim: 3, npe: 6, ids: new Float64Array([1]), connectivity: new Float64Array([1, 2, 3, 4, 5, 6]) },
      { alias: "pyramid", code: -1002, dim: 3, npe: 5, ids: new Float64Array([2]), connectivity: new Float64Array([1, 2, 3, 4, 5]) },
      { alias: "linequad", code: -1003, dim: 1, npe: 4, ids: new Float64Array([3]), connectivity: new Float64Array([1, 2, 3, 4]) },
      { alias: "tri", code: -1004, dim: 2, npe: 3, ids: new Float64Array([4]), connectivity: new Float64Array([1, 2, 3]) },
      { alias: "gmsh_unknown", code: 999, dim: 3, npe: 4, ids: new Float64Array([5]), connectivity: new Float64Array([1, 2, 3, 4]) },
    ],
    physicalGroups: [], labels: [], opensees: null, sequence: { status: "absent" }, warnings: [], readMs: 0,
  };
  const mesh = buildMesh(model);
  assert.equal(mesh.counts.faceCells, 1);
  assert.equal(mesh.counts.solidCells, 0);
  assert.equal(mesh.triRefs.length, 1);
  assert.deepEqual(mesh.warnings, [
    "/elements/wedge: file-coded type code -1001 (dim 3, npe 6) resolves to no gmsh shape; not drawn (1 cells)",
    "/elements/pyramid: file-coded type code -1002 (dim 3, npe 5) resolves to no gmsh shape; not drawn (1 cells)",
    "/elements/linequad: file-coded type code -1003 (dim 1, npe 4) resolves to no gmsh shape; not drawn (1 cells)",
    "/elements/gmsh_unknown: gmsh type code 999 is not drawn (1 cells)",
  ]);
});
