// A file-coded element block (code = -class_tag) -> the gmsh code of its shape.
//
// A model.h5 rebuilt from a `.ladruno` or MPCO result file (Python's
// `read_fem_from_ladruno` / `read_fem_from_mpco`) keys each element block by
// `-class_tag`, which names no topology. The block's own `dim` and `npe`
// attributes do, but only when `dim` is known: Python sets it to the
// response catalog's parent dimension for a catalogued class (the width of
// its Gauss rows), else the `.ladruno` BASIS `TOPOLOGY`, else a class-name
// guess. The model.h5 does not say which, so a block is drawn only when its
// class (`gmsh_name`) is in the catalog with that same dim
// (CATALOG_PARENT_DIM); any other class is not drawn, and warns. The shape
// is then Python's
// `results/_gauss_extrapolation.py::_GMSH_CODE_BY_PARENT_DIM_NPE`, copied
// entry for entry (test/meshFileCoded.test.ts reads the Python table and
// fails on any difference). Every OpenSees / Ladruno class of these shapes
// numbers its corners first, in gmsh corner order, which is all the mesh
// draws. Wedges and pyramids are absent, as in Python: no fork class has a
// verified corner convention, so they stay undrawn rather than guessed.

/**
 * Python's `mesh/_femdata_dims.py::_catalog_parent_dims()`: each class the
 * response catalog holds, with its parent dimension. Copied entry for entry;
 * test/meshFileCoded.test.ts compares it with the golden the fixture
 * generator writes from Python (test/fixtures/file_coded/README.md).
 */
export const CATALOG_PARENT_DIM: Readonly<Record<string, number>> = {
  ASDShellQ4: 2, ASDShellT3: 2, BbarBrick: 3, BezierTet10: 3, BezierTri6: 2, Brick: 3,
  CorotTruss: 1, CorotTruss2: 1, DispBeamColumn2d: 1, DispBeamColumn3d: 1, EightNodeQuad: 2,
  ElasticBeam2d: 1, ElasticBeam3d: 1, ElasticForceBeamColumn2d: 1, ElasticForceBeamColumn3d: 1,
  ElasticTimoshenkoBeam2d: 1, ElasticTimoshenkoBeam3d: 1, ForceBeamColumn2d: 1, ForceBeamColumn3d: 1,
  ForceBeamColumnCBDI2d: 1, ForceBeamColumnWarping2d: 1, FourNodeQuad: 2, FourNodeTetrahedron: 3,
  InertiaTruss: 1, LadrunoBrick: 3, LadrunoBrick20: 3, LadrunoCST: 2, LadrunoDispBeamColumn2d: 1,
  LadrunoDispBeamColumn3d: 1, LadrunoForceBeamColumn2d: 1, LadrunoForceBeamColumn3d: 1, LadrunoIMKBeam: 1,
  LadrunoIMKBeam2d: 1, LadrunoLST: 2, LadrunoQuad: 2, ModElasticBeam2d: 1, SSPbrick: 3, SSPquad: 2,
  SSPquadUP: 2, ShellDKGQ: 2, ShellDKGT: 2, ShellMITC4: 2, ShellMITC9: 2, ShellNLDKGQ: 2, ShellNLDKGT: 2,
  SixNodeTri: 2, TenNodeTetrahedron: 3, Tri31: 2, Truss: 1, Truss2: 1, Twenty_Node_Brick: 3, ZeroLength: 1,
  ZeroLengthContact: 1, ZeroLengthInterface2D: 1, ZeroLengthND: 1, ZeroLengthSection: 1,
};

/** `(dim, npe)` -> gmsh element-type code, as `"dim,npe"` keys. */
export const GMSH_CODE_BY_DIM_NPE: Readonly<Record<string, number>> = {
  "1,2": 1, // Line2
  "2,3": 2, // Tri3       Tri31, LadrunoCST, ShellDKGT, ASDShellT3
  "2,4": 3, // Quad4      FourNodeQuad, ShellMITC4, ASDShellQ4
  "2,6": 9, // Tri6       SixNodeTri, LadrunoLST, BezierTri6
  "2,8": 16, // Quad8     EightNodeQuad
  "2,9": 10, // Quad9     NineNodeQuad, ShellMITC9
  "3,4": 4, // Tet4       FourNodeTetrahedron
  "3,8": 5, // Hex8       Brick, SSPbrick
  "3,10": 11, // Tet10    TenNodeTetrahedron, BezierTet10
  "3,20": 17, // Hex20    20NodeBrick
  "3,27": 12, // Hex27    27NodeBrick
};

/** True for a block keyed by `-class_tag` rather than a gmsh element-type code. */
export function isFileCoded(code: number): boolean {
  return code < 0;
}

/** The gmsh code a file-coded block of this `dim` and `npe` is drawn as, or null when Python resolves no shape for it. */
export function gmshCodeOfFileCoded(dim: number, npe: number): number | null {
  return GMSH_CODE_BY_DIM_NPE[`${dim},${npe}`] ?? null;
}

/** A file-coded block as the mesh sees it: its class (`gmsh_name`, null when the file has none), code, dim and npe. */
export interface FileCodedBlock {
  className: string | null;
  code: number;
  dim: number;
  npe: number;
}

/**
 * The gmsh code a file-coded block is drawn as, or why it is not drawn.
 * Only a class in CATALOG_PARENT_DIM with the catalog's own dim has a known
 * dim; any other class's dim may be Python's class-name guess, so it is
 * refused rather than drawn as the guess.
 */
export function resolveFileCoded(b: FileCodedBlock): { code: number } | { refused: string } {
  const what = `file-coded type code ${b.code} (dim ${b.dim}, npe ${b.npe})`;
  if (b.className === null) {
    return { refused: `${what} has no gmsh_name, so its class, and whether its dim is known, cannot be told` };
  }
  const named = `file-coded class ${b.className} (code ${b.code}, dim ${b.dim}, npe ${b.npe})`;
  if (!Object.hasOwn(CATALOG_PARENT_DIM, b.className)) {
    return { refused: `${named} is not in Python's response catalog, so its dim may be a class-name guess` };
  }
  const catalogued = CATALOG_PARENT_DIM[b.className]!;
  if (catalogued !== b.dim) {
    return { refused: `${named} contradicts the response catalog's parent dim ${catalogued}` };
  }
  const code = gmshCodeOfFileCoded(b.dim, b.npe);
  return code === null ? { refused: `${named} resolves to no gmsh shape` } : { code };
}
