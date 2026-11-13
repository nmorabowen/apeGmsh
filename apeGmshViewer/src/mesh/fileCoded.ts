// A file-coded element block (code = -class_tag) -> the gmsh code of its shape.
//
// A model.h5 rebuilt from a `.ladruno` or MPCO result file (Python's
// `read_fem_from_ladruno` / `read_fem_from_mpco`) keys each element block by
// `-class_tag`, which names no topology. The block's own `dim` and `npe`
// attributes do: `dim` is the response catalog's parent dimension for a
// catalogued class (the width of its Gauss rows), else the `.ladruno` BASIS
// `TOPOLOGY`, else a class-name guess that Python warned about when it built
// the FEMData ("drawing follow[s] that guess"). This is Python's
// `results/_gauss_extrapolation.py::_GMSH_CODE_BY_PARENT_DIM_NPE`, copied
// entry for entry (test/meshFileCoded.test.ts reads the Python table and
// fails on any difference). Every OpenSees / Ladruno class of these shapes
// numbers its corners first, in gmsh corner order, which is all the mesh
// draws. Wedges and pyramids are absent, as in Python: no fork class has a
// verified corner convention, so they stay undrawn rather than guessed.

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
