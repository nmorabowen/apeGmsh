### FIXED — The app draws the cells of a model.h5 rebuilt from a `.ladruno` or MPCO file (#2961)

A `model.h5` written from a FEMData that `read_fem_from_ladruno` or
`read_fem_from_mpco` rebuilt keys its element blocks by `-class_tag`, which
names no shape. The app drew none of those cells and warned, for example,
"gmsh type code -33000 is not drawn", so a BezierTri6 or tet model opened
empty.

The app now draws a file-coded block as the gmsh shape that its `dim` and
`npe` attributes give, through the same table Python's Gauss extrapolation
uses (`_GMSH_CODE_BY_PARENT_DIM_NPE`): lines, 3/6-node triangles,
4/8/9-node quads, 4/10-node tets and 8/20/27-node hexes, corner nodes only.
A block whose `(dim, npe)` names no shape there, such as a wedge or a
pyramid, stays undrawn and warns with its code, dim and npe. Section-plane
caps and slices still read the block's own code, so they do not yet cut
file-coded solids.
