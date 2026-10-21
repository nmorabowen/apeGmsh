### FIXED — Gauss-point world coordinates with a file's own FEMData place P2 simplices by their shape functions (program slice B, #2304)

With a FEMData read from a `.ladruno` or MPCO file's own `MODEL` (codes `-class_tag`), the Gauss-point
world coordinates behind `GaussSlab.global_coords`, the viewer's Gauss-point markers, principal-direction
arrows and Gauss pick targets mapped only linear classes to a shape. SixNodeTri, LadrunoLST and
TenNodeTetrahedron took the centroid+bbox approximation, which for a simplex is not even the right affine
map. So did BezierTri6 and BezierTet10 on MPCO, whose FEMData calls them order 1. Each GP landed about
half an element off, and the warning reached only stderr. Such an element is now resolved from its node
count and the width of its own Gauss rows, the way the Gauss-to-node extrapolation resolves it, and placed
by its full shape functions; a Bezier class's basis order comes from its control-point count. The GPs match
the engine's to round-off, checked against the recorded strain under `u = x²/2` and the fork's
`GLOBAL_GP_COORDS`. The bbox approximation remains for classes with no shape functions, and the warning
now names each class and why.
