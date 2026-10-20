### FIXED — Gauss-to-node extrapolation uses the Gauss positions with a file's own FEMData (program slice B, #2240)

A FEMData read from a `.ladruno` or MPCO file's own `MODEL` keys each class by `-class_tag`, which the
gmsh-keyed shape-function catalog did not know. So every multi-point element (SixNodeTri, LadrunoLST,
BezierTri6, TenNodeTetrahedron, BezierTet10, and also FourNodeQuad and Brick) was averaged to its element
mean, silently, and a linear field came back flat. The extrapolation now resolves such an element from its
node count and the width of its own Gauss rows. It paints only the element's corners. A linear field is
reproduced exactly at the nodes, checked against the engine's recorded strain under `u = x²/2` and against
its `GLOBAL_GP_COORDS`. An element whose Gauss positions cannot be used (no shape functions for its
topology, or rows with no natural coordinates) still takes the mean, but one `WarnGaussExtrapolationDegraded`
now names its class. This applies to gmsh-coded FEMData too.
