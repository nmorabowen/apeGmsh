### FIXED — apeGmshViewer: native results files written before the simplex Gauss-coordinate fix extrapolate their triangle and tet groups to the right nodes (#2274)

Native `results.h5` files written before #2228 store the triangle and tet
Gauss rows of Tri31, LadrunoCST, SixNodeTri, LadrunoLST, ShellDKGT,
ShellNLDKGT, FourNodeTetrahedron, TenNodeTetrahedron and BezierTet10
node-0-first. The web viewer reads `_natural_coords` itself and evaluates
them as gmsh reference coordinates, so on such a file a Tet10 or tri6
contour in the node modes (`gauss_node`, `gauss_node_discrete`) came back
rotated by one node, with no warning.

The viewer now applies the rule the Python reader got in #2239, with the
same tables, pinned to Python's by a drift test:

- Rows equal to a class's older table are mapped exactly to its current
  one: the same Gauss points, in the same order, in the gmsh frame.
- Rows equal to the current table, and every other class, read unchanged.
- Rows that match neither table are not extrapolated. A warning names the
  class, the group and the file when the file opens, and the node modes
  read "unsupported" for that field. The element-mean contour still draws.
  The fix is to re-capture the results, or re-transcode the recorder output.

`.ladruno` files and files written after #2228 are unchanged.
