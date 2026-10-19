### FIXED — Native results files written before the simplex Gauss-coordinate fix read their triangle and tet Gauss points in the right place (#2239)

Before #2228, `DomainCapture` and the recorder transcoder stored the
node-0-first rows of SixNodeTri, LadrunoLST/CST, Tri31, ShellDKGT/NLDKGT,
FourNodeTetrahedron, TenNodeTetrahedron and BezierTet10 as
`_natural_coords`. Since #2228 the readers take `natural_coords` as gmsh
reference coordinates, so those older files read with every multi-point
Gauss point misplaced. A TenNodeTetrahedron's extrapolated corner values
came back rotated by one node, with no warning.

The native reader now recognises the older rows by value, with no format
or schema change:

- Rows equal to the older table are mapped exactly to the current one:
  the same Gauss points, in the same order, in the gmsh frame.
- Rows equal to the current table, and every other class, read unchanged.
- Rows that match neither table read unchanged, and a
  `LegacySimplexGaussCoordsWarning` names the class and the file once per
  class. The fix is to re-capture the results, or re-transcode the
  recorder output.

The web viewer reads native files on its own and does not apply this
mapping yet.
