### FIXED — apeGmshViewer reads a `.ladruno` Gauss bucket whose element group has no `QUADRATURE/GP_PARAM`, instead of refusing the field

The fork's recorder writes no `QUADRATURE` group for an element class its
rule table does not list. `LadrunoBrick` (class 33002) is one, at fork build
6050d78e: its `MODEL/ELEMENTS` group says `INTEGRATION_RULE 0` and
`TOPOLOGY "custom"`. apeGmshViewer refused every component of such a
bucket, so the YOO Guayaquil mat's `stresses` (95 780 hexes, 8 Gauss
points, 6 components) could not be drawn at all. The message read "its
groups disagree on its components".

The viewer now reads the bucket. The Gauss-point count comes from
`COLUMN_MAP/GAUSS_ID`, each value from the `DATA` column that its
`GAUSS_ID` row names, and the bucket offers no natural coordinates. The
field contours per element. Gauss-point markers and extrapolation to nodes
are refused for that group, with the reason. The open warns once per bucket
and names the class. The viewer never guesses a position from a TypeScript
rule table. A `GP_PARAM` that is present but is not a floating-point
dataset is still refused, and so is a `GAUSS_ID` that does not list each
point once. `Results.from_ladruno` already read these stresses, because
the response catalog places the points; on `T2_solid.ladruno` its values
match the viewer's.

The positions need a fork-side fix: list `ELE_TAG_LadrunoBrick` in the
8-node, 2×2×2 arm (`Hexahedron_8N` / `Hexahedron_GaussLegendre_2`) of
`getGeometryAndIntRuleByClassTag` in
`SRC/recorder/Ladruno_ElementResults.h`, beside `ELE_TAG_Brick`.
`LadrunoBrick` loops its points in i, j, k order with ζ fastest, which is
the order of the tabulated rule, for every `-formulation`. The
single-point forms mirror slot 0 onto all eight points.
