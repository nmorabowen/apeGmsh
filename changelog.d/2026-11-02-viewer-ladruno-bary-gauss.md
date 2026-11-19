### ADDED — apeGmshViewer places the Gauss points of a `.ladruno` simplex group whose `PARAM_DOMAIN` is `"bary"`

`apeGmshViewer` used to keep natural coordinates only for `.ladruno` element
groups with `PARAM_DOMAIN = "[-1,1]"`. A triangle or tet group (`"bary"`)
showed the banner "its Gauss points are read, but not extrapolated to nodes",
so it had no `gauss_node` contour when it had more than one Gauss point, and
no Gauss markers. The reader now converts a `"bary"` `GP_PARAM` to gmsh
`(u, v[, w])` with Python's rule (`gp_param_to_gmsh`, ported as
`baryToGmsh` in `reader/ladrunoElements.ts`). The element decides which node
each barycentric column weights:

- Lagrange `tri`/`tet` (FourNodeTetrahedron, Tri31, LadrunoLST, ...) and
  BezierTet10 are node-0-first, as the fork's `computeGlobalGP` writes them.
- BezierTri6 (`tri`/`bernstein`) is gmsh already: its node 0 carries
  `1 - ξ1 - ξ2` (`BezierTri6.cpp`).

Any other `TOPOLOGY` or `FAMILY` is still refused by name, with the reason
in the banner. So is a row that is not a point of the simplex (a weight below
0, or weights summing above 1, beyond rounding): the banner names the row.
Python's `gp_param_to_gmsh` converts such a row without a check.
`architecture/h5-schema.md` now has the table.

A BezierTri6 or BezierTet10 group with more than one Gauss point used to
draw its element means under `gauss_node` (the `cell_averaged` fallback of
#2940 and #2950). It is now extrapolated to its nodes like any other group.
The fallback stays for a group that still has no natural coordinates.

The tests place every vertex, the centroid and an off-centre point through
the viewer's own `shapeOf`. They also extrapolate a linear field, sampled
where the writer puts the Gauss points of a skewed tet and triangle, back
onto each corner exactly. Swapping the two frames makes both tests fail.
On the fork's own `tests/fixtures/simplex_gp/tri.ladruno` and `tet.ladruno`
(Tri31, BezierTri6, FourNodeTetrahedron, TenNodeTetrahedron, BezierTet10),
the viewer's rows equal Python's `gp_param_to_gmsh` to 1e-15.

On the reference models: `shear_connection_tet` (FourNodeTetrahedron, one
Gauss point) opens with no banner. Its contour is unchanged, because a single
point never needed coordinates. `tims_c3b` is still refused, because
tims-workbench's `enrich_native_ladruno` rewrites the group's `TOPOLOGY` to
`"triangle"`. It also rewrites `GLOBAL_GP_COORDS` node-0-first, which
contradicts the element. With `TOPOLOGY` restored to `"tri"`, the BezierTri6
frame gives inter-element jumps at shared nodes 3.3 times smaller than
node-0-first on the elastic steps.
