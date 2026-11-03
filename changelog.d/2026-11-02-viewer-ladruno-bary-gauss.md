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
in the banner. `architecture/h5-schema.md` now has the table.

The tests place every vertex, the centroid and an off-centre point through
the viewer's own `shapeOf`. They also extrapolate a linear field, sampled
where the writer puts the Gauss points of a skewed tet and triangle, back
onto each corner exactly. Swapping the two frames makes both tests fail.

On the reference models: `shear_connection_tet` (FourNodeTetrahedron, one
Gauss point) opens with no banner. Its contour is unchanged, because a single
point never needed coordinates. `tims_c3b` is still refused, because
tims-workbench's `enrich_native_ladruno` rewrites the group's `TOPOLOGY` to
`"triangle"`. It also rewrites `GLOBAL_GP_COORDS` node-0-first, which
contradicts the element. With `TOPOLOGY` restored to `"tri"`, the BezierTri6
frame gives inter-element jumps at shared nodes 3.3 times smaller than
node-0-first on the elastic steps.
