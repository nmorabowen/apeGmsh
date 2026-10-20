### FIXED — Triangle and tet Gauss points read in gmsh reference coordinates, so extrapolated corner values and GP world coordinates are no longer rotated (#2228)

`GaussSlab.natural_coords` is now gmsh reference coordinates for every
element family: `[-1, 1]^d` on lines, quads and hexes, and `(u, v[, w])` on
the unit simplex, with `1 - Σ` on node 0, on triangles and tets. These are
the coordinates the Gauss-to-node extrapolation and `global_coords` evaluate.
Most fork elements write their simplex rule node-0-first: column `k` is node
`k`'s weight. The readers passed those rows through unchanged, so every
multi-point simplex Gauss point landed at the wrong place, with no warning.
Two cases show the size of the error:

- A TenNodeTetrahedron's or BezierTet10's extrapolated corner values came
  back rotated by one node.
- A SixNodeTri's or LadrunoLST's Gauss points sat up to the element size
  away from their true position.

The fix converts the rule once, where each reader takes it:

- **The response catalog.** Its simplex tables, which feed MPCO, the
  native capture, the recorder transcoder and the `.ladruno` catalog
  fallback, now hold gmsh coordinates. The new `coord_system` tags,
  `simplex_tri_node0`, `simplex_tet_node0` and `simplex_tri_gmsh`, record
  which frame each element writes its rule in. They replace
  `barycentric_tri` and `barycentric_tet`. BezierTri6 and ASDShellT3
  already write gmsh `(u, v)`, so their values are unchanged.
- **The `.ladruno` reader.** It converts `QUADRATURE/GP_PARAM` by
  `(PARAM_DOMAIN, TOPOLOGY, FAMILY)`. A file whose frame it cannot place
  raises instead of being guessed, for example a `GP_PARAM` with no
  `PARAM_DOMAIN`.
- **The Bezier world-coordinate path.** It rebuilds full barycentric rows
  before it evaluates the Bernstein basis, so BezierTet10 world coordinates
  are unchanged.
- **The MPCO-to-`.ladruno` converter.** It writes the `.ladruno` frame,
  and it writes FAMILY `bernstein` for BezierTri6 and BezierTet10, which it
  used to label `lagrange`.

Single-point centroid rules (Tri31, LadrunoCST, FourNodeTetrahedron) and
every quad and hex are unchanged.
