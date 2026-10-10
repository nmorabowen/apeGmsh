### ADDED — `ops.uniaxialMaterial.Parallel` with `-factors`; FIXED — a tie onto quad4 faces emits a triangle, not a flat tet

`ops.uniaxialMaterial.Parallel(materials=[...], factors=[...])` emits
`uniaxialMaterial Parallel $tag $tag1 ... -factors $f1 ...`: members in
parallel, each scaled by its factor (omit `factors` for all 1). One unit
`Elastic 1.0` and one `Viscous 1.0 1.0` then serve every spring of a
distributed soil-spring bed, with the stiffness and the dashpot coefficient in
the factors; the emitted lines are byte-equal to the San Ramón Tier-2 T2S
deck. A `Viscous` member inside a `Parallel` is now refused by
`section.Aggregator`, which walks the members (a dashpot there is inert).

`g.constraints.tie` (and `tied_contact`) onto a hexahedral master wrote
`ASDEmbeddedNodeElement` with the 4 corners of the quad face, which the
element reads as a tetrahedron: a zero-volume tet, a singular stiffness and a
failed first solve. The emit now writes the triangle of the face's (0, 2)
diagonal split that holds the projected point, as `embedded` does for quad
hosts. The FEM record keeps the 4 corners and their weights, so the
`enforce="equation"` route is unchanged.
