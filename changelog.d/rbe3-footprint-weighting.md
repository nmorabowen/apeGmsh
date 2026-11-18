### ADDED — `distributing_coupling(weighting="footprint", footprint=[...])`: RBE3 weights from a polygon's overlap with the slave faces

An RBE3 can now spread its reference load as a uniform pressure over a convex
planar polygon on the slave surface, which need not follow the mesh: a column or
pile section on a slab or mat mesh. Each slave face is projected onto the
polygon's plane and clipped by it; the overlap area is split equally among the
face's nodes, and only the nodes with a positive share become independents.
The weights sum to the polygon's area. `footprint=` may also be a list of
polygons whose shares add, for a wall node at a junction that spreads over the
strip of each wall it lies on. `footprint=` is required by this
weighting and refused by `"uniform"` and `"area"`; a degenerate, non-planar or
non-convex polygon is refused when the coupling is declared, and a footprint
that overlaps no face fails when it resolves. Both the build-phase and the
chain-phase resolver read the slave faces for it, as for `"area"`. The record
is the existing `InterpolationRecord` with its `weights`, so nothing changes in
the emit or in `model.h5`.
