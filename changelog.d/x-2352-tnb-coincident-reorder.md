### FIXED — a Twenty_Node_Brick on a coincident node is no longer refused over a FEMData archive (#2352)

On a stock build, `spec.emit_default(...)` over a FEMData archive
(`fem.to_h5`) checks that each recorded node at a shared position is
pinned by the live elements touching it (#2251, #2342). That check
compared each element's `eleNodes` with its FEM connectivity in Gmsh order.
A `Twenty_Node_Brick` holds its nodes in upstream's serendipity order
(#2313), so a correctly tagged brick touching a coincident node was refused
with "FEM element ... has nodes ..., the live element ... has ...".

The check now compares each element in its live class's node order, as the
element pairing already does. A brick with a zero-length spring to a
coincident node reads back equal to the live `nodeDisp` at rtol 1e-12 on
pip openseespy. A swap of the two coincident tags still refuses, naming the
node, the element and the order it expected.

The `solver` parameter of `capture_live_modes` now also says that
`"-genBandArpack"` is the stock default, so passing it does not guard an
indefinite stiffness on its own.
