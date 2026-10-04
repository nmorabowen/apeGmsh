### FIXED — ops.model(ndm=2) refuses a node with a non-zero z (#1337)

The emitters trim every node to `ndm` coordinates. That trim used to drop
whatever lay beyond `ndm`. A portal frame drawn in the x-z plane under
`ops.model(ndm=2, ndf=3)` therefore emitted coincident column-end nodes.
The deck gave no warning, and the live run died inside OpenSees with
`LinearCrdTransf2d ... 0 length`. `trim_coords_to_ndm` now drops padding
only. A dropped coordinate that is non-zero (beyond a 1e-9 relative
tolerance) raises a `BridgeError` on every text and live route
(`ops.tcl`, `ops.py`, `ops.analyze`), before the deck is written. The error
names the node, the axis and the value, and it gives the fix: model in the
z = 0 plane, or declare `ndm=3`. Models that already lie in z = 0 emit
byte-identically.
