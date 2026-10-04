### FIXED — ops.model(ndm=2) refuses nodes that do not share one z (#1337)

The emitters trim every node to `ndm` coordinates. That trim used to drop
whatever lay beyond `ndm`. A portal frame drawn in the x-z plane under
`ops.model(ndm=2, ndf=3)` therefore emitted coincident column-end nodes.
The deck gave no warning, and the live run died inside OpenSees with
`LinearCrdTransf2d ... 0 length`.

Dropping an axis is lossless only when every node has the same value on
it. A new `DroppedAxisGuard` (`opensees/emitter/base.py`) does the trim
in every emitter: `ops.tcl`, `ops.py`, `ops.analyze` (live), `ops.h5`
and the recording emitter. The first node fixes the reference value of
the dropped axis. A later node that differs from it by more than a 1e-9
relative tolerance raises a `BridgeError` before any file is written. The
error names both nodes, the axis and the two values, and it gives the
fix: model in one plane of constant z, or declare `ndm=3`.

- A model in the plane z = z0 is still accepted, and its deck is
  identical to the one at z = 0. `interop/strut_tie` meshes at the STM's
  z and relies on this.
- Models that already lie in z = 0 emit byte-identically.
