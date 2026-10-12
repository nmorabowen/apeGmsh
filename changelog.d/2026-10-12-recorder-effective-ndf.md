### FIXED — declared node recorders use each node's effective ndf, including `ops.ndf` (B5-e, #1745)

The temporary regression from #1738 is fixed. The `recorder.declare` emit now
gets the bridge's per-node effective ndf: the inferred ndf plus the `ops.ndf`
overlay. An element-less master raised by `ops.ndf(h, 6)` in an `ndf=3` solid
model records its rotations at DOFs 4-6 again. They are read by `ops.results`
and by `.read`.

Two related cases:

- **Lowered nodes now refuse.** An element-less node lowered by `ops.ndf(h, 3)`
  in an `ndf=6` model has no rotation DOFs, so a declared rotation there now
  raises a `BridgeError` naming the node and its ndf. Before, the deck wrote
  `-dof 4..6`, and OpenSees answers a DOF past the node's ndf with 0.
- **Only u-p classes block element-less nodes.** An element-less node, such as
  a decoupled or diaphragm master, records its rotation unless the model holds
  a u-p class (`LadrunoUP`). Before, any class the rotation-role table does not
  name blocked it, so a diaphragm master over an `ASDAbsorbingBoundary` soil
  box was refused. A node that an unnamed class touches is still refused.
