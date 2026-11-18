### FIXED — `capture_live_modes` reads the rotations at embedded and mixed-beam nodes, and refuses a rotation slot it cannot settle instead of writing 0.0 (#2407, #2414)

`capture_live_modes` wrote 0.0 into the rotation slots of a raw deck's
nodes when a live element class it did not know touched them, and only
warned (`RotationSlotUnresolvedWarning`). On stock openseespy a beam node
tied into a brick by `ASDEmbeddedNodeElement ... -rot` lost its nonzero
live rotations, and so did every node of a cantilever with a
`mixedBeamColumn`.

Now `ASDEmbeddedNodeElement` neither gives nor takes away a rotation at
the nodes it ties, so a beam node it touches reads the live
`nodeEigenvector`. A census of stock beam classes types the rotations of
`MixedBeamColumn2d/3d`, `MixedBeamColumnAsym3d`, `ForceBeamColumnCBDI3d`,
`ElasticForceBeamColumn2d/3d`, `ForceBeamColumn2d/3dThermal`,
`DispBeamColumnNL2d/3d`, `DispBeamColumnAsym3d`,
`DispBeamColumn2d/3dThermal`, `TimoshenkoBeamColumn2d/3d` and
`ModElasticBeam3d`. These classes type rotation roles only: a recorder
record of their responses still refuses by name, since no catalog entry
decodes them. A rotation slot the live classes still cannot settle raises
`ValueError` naming the node, its rotation slots and the blocking class,
and nothing is written. A `.ladruno` that held 0.0 there before now fails
to capture.
