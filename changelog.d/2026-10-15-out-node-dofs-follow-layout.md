### FIXED — `.out` node recorders take each DOF from the node layout; `model=` is required for node records — **BREAKING** (#2018)

A 2-D frame's `rotation_z` recorded through `spec.emit_recorders`,
`to_tcl_commands` or `to_python_commands` used to land at `-dof 6`. A
planar frame node (ndm 2, ndf 3) has no DOF 6, so every 2-D frame rotation
read back as 0.0, with no warning. Each node record's `-dof` list now comes
from the built model's `ndm` and per-node ndf, through the bridge's
`_node_layout`: a 2-D frame's `rotation_z` is DOF 3, and a 3-D frame keeps
DOFs 1–6. A component that no recorded node carries is refused instead of
being recorded at another DOF. Examples are a 2-D `displacement_z`, which
used to record the rotation, and a rotation on ndf-4/5 u-p or solid nodes.
A translation that some recorded nodes lack is also refused, and so are
nodes that place one component at different DOFs. A rotational DOF also
needs an element that establishes a rotation there. The check uses the
rotation-role table that `recorder.declare` and `.read` use, matched on
the model's element tokens. A 2-D LadrunoUP vertex node's DOF 3 is pore
pressure, so its `rotation_z` is refused instead of recording the
pressure. An unknown element token also blocks the rotations at its
nodes. Rotations that some nodes lack are still recorded, then read back
as NaN (#1971).

**BREAKING:** every node record now needs `model=`, the built
`OpenSeesModel` (for example `OpenSeesModel.from_h5` of the bridge's
`ops.h5(...)` file), on `emit_recorders`, `to_tcl_commands` and
`to_python_commands`. Element records already did. A translation-only
record needs it too. Without the model the routes raise `ValueError` and
name `model=`, and `format_tcl`, `format_python` and `to_ops_args` refuse
a node `LogicalRecorder` that `to_deck` has not mapped.
`OpenSeesModel.nodes_ndf` is the new read-only per-node ndf map.
`emit_logical` and the `.out` read side are unchanged.
