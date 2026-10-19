### FIXED — `emit_default` records a hand-built deck with a FEMData archive on stock openseespy, reading ndm, per-node ndf and element classes from the live domain (#2251)

On a build with no MPCO recorder (pip openseespy), `spec.emit_default(...,
model_h5=<fem.to_h5 file>)` refused with "model= declares no ndm": a
FEMData archive has no `/opensees` zone, so it declares no ndm and archives
no per-node ndf or element classes. `model_h5=` now takes a bridge archive
or a FEMData archive. With a FEMData archive the classic-recorder route
reads each fact from the live domain, never from the FEMData:
`ops.getNDM()`, `ops.getNDF(node)` for each recorded node, and
`ops.eleType` / `ops.eleNodes` of every element for the rotation roles.
`run.results()` reads the `.out` files into a native file with no
`/opensees` zone.

On real stock openseespy 3.7.1, the `column_to_surface` case-study deck
records every node's displacement and reaction, and the reference nodes'
rotations, equal to the live `nodeDisp` / `nodeReaction` at rtol 1e-12.
The route raises its one `DefaultRouteWarning` and nothing else.

With a FEMData archive the route refuses, by name:

- a rotation at a node whose live ndf has no rotational DOF, since the read
  has no per-node ndf to mask the recorder's padded column;
- a rotation the live element classes do not establish, for example at a
  node that only a truss or an unlisted class touches;
- element, gauss, line-station and fibre records, because a FEMData archive
  pairs no FEM element id with an OpenSees tag;
- a node that is not in the live domain.

Unchanged:

- A bridge archive never asks the live domain. Its `.out` files are
  byte-identical to `main`'s.
- With no `ops.model` declared, the route keeps its "declares no ndm"
  refusal.
- An offline `Results.from_recorders(..., model=<FEMData archive model>)`
  keeps refusing with `OpenSeesModel`'s "has no declared ndm".
