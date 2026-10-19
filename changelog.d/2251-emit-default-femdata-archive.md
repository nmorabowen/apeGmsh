### FIXED — `emit_default` records a hand-built deck's node results with a FEMData archive on stock openseespy, reading ndm, per-node ndf and element classes from the live domain (#2251)

On a build with no MPCO recorder (pip openseespy), `spec.emit_default(...,
model_h5=<fem.to_h5 file>)` refused with "model= declares no ndm": a
FEMData archive has no `/opensees` zone, so it declares no ndm and archives
no per-node ndf or element classes. `model_h5=` now takes a bridge archive
or a FEMData archive. With a FEMData archive the classic-recorder route
reads each fact from the live domain, never from the FEMData:
`ops.getNDM()`, `ops.getNDF(node)` for each recorded node, and
`ops.eleType` / `ops.eleNodes` of every element for the rotation roles.
Each recorded node's `ops.nodeCoord` must match the FEM node's first `ndm`
coordinates, so the live node is known to be the FEM node. The tolerance
follows the mesh: a quarter of the node's distance to the nearest FEM node
at another position. Within it the live point is nearer that node than
any other. A deck that rounds its coordinates (6 significant digits,
float32, or millimetres to 3 decimals) therefore records, while a point
one node over refuses, however fine the mesh. `run.results()` reads the `.out` files into a
native file with no `/opensees` zone.

This records **node** results only. Element results (element, gauss,
line-station and fibre records) over a FEMData archive are the follow-up
#2269. Until it lands, the raw-deck parity row stays open.

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
  pairs no FEM element id with an OpenSees tag (the follow-up is #2269);
- a node that is not in the live domain;
- a node whose live coordinates are not the FEM node's. A deck built from
  another FEMData can share node tags. A planar deck of a column along
  global Z, which maps (x, z) to (x, y), would otherwise record the FEM's
  z motion as `displacement_y`.

Unchanged:

- A bridge archive never asks the live domain. Its `.out` files are
  byte-identical to `main`'s.
- With no `ops.model` declared, the route keeps its "declares no ndm"
  refusal, ahead of the element-record refusal.
- An offline `Results.from_recorders(..., model=<FEMData archive model>)`
  keeps refusing with `OpenSeesModel`'s "has no declared ndm".
