### FIXED — an equation-route tie replays from `model.h5` tied: its `equationConstraint` lines are archived (opensees 2.35.0, #2416)

An `enforce="equation"` tie, distributing coupling or embedded node wrote
its `equationConstraint` lines to the bridge's deck, but `model.h5` kept
none of them. A replay (`OpenSeesModel.from_h5(path).build(...)`) wrote
the model untied, with only a `ReplaySkippedStreamWarning`. Since #2405 it
refused by name instead.

- The writer now stores every global `equationConstraint` line, one row
  per tied DOF, in the new optional group
  `/opensees/constraints/equationConstraint`. Each row is linked to the
  `/opensees/program` call that wrote it, so the replay writes it at the
  bridge's position, between the other MP lines and with a named tie's
  comment line. A partitioned archive stores each replicated line once.
- `apeSees.equation_constraint` rows are stored in the same group. Their
  `/opensees/decl_params` records must agree with them, or the file is
  refused as malformed.
- An archive below opensees 2.35.0 still refuses an equation-route tie by
  name, so it is never replayed untied. The same holds for a 2.35 file
  that has lost rows.
- Interim scope: a tie a stage claims (`s.embedded`, `s.tie`) still
  refuses by name, because its lines are not archived yet (#2448).
- The reader refuses the group in a file below 2.35.0, and refuses one
  whose columns disagree, that has a DOF below 1, or that has a zero or
  non-finite coefficient.
- Opensees schema 2.35.0 (additive). A 2.34 reader would have replayed
  the model untied, so it now refuses a 2.35 file. An archive with such a
  line hashes differently once; every other archive keeps its bytes.
