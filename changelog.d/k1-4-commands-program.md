### ADDED — model.h5 records the emit order and the calls it used to drop (opensees 2.23.0, K1-4)

`model.h5` gains two tables. `/opensees/program` records the order in
which the bridge emitted the model, as runs of calls.
`/opensees/commands` is a generic store for calls that have no typed
store. Four calls that `ops.h5()` used to drop silently are now
`commands` rows:

- a global `ops.damping.rayleigh`;
- `ops.damping.modal`, which emits two lines, `eigen` and `modalDamping`;
- the `profiler` lines of a stage's `s.profile` bracket, which
  `ops.h5()` used to refuse.

`OpenSeesModel.build('tcl')` replays each line where the bridge emitted
it, and `OpenSeesModel.to_h5` keeps both tables and `model_hash`.

- Three new readers give a record's emit index: `H5Model.program()`,
  `H5Model.commands()` and `H5Model.emit_index(method, row, stage=-1)`.
  `OpenSeesModel` delegates to them. There is no per-record column.
- Some verbs are still not carried by `/opensees`: contact, embedded
  rebar, embedded node and `equationConstraint`. `ops.h5()` now raises
  `H5LedgerWarning` about them, once per bridge for each distinct set of
  such verbs. The VERBS ledger shrank from 14 verbs to 10; the other four
  are live-only modal verbs that never reach `ops.h5()`.
- **`model_hash` changes once for identical models** at 2.23.0, because
  `program` and `commands` are hashed (ADR 0114 Q5). A results file
  paired with a `model.h5` rewritten at 2.23.0 warns once about the
  mismatch.
- An opensees 2.22.x reader refuses a 2.23.0 file. A 2.23 reader still
  opens every file from 2.12.0 on.
