### FIXED — a stage-claimed equation tie replays from `model.h5`: each stage block stores its `equationConstraint` lines (opensees 2.37.0, #2448)

An `enforce="equation"` tie, distributing or embedded record that a stage
claims (`s.tie(name=...)`, `s.distributing(...)`, `s.embedded(...)`)
writes `equationConstraint` lines inside that stage's block. Opensees
2.35.0 (#2447) archived only the global lines, so the archive held none of
these: a deck replay (`OpenSeesModel.from_h5(path).build(...)`) refused the
model by name rather than run the stage untied.

- The writer stores each stage's lines under
  `/opensees/stages/stage_NNN/constraints/equationConstraint`, in the
  global group's columns, with `equationConstraint_emit_index` from the
  stage's MP sequence. A partitioned stage captures each line once.
- The replay merges them with the stage's `equalDOF`, `rigidLink`,
  `rigidDiaphragm` and `embeddedNode` rows by emit index, so the replayed
  deck equals the bridge's, flat and partitioned. `to_h5` echoes them.
- `equationConstraint` leaves the `VERBS` ledger (9 to 8 rows). The ADR 0114
  amendment of 2026-10-23 reverses #1341 Q3 for it: no `H5LedgerWarning`
  fires for an equation tie any more.
- A partitioned archive whose stage holds equationConstraint lines and
  repeats a fix still in force (the ghost replay on a rank that does not
  own the node, a zero-weight master's included) refuses by name: its flat
  replay would fix the node twice (#2530). A re-fix after a full
  `remove_sp` release still replays. The refusal lifts when #2530 is
  fixed.
- An archive below 2.37.0, or a `to_h5` rewrite of one, stores no stage's
  line, so a stage-claimed equation tie in it still refuses by name
  ("below opensees 2.37.0, or a rewrite of one"). The
  reader refuses a stage group in a file below 2.37.0, a group without its
  emit index (or the reverse), and an emit index whose length differs from
  the rows, with `MalformedH5Error`.
- Opensees schema 2.37.0 (additive). A 2.36 reader refuses a 2.37 file.
  An archive with no stage-claimed equation tie keeps its bytes apart from
  the version stamp.
