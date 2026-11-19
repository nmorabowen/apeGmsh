### FIXED — a partitioned staged archive no longer replays a ghost node's fixes twice (#2530)

A stage-claimed MP tie that crosses ranks declares its far nodes as ghosts
on the rank that does not own them, and each ghost re-plays its owner's
SP history inside the stage (ADR 0027 INV-2). The archive stores those
calls as rows of the stage, so a deck replay
(`OpenSeesModel.from_h5(path).build("tcl")`) wrote them again. A stage
`equalDOF` from the fixed base node to another rank fixed that node twice
with no warning. An equation tie in the same position was refused by name
(#2448).

- The deck replay drops a stage `fix` or `remove_sp` row whose
  `/opensees/decls` declaration first appears in an earlier scope (the
  global fixes or an earlier stage). That is a ghost's replay of an
  earlier call. The replayed deck now equals the bridge's for every MP
  kind, flat and partitioned.
- The #2448 refusal is gone. A deliberate re-fix with the base fix's mask
  after a release now replays exactly; before, it was refused too.
- The writer and the schema are unchanged (opensees 2.40.0). An archive
  written before this fix replays by the same rule, and `to_h5` echoes
  the rows.
- A partitioned archive without declaration rows (below opensees 2.25.0,
  or a rewrite of one) refuses by name when a stage row repeats an earlier
  `fix` or `remove_sp`, rather than replay a duplicate.
