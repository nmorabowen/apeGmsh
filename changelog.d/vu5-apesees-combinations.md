### Combinations declared on `apeSees` (ADR 0130 slice 5, opensees 2.39.0)

`ops.combination(name, terms, rule=...)` declares a combination next to the
stages it combines. It is archived in `model.h5` at `/opensees/combinations`
and survives `OpenSeesModel.to_h5` and `ModelData.write`. Adding one leaves
`model_hash` unchanged and changes no deck. Nothing evaluates a combination
yet: that is the next slice of the ADR.

- A term is `(stage_name, factor)`, `(combination, factor)`, or a
  `CombinationTerm` for a chosen step or a mode.
- A bad name, a repeated name, a term naming an undeclared combination, and a
  `spectrum` or `reference` term are refused when declared, with nothing
  registered. A case lives in `.user.h5`, so declare that combination there
  after the run.
- A hand-written file is checked when read. A bad row, a duplicate or a cycle
  refuses that row with a warning and the file still opens. Malformed columns
  refuse the zone, and a rewrite of such a file then raises.

The opensees zone moves to 2.39.0, so a 2.38 reader refuses a 2.39 file.
