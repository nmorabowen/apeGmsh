### CHANGED — `Results.from_ladruno(model_h5=)` takes the program's stage names and kinds (#2101)

A `.ladruno` opened beside its `model.h5` now names and kinds its stages
after the `ops.stage(name=...)` blocks in `/opensees/stages`, as
`Results.from_mpco` does (ADR 0122 design point 9): `results.stage("gravity")`
works on a `.ladruno`, and `MODEL_STAGE[<k>]` stays an alias. The count,
duplicate and shadow rules are `from_mpco`'s. A `.ladruno` stage also
records the `KIND` it ran; where that disagrees with the program kind the
pairing is misaligned, so nothing is renamed and `StageKindMismatchWarning`
is the one warning (#2079 Q7). Before, a partial run, a duplicate name or a
shadowing name in that case also warned as if the names had been paired.
Without `model_h5=` the file's names stand.
