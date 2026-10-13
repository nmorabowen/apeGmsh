### CHANGED — a `model_hash` mismatch across opensees 2.31.0 names the hash change (#1854)

At opensees 2.31.0 the hashed `/opensees/program.decl` column began to carry
declaration ids, so every re-emitted archive's `model_hash` changed once.
`Results.lineage` now adds the cause to its "model hash mismatch" warning when
one side's hash includes the ids and the other's does not, with the remedy
(re-run against the re-saved model, or pair the results with the model they
were run on). Each side's era is its `/opensees/decls@program_decl` marker, not
its version stamp: a `to_h5` rewrite of an older archive is stamped 2.31.0 but
keeps the old hash. The `assess` `RES.LINEAGE` finding names the cause too. It
stays a warning. A same-era mismatch, or a results file with no `/opensees`
zone, keeps the old text.
