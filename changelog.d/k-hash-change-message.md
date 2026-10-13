### CHANGED — a `model_hash` mismatch across opensees 2.31.0 names the hash change (#1854)

At opensees 2.31.0 the hashed `/opensees/program.decl` column began to carry
declaration ids, so every re-saved archive's `model_hash` changed once.
`Results.lineage` now adds the cause to its "model hash mismatch" warning when
the results file and the model sit on opposite sides of that version, with the
remedy (re-run against the re-saved model, or pair the results with the model
they were run on). The `assess` `RES.LINEAGE` finding names it too. It stays a
warning. A same-era mismatch, or one where either version cannot be read, keeps
the old text.
