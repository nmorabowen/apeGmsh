### FIXED — `Results.lineage` and `assess` name the opensees 2.32.0 side-channel hash change (#1911)

At opensees 2.32.0 (#1853) the calls of the six side-channel verbs began to carry their declaration ids in the hashed `/opensees/program.decl` column. The `program_decl` marker reads `"complete"` on both sides of that step, so a results file from a 2.31 emit paired with the re-emitted model showed a plain `model hash mismatch`. It now carries a cause line:

> likely cause: the opensees schema 2.32.0 hash change (side-channel calls' declaration ids entered model_hash); re-run the analysis against the re-saved model, or open the results with the model.h5 they were run on

`assess` names it in `RES.LINEAGE`. The line appears when both markers are `"complete"`, both sides carry side-channel calls, and exactly one side's calls carry their declaration ids. The rule reads that column, not the version stamp: a `to_h5` rewrite of a 2.31 archive is stamped current but keeps the old column and hash, so it counts as the 2.31 side. A pair with no side-channel call, or a mismatch within one version, keeps the plain text. It stays a warning. A later `model_hash` change gets a new marker value, not another rule of this kind.
