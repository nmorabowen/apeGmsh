### FIXED — `FootfallResult.to_results` refuses the archive of another model of the same fem (program B, #2801)

`to_results(fem, path, model_h5=...)` checked only that the archive's
`fem_hash` was `fem.snapshot_id`. Two archives of one fem, such as a cracked
and an uncracked stiffness variant, share that hash and differ only in
`model_hash`, so either was accepted and the `.user.h5` was stamped with the
`model_hash` of whichever archive was passed. `footfall_walking` now keeps
its bridge's lineage key, and `to_results` raises `UserWriteRefused`, naming
both hashes and writing nothing, when the archive's `model_hash` is not the
one of the model the evaluation solved. It refuses only when both sides
carry a `model_hash`. An archive with only a `fem_hash` still pairs, and so
does a result whose model archive this process never wrote (for example one
written in another process).
