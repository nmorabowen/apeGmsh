### FIXED — `Results.from_mpco(model_h5=...)` binds the archive's stage names and physical groups (#1324, #1325)

An `.mpco` names its stages `MODEL_STAGE[<k>]` and its `MODEL/` group carries
no physical groups, so `results.stage("elastic_50pct")` raised and every
`pg=` query failed with `"No group named ... Available: []"` unless the
in-memory `fem=` was passed too. The sibling `model.h5` that `from_mpco`
requires carries both. It now uses them: the `ops.stage(name=...)` names
from `/opensees/stages` replace the `MODEL_STAGE[<k>]` names when the counts
agree (`MODEL_STAGE[<k>]` stays resolvable as an alias, `StageInfo.aliases`),
and the neutral FEMData archived in `model_h5` is the bound `results.fem`
whenever its node ids cover the capture's. An explicit `fem=` keeps priority.
Two new warnings mark a wrong pairing instead of guessing:
`StageCountMismatchWarning` (the archive declares a different number of
stages; the file's names are kept) and `ModelFemMismatchWarning` (the
archive's FEMData does not cover the capture's nodes; the MPCO `MODEL/`
synthesis is bound, as before). Both live in `apeGmsh.results._bind`.
