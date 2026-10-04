### FIXED — `Results.from_mpco(model_h5=...)` binds the archive's stage names and physical groups (#1324, #1325)

An `.mpco` names its stages `MODEL_STAGE[<k>]` and its `MODEL/` group carries
no physical groups, so `results.stage("elastic_50pct")` raised and every
`pg=` query failed with `"No group named ... Available: []"` unless the
in-memory `fem=` was passed too. The sibling `model.h5` that `from_mpco`
requires carries both. It now uses them: the `ops.stage(name=...)` names
from `/opensees/stages` replace the `MODEL_STAGE[<k>]` names in order
(`MODEL_STAGE[<k>]` stays resolvable as an alias, `StageInfo.aliases`), and
the neutral FEMData archived in `model_h5` is the bound `results.fem`
whenever its node ids cover the capture's at the capture's coordinates. An
explicit `fem=` keeps priority. Three warnings in `apeGmsh.results._bind`
mark a doubtful pairing instead of guessing: `ModelFemMismatchWarning` (the
archive's FEMData misses a capture node, or is another mesh of the same part
whose ids cover the capture's at other coordinates; the MPCO `MODEL/`
synthesis is bound, as before), `StageCountMismatchWarning` (a partial run
holds fewer `MODEL_STAGE` groups than the program declares: the names are
paired onto the prefix; more groups than program stages keeps the file's
names), and `DuplicateStageNameWarning` (two `ops.stage` blocks share a
name; `stage(name)` picks the first, the `stage_<k>` ids stay unique).

`results.stage(x)` and the results viewer's `ResultsDirector.set_stage(x)`
now resolve by exact id, then by name, then by alias, so the `stage_<k>` ids
the viewers scope by never land on a program stage that happens to be
*named* `stage_<k>`.
