### FIXED — `Results.from_mpco(model_h5=...)` binds the archive's stage names and physical groups (#1324, #1325)

An `.mpco` names its stages `MODEL_STAGE[<k>]` and its `MODEL/` group carries
no physical groups, so `results.stage("elastic_50pct")` raised and every
`pg=` query failed with `"No group named ... Available: []"` unless the
in-memory `fem=` was passed too. The sibling `model.h5` that `from_mpco`
requires carries both. It now uses them: the `ops.stage(name=...)` names
from `/opensees/stages` replace the `MODEL_STAGE[<k>]` names in order
(`MODEL_STAGE[<k>]` stays resolvable as an alias, `StageInfo.aliases`), and
the neutral FEMData archived in `model_h5` is the bound `results.fem`
whenever its node ids cover the capture's at the capture's coordinates (the
model's `ndm` columns, or the capture's own column count on a `fem.to_h5`
archive that declares none: a 2-D model on an offset plane, legal since
#1346, compares in `x, y` only) and, on an archive without an element tag
map (`fem.to_h5`, not `ops.h5`), whenever the capture's element ids and
nodes are the archive's: ops tags are read as fem element ids on that route,
and the bridge renumbers elements densely, so a mismatch would answer
`gauss.get(pg=...)` with the wrong elements. An explicit `fem=` keeps
priority. Four warnings in `apeGmsh.results._bind`
mark a doubtful pairing instead of guessing: `ModelFemMismatchWarning` (the
archive's FEMData misses a capture node, is another mesh of the same part
whose ids cover the capture's at other coordinates, or carries no element
tag map and other elements than the capture's; the MPCO `MODEL/`
synthesis is bound, as before), `StageCountMismatchWarning` (a partial run
holds fewer `MODEL_STAGE` groups than the program declares: the names are
paired onto the prefix; more groups than program stages keeps the file's
names), `DuplicateStageNameWarning` (two `ops.stage` blocks share a
name; `stage(name)` picks the first, the `stage_<k>` ids stay unique), and
`ShadowedStageNameWarning` (a stage is named like another stage's
`stage_<k>` id or `MODEL_STAGE[<k>]` alias, which id-first, then name,
resolution makes unreachable by that id or alias).

`results.stage(x)` and the results viewer's `ResultsDirector.set_stage(x)`
now resolve by exact id, then by name, then by alias, so the `stage_<k>` ids
the viewers scope by never land on a program stage that happens to be
*named* `stage_<k>`.
