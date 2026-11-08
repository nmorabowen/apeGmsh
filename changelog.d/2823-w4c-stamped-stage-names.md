### FIXED — a `.ladruno` reopened with `model_h5=` no longer takes the program's stage names by position (#2823, #2824)

`Results.from_ladruno(path, model_h5=...)` paired the program's first
`ops.stage` names onto the file's stages by position, and counted a
modes-only stage (`KIND="eigen"`) as a static one. Three producers came
back under the wrong name, two of them silently: `capture_live_modes` and
`ParallelModalResult.to_ladruno` named the modes stage after the program's
first stage (`pushover`), and a fork recorder opened in a later stage named
its stage `gravity`. The `.out` transcode had the same exposure, so W4a kept
any plan that was not a program prefix in a native file.

- **Stamped names bind by name.** A file stamped with
  `INFO/PROVENANCE@stage_names` binds each stage to the program stage its
  stamp names, in any order; a stage stamped with its own `MODEL_STAGE[k]`
  name keeps it. A stamped name the program does not declare, or a program
  stage of another kind than the recorded `KIND`, raises
  `StageStampRefused` naming it.
- **Unstamped stages pair only on a proven kind match.** A modes-only
  stage is never a program stage. Fewer stages than the program pair with
  its first stages only when their `KIND`s fit no other run of program
  stages; otherwise they keep their names and `UnpairedStageWarning` names
  them.
- **Every Python writer stamps the names.** `write_modes` (and so
  `EigenResult.to_ladruno`, `ParallelModalResult.to_ladruno` and
  `capture_live_modes`) stamps the modes stage with its own name.
  `stamp_provenance(..., recorded_stages=[...])` stamps the program stage
  each file stage recorded, checked against the program.
  `stamp_provenance(..., stage_names=[...])` now renames for display only:
  the program's names still bind, the rename is stamped as
  `stage_display_names` and read back as the stages' names, with the
  program name kept as an alias (`r.stage("pushover")` still reads it). It
  raises `ValueError` when the program's names cannot be determined, or
  when a shown name is one another stage answers to (so
  `stage_names=["pushover", "gravity"]` can no longer swap two stages).
  The desktop viewer shows the display names too; it still pairs by the
  lineage hashes.
- **The `.out` transcode writes a `.ladruno` for every nodal plan** whose
  stages are program stages of their kind, in any order or subset, stamped
  with the plan's names. A prefix plan no longer warns
  `StageCountMismatchWarning` on every open, the results are bound to the
  caller's `model=` (`r.model is model`), and a cached file whose binding
  fails is removed so the next read transcodes again.
