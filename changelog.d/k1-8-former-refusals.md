### ADDED — `model.h5` stores every parameter tag; staged `s.update_parameter` and `s.zero_velocities` archive (opensees 2.27.0; K1-8, #1465)

`ops.h5()` used to refuse a stage that called `s.update_parameter` or
`s.zero_velocities`, and it dropped the `parameter` tags the build planned
for initial stresses and absorbing flips, so a deck replayed from the
archive re-minted them. From opensees 2.27.0 both verbs archive, and every
record stores its own tags:

- **`param_tags`** on every `initial_stress/stress_NNN` (global and per
  stage: the XX, YY and ZZ ramp tags), and a `(rank, pid)` row per flip
  written on every `activate_absorbing/absorb_NNN`.
- **`stages/stage_NNN/update_parameter/update_NNN`**: the name, value,
  optional material tag, `pg` or `elements` target and the `(rank, pid)`
  rows of one `s.update_parameter`.
- **`stages/stage_NNN/zero_velocities/zero_NNN`**: one `s.zero_velocities`,
  with its `nodes` (none for the whole domain). The replay zeroes them at
  their archived ndf, after `reset` and before the analyze.

`OpenSeesModel.from_h5(...).build("tcl")` of a 2.27 archive writes the
stored tags and mints none, so it equals the bridge's own deck with no
parameter-tag mask, flat and staged. An archive below 2.27.0 still opens
and replays: its replay re-mints the tags, behind a version check
(`PARAM_TAGS_FROM`), and `to_h5` of it stores the tags that replay mints.
Each new `update_parameter` / `zero_velocities` record is a declaration
with its own `/opensees/decls` key and rows column.

The tag-law waivers `initial-stress-replay` and `staged-replay-params`
become `pre-2-27-initial-stress-replay` and
`pre-2-27-staged-replay-params`, scoped to pre-2.27 archives.

**Changed (behaviour): `model_hash`.** The maintainer ruled that the stored
`param_tags` are hashed: they are authored model state, since they appear
in the deck. Two kinds of archive therefore hash differently once:

- every archive holding a global or per-stage initial stress, or a stage
  absorbing flip, written from 2.27.0 on, because its new `param_tags`
  fold into `/opensees`;
- a pre-2.27 archive of that kind re-saved through
  `OpenSeesModel.to_h5`, because the rewrite stores the tags its replay
  re-mints.

An archive with neither hashes as before. A stage `s.update_parameter` or
`s.zero_velocities` had no archive before (the writer refused it), so it
has no earlier hash to change.

**Changed (behaviour): an empty flip or update target refuses.** An
`s.activate_absorbing` or `s.update_parameter` record whose target holds no
element has no parameter tag. The archive could not tell which record each
of the others belongs to, so `ops.h5()` now refuses the stage. The message
names the stage, the verb and the records' targets. The Tcl and openseespy
decks are unaffected. The refusal covers authoring only: a pre-2.27 archive
that already holds such a record still re-saves through `to_h5`, storing an
empty `param_tags` run (0 rows) for it, and its replay checks that the
record still resolves no element.
