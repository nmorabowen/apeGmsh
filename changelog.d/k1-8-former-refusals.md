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
The tags fold into `model_hash`, so an initial-stress or absorbing archive
hashes differently once at this minor; other files hash as before.

A flip or update record whose target holds no element now refuses the
write: it has no tag, so the archive cannot tell which record the others
belong to. The tag-law waivers `initial-stress-replay` and
`staged-replay-params` become `pre-2-27-initial-stress-replay` and
`pre-2-27-staged-replay-params`, scoped to pre-2.27 archives.
