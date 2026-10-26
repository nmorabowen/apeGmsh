### FIXED — an in-process re-run no longer inherits the previous run's class-wide material stage (#2463)

The stage flags of `ManzariDafalias` (with `LadrunoSANISAND`),
`SAniSandMS`, `PM4Sand` and `PM4Silt` are C++ statics, so they survive
`ops.wipe()`. A second model run in the same process started at whatever
stage the first one left. After a run that flipped `PM4Sand` to plastic,
for example, the next elastic gravity stage failed to converge. Each deck
target and the in-process run now state every group's **fresh stage**,
the value it holds in a new process, at the first solve after the
elements (#2440 did this for the `ManzariDafalias` group only):

- `ManzariDafalias` and `LadrunoSANISAND`: `-stage 0`, as since #2440.
- `PM4Sand` and `PM4Silt`: `-stage 0` (elastic).
- `SAniSandMS`: `-stage 1`. It starts plastic on stock 3.8.0 and on the
  fork alike, so a fresh run gives the same answer as before.

A user flip still wins, one group at a time: once an
`s.update_material_stage` or a stage-flag `s.update_parameter` names a
group, that group gets no more lines and the others keep theirs.
`model.h5` stores none of the lines, so `model_hash` is unchanged, and a
replayed deck writes them back.
