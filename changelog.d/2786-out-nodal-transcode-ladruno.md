### CHANGED — `Results.from_recorders` transcodes nodal `.out` records to a `.ladruno`; a Python-written `.ladruno` no longer claims the fork's engine (#2786, ADR 0131 D6)

`Results.from_recorders` used to transcode every recorder run into a native
`results.h5`. A spec of nodal records only (displacement, rotation,
velocity, acceleration and their angular forms, reactions, unbalanced
forces, pore pressure) is now written as `<key>.ladruno` in the cache:
one `MODEL_STAGE` per stage, each with its own nodes, kind and time,
stamped against the model archive written beside it (`<key>.model.h5`).
It opens as a `.ladruno`-backed `Results` with the caller's `fem=` and
`model=` bound as given. The stages keep the names and kinds the read gives
them (`stage_id=`, `stage_name=`, `stages=live.stages`). The program's
`ops.stage` names are never paired onto them by position, so a staged
deck's `stage_id="push"` read alone is still named `push`. Every stage,
time, node id and value, the NaN of a node a record does not hold
included, equals the old native read. The default route's classic-recorder
branch over a FEMData archive transcodes the same way and is stamped by
`fem_hash`.

A spec with any element record (Gauss points, line stations, nodal forces,
fibres) keeps the native file until W4b. So does a spec with
`displacement_increment`, which the `.ladruno` format has no group for. No
record is dropped: a `.ladruno` target given such a spec directly refuses,
naming the records. `PARSER_VERSION` is now `1.4`, so a cached native
transcode is made again.

`PROVENANCE@opensees_backend` used to default to `"ladruno"`. Now a file
written by `write_modes` or `write_nodal_history`, and so every
`EigenResult.to_ladruno`, `ParallelModalResult.to_ladruno` and recorder
transcode, is stamped `"unknown"`. `capture_live_modes` stamps the fork's
engine only when the module it solved on is the fork (it answers
`ladrunoBuild`). On stock openseespy it stamps `"unknown"`, because §7b.1
keeps `"stock"` for a converted `.mpco`.
