### CHANGED — `Results.from_recorders` transcodes nodal `.out` records over the program's stages to a `.ladruno`; a Python-written `.ladruno` no longer claims the fork's engine (#2786, ADR 0131 D6)

`Results.from_recorders` used to transcode every recorder run into a native
`results.h5`. It now writes `<key>.ladruno` in the cache when both of these hold:

- every record is a nodal record (displacement, rotation, velocity,
  acceleration and their angular forms, reactions, unbalanced forces, pore
  pressure);
- the read's stages (`stage_id=`, `stage_name=`/`stage_kind=`,
  `stages=live.stages`) are the first of the model's `ops.stage` blocks, with
  the same names and kinds in the same order.

The file holds one `MODEL_STAGE` per stage, each with its own nodes, kind and
time. It is stamped against the model archive written beside it
(`<key>.model.h5`) and opened with `Results.from_ladruno(model_h5=)`.
`from_recorders` first checks that the stages it binds are the read's. A
`.ladruno` stores no stage names; every reader, the viewer included, pairs the
program's names onto its stages by position. The rule above is what makes that
pairing right. Every stage, time, node id and value equals the old native
read, including the NaN of a node that a record does not hold.

Every other read keeps the native file, which stores its names. That covers:

- a reordered plan, a single stage that is not the program's first, or a
  stage of another kind;
- a program with no `ops.stage`;
- the default route's classic-recorder branch over a FEMData archive;
- any element record (Gauss points, line stations, nodal forces, fibres),
  until W4b;
- `displacement_increment`, which the format has no group for.

No record is dropped. A `.ladruno` target handed such a spec directly refuses
and names the records. `PARSER_VERSION` is now `1.4`, so a cached native
transcode is made again.

`PROVENANCE@opensees_backend` used to default to `"ladruno"`. Every file that
`write_modes` or `write_nodal_history` writes is now stamped `"unknown"`. That
includes `EigenResult.to_ladruno`, `ParallelModalResult.to_ladruno` and every
recorder transcode. `capture_live_modes` stamps the fork's engine only when the
module it solved on is the fork (it answers `ladrunoBuild`). On stock
openseespy it stamps `"unknown"`, because §7b.1 keeps `"stock"` for a
converted `.mpco`.
