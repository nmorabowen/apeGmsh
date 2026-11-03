### CHANGED — `EigenResult.results()` writes the modes as `.ladruno` (#2762, ADR 0131 D10 stage 0)

`ops.eigen(n).results()` used to write a native results file through
`to_native` and open it with `Results.from_native`. It now writes a
`.ladruno` through `to_ladruno`, into a fresh temp directory beside its
`<stem>.h5` model archive, and opens it with
`Results.from_ladruno(path, model_h5=...)`. `results(model_h5)` binds the
archive you name instead of that one. `.modes` still lists one mode-scoped
`Results` per eigenvalue, and the non-positive-eigenvalue warning still
names your line.

Files written this way pass the fork's own `ladruno_format.validate()` at
`FORK_PIN`. `to_native` is unchanged and stays the only writer of native
modes files. `ParallelModalResult` keeps `to_ladruno` and gains no
`results()`. `apeGmsh.results.live.capture_live_modes` still writes native
modes; it follows in a later slice.
