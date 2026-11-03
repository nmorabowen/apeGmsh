### CHANGED — `EigenResult.results()` writes the modes as `.ladruno`, and `ParallelModalResult.results(model_h5)` opens a distributed run's modes the same way (#2762, ADR 0131 D10 stage 0)

`ops.eigen(n).results()` used to write a native results file through
`to_native` and open it with `Results.from_native`. It now writes a
`.ladruno` through `to_ladruno`, into a fresh temp directory beside its
`<stem>.h5` model archive, and opens it with
`Results.from_ladruno(path, model_h5=...)`. `results(model_h5)` binds the
archive you name instead of that one. `.modes` still lists one mode-scoped
`Results` per eigenvalue, and the non-positive-eigenvalue warning still
names your line.

`ParallelModalResult` gains `results(model_h5)`, the same route for a
`modal_deck` harvest: it writes a temp `.ladruno` through `to_ladruno`,
stamped against `model_h5`, and opens it bound to that archive.

Files written by either route pass the fork's own `ladruno_format.validate()`
at `FORK_PIN`. `to_native` is unchanged and stays the only writer of
native modes files. `apeGmsh.results.live.capture_live_modes` still writes
native modes; it follows in a later slice.
