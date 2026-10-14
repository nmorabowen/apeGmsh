### ADDED — `--phases` attributes the pre-block emit transient (program slice S3-c, #1920)

`tests/benchmarks/emit_throughput_profile.py --phases` now also reports the
window before the first rank block opens: `build()` plus the global preamble
of `BuiltModel.emit`, where the #1905 re-measure found the process emit peak.
It prints the exact traced peak of each sub-window, then splits the snapshot
nearest that peak into a transient part (freed before the first
`partition_open`) and a held part, and the resident set at the first open
into held plus allocated-after-the-peak. Each part is listed by allocating
`apeGmsh/<path>::<qualname>:<line>` site in bytes per global node. The
snapshots are taken from a main-thread return hook, so the peak counter is
never reset behind the main thread. The coverage of the peak by the snapshot
is printed with it. Measurement only: nothing under `src/` changed.
