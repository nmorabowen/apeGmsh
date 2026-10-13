### CHANGED — `ops.py` / `ops.tcl(run=True)` return a `RunReport`; an aborted Tcl deck raises (R3-S6, #1790; ADR 0106 Amendment 1)

**Changed (behaviour):** `ops.py(path, run=True)` and `ops.tcl(path, run=True)`
now always return a frozen `RunReport` instead of `RunSolverStats | None`. A
`-stats` caller reads the record from `rep.solver_stats`; it is `None` when the
deck never asked for `-stats`. `run=False` still returns `None`, and
`run_remote` still returns its `Job`.

- **The fields.** `returncode`, `log` (the absolute path of the tee'd log),
  `elapsed_s`, `increments` and `solver_stats`. `increments` is the analyze
  increments the run completed, summed over every analyze loop of the deck
  (the flat `analyze_steps` drive and each stage). It is read from the deck's
  own `APEGMSH_PROGRESS` markers, which each loop prints on its last increment.
  It is `None` when the count is not established: the deck was emitted with
  `progress=False`, or its markers do not form whole loops.
- **Changed (behaviour): an aborted Tcl deck now raises.** `OpenSees.exe` exits
  0 after a Tcl `error`, the deck's aborted-analyze guard included, so
  `ops.tcl(run=True)` used to return normally from a run that stopped part
  way, and only `ops.results` refused it. A run that exits 0 without the
  driver's `APEGMSH_RUN_COMPLETE` line now raises the same `RuntimeError` (log
  tail plus log path) as a non-zero exit. The py route is unchanged: a failed
  py deck already exits non-zero.
- `ops.results(path)` reads a finished run as before.
