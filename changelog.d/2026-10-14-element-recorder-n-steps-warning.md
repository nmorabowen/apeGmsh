### FIXED — element recorder records with `n_steps=` and no `dt=` now warn (#1947)

A classic `.out` recorder has no step-count option, so a record with
`n_steps=` and no `dt=` is emitted as an every-step recorder. Only nodes
records said so; an `elements`, `gauss` or `line_stations` record silently
recorded every step. All four categories now raise the same `UserWarning`,
once per record, on `to_tcl_commands()`, `to_python_commands()` and the live
`begin_stage`. The message names `dt=` for a time cadence on `.out`, and
MPCO's or Ladruno's `-T nsteps` for a step cadence. The emitted recorder
commands are unchanged.
