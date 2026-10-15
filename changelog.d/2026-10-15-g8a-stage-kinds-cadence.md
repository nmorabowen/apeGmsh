### CHANGED — MPCO stage kinds from the archive, stage-aware `.out` cache, one-cadence warning (#1948, G8a)

`Results.from_mpco(..., model_h5=)` now gives each stage the kind its program
declared. An `.mpco` does not tag its stages, so every stage used to read
`transient`. The kind comes from the archive's own analysis record
(`/opensees/stages/stage_NNN/analysis@analysis`): `Static` is `static`, and
`Transient` and `VariableTransient` are `transient`. Any other value, or a
stage with no record, raises. A stage's `analyze_dt` does not decide it,
because a `Static` stage may carry `dt` equal to its `dlam`.

`Results.from_recorders` now includes the stage name and kind in its cache
key. Before, a second read of the same files with another `stage_kind=` or
`stage_name=` returned the first read's file, with the first read's label.

`spec.emit_mpco` and the MPCO Tcl and Python emitters take one `-T` cadence
per recorder: the smallest `dt=`, else the smallest `n_steps=`. When the
records ask for different cadences, they now warn once and name the cadence
used, for example `-T dt 0.01`. Sampling at arbitrary instants has no
recorder equivalent, and the `from_recorders` and `emit_mpco` docstrings now
say so.

The fork-lane test `tests/opensees/integration_ladruno/test_ladruno_stage_kinds_live.py`
runs a gravity stage, then a Newmark stage, and checks the kinds and the
last-step displacements against `ops.nodeDisp` at `rtol=1e-10`. The `.ladruno`
transient kind is a strict xfail on #2034, because the fork stamps one KIND
per recorder, not per stage.
