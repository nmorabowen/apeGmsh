### FIXED — every in-process run on stock refuses a fork-only chain before the wipe, not only `analyze` (#2449)

#2440 moved the stock refusal of a fork-only analysis chain ahead of the
wipe for `apeSees.analyze` only. The other in-process paths still wiped the
live domain first and refused at the chain command, so a refused call had
already cleared whatever the user had in the domain. These paths now refuse
before the wipe too:

- `ops.run()`, whether `wipe=True` or `wipe=False`. With `wipe=False` the
  refusal now also comes before the model is emitted into the domain.
- `ops.eigen()`, `ops.modal_properties()`, `ops.eigen_feast()`,
  `ops.complex_eigen()`, `ops.modal_response_history()` and
  `ops.response_spectrum_analysis()`.
- `ops.critical_time_step()` and `ops.analyze_explicit()`.
- `ops.frequency_response()`, `ops.steady_state_dynamics()` and
  `ops.random_response()`.

A fork-only chain is a declared or auto-emitted `system`, `integrator`,
`numberer`, `test` or constraint handler that only the Ladruno fork knows
(`LadrunoProjection`, `LadrunoContact`, `ExplicitBathe`, `Pardiso`, ...).
The refusal and its message are unchanged, and the fork skips the check.
`ops.run()` also keeps the read session of the previous solve when it
refuses. A source-scan test now fails on any new `LiveOpsEmitter`
construction that can wipe without the check before it.
