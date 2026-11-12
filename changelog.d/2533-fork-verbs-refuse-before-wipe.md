### FIXED — fork-only verbs refuse by name on stock before the wipe (#2533)

#2449 made every in-process path refuse a fork-only *chain* before it wiped
the live domain. A fork-only *verb* run with a stock chain still passed that
check: it wiped the domain, emitted the model, and only then failed at the
fork command, so the user's domain was gone. On a stock build these now
refuse by name, before the wipe, with the same `has_fork` verdict
`ops.eigen_feast()` already read:

- `ops.complex_eigen()`, `ops.modal_response_history()` and
  `ops.response_spectrum_analysis()`.
- `ops.frequency_response()`, `ops.steady_state_dynamics()` and
  `ops.random_response()`.
- `ops.critical_time_step()` and `ops.analyze_explicit()`, which used to
  fail with a bare `AttributeError: criticalTimeStep`.
- `ops.analyze(profile=...)`, which wiped and ran the emit before the
  profiler gate refused.

The message reads `apeSees.<verb> requires the Ladruno fork build of
OpenSees (...)`. Argument errors still come first, and the fork skips the
check. The verdict is the `ladrunoBuild` stamp, so a fork built before that
stamp (fork PR #718) is now refused by name too, where some of these verbs
(`analyze(profile=...)`, for one) used to run on it; the message carries the
same "rebuild the fork" note as the chain refusal. `ops.run()` on a staged model now raises `NotImplementedError`
before the wipe, as `ops.eigen()` already did; it used to wipe and then
fail at the first stage. A source-scan test fails on any wiping site that
calls a fork-only `LiveOpsEmitter` method without the check.
