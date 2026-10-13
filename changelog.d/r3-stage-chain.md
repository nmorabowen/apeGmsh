### ADDED — `ops.analysis.chain`: one analysis chain shared by stages (R3-S1, #1785)

`solver = ops.analysis.chain(test=, algorithm=, constraints=, numberer=,
system=, analysis=)` returns a frozen value object (not a primitive: it
registers and emits nothing). A stage takes it as
`s.analysis(solver, integrator=...)`; the emitted deck is byte-identical to
the keyword form, which keeps working. A chain plus a keyword for a slot it
already holds is refused.

A stage using `ops.algorithm.Linear()` may now omit its convergence test: the
stage emits no `test` line, and the H5 archive stores and replays none. Each
algorithm declares `needs_test` (False only for `Linear`). The test stays
required for every iterating algorithm and for a stage with
`s.run(strategy=)`.

**Added:** `InertStaticDtWarning` (a `UserWarning`, exported from
`apeGmsh.opensees`). A `Static` stage driven by `LoadControl` (or
`LadrunoLoadControl`) whose `s.run(dt=)` differs from the integrator's `dlam`
warns when the stage closes (maintainer ruling D1): "dt has no effect under
Static; dlam sets the step", with the stage name, `dt` and `dlam`. OpenSees
ignores `dt` under `Static`, so such a `dt` never acted. It is silent when
`dt` is omitted or equals `dlam`, and under `Transient`. No deck changes.
