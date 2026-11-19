### CHANGED — docs on the inertia balance and the rigid term's tolerance (chain A link A3 follow-up)

No behaviour changes. `SpectrumModalResult.inertia_balance` now says that the
balance holds by construction for residual and Ritz modes, whose force is read
off the same leg as the reactions, and stays a real check for eigenmodes.

`combined_period_tolerance(sa_periods=)` keeps rescaling overridden modes by
`(T_ref/T)^p`, and its docstring says why. The group's displacement is the
static residual `K⁻¹ M r Sa(T)`, which scales by exactly `(T_ref/T)²` when two
models share their shapes, while its force does not scale. A new test checks
this against an independent solve of the podium with a stiffer `K`.
