### CHANGED — `InertStaticDtWarning` covers every `Static` integrator (R3-S1b, #1822)

Under `Static`, OpenSees ignores `analyze`'s `dt`, so a stage's
`s.run(dt=)` never acts. `LoadControl` and `LadrunoLoadControl` keep the
R3-S1 rule: a `dt` that differs from `dlam` warns. Every other static
integrator (`DisplacementControl`, `ArcLength`, `LadrunoArcLength`,
`LadrunoIndirectControl`, `LadrunoDynamicRelaxation`) has no `dlam` to match,
so it now warns whenever `dt` is given, and the message says so. Each
integrator declares the rule as a `static_step_field` ClassVar (`"dlam"` or
`None`); an integrator that does not declare it warns on any `dt`. Omitting
`dt`, and every `Transient` stage, stay silent. No deck changes.
