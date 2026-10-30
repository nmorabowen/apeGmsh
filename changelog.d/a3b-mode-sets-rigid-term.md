### ADDED — Ritz and residual mode sets in `modal_after_gravity` / `spectrum_after_gravity`, mode kinds, and the rigid Sa-period term (chain A link A3b)

`modal_after_gravity` and `spectrum_after_gravity` take keyword-only mode-set
options. Without them both behave as before:

- `basis="ritz"` runs the fork's `ritzVectors num_modes` in place of
  `eigen`. Its starting loads are `ritz_dirs` (inertia `M ι_d`) and
  `ritz_patterns` (`ops.pattern.Plain` handles or names; the gravity pattern
  gives a basis with 100 % static participation in it).
- `residual=dirs` appends `residualModes -dir dirs`, so the modal mass of
  each direction reaches 100 %.
- `residual_period=T` adds `-period T`. It needs `residual=` and is refused
  on a Ritz basis, as the fork refuses it.

The order is fixed: basis, residual, then `modalProperties`, with no `eigen`
between them. Each option refuses an MPI model and a build without chain M
before anything is built.

`ModalPropertiesResult.kinds` and `SpectrumModalResult.kinds` give each
mode's kind (`"eigen"`, `"residual"`, `"ritz"`) from the fork's `modeKind`,
and `"eigen"` on a build without mode tags. A Ritz or residual mode's period
is its Rayleigh period.

In `spectrum_after_gravity`, with `residual_period=T` the residual
pseudo-modes read Sa at `T` while keeping their Rayleigh λ, so each value is
`ψ Γ Sa(T) / λ`. `SpectrumModalResult.sa_periods` records the override, and
every combination on the result (`combine`, `combine_values`, `base_shear`,
`scale_to_base_shear`) sums the pseudo-modes into one rigid term at
`ω = 2π/T` before the rule, as the fork's `responseSpectrumAnalysis
-combine` does (fork ADR 213 D3b). `combine_modes` takes the same
`sa_periods=`. Ritz vectors and pseudo-modes are not eigenvectors, so a mode
set holding them needs the `-increment` route, and the static-probe route
refuses it.
