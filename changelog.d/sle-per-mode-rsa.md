### ADDED — `spectrum_after_gravity`: per-mode response spectrum about the gravity state, combined per quantity (ADR 0124 D4–D7)

`apeSees.spectrum_after_gravity(num_modes, *, spectra, extractors, p_delta=True, solver=...)` runs the
`modal_after_gravity` sequence and then, on the same domain, applies each mode's inertia force
`± c M φ_k` as a nodal pattern, solves with the declared algorithm at `dλ = 0`, calls every response
extractor on the committed state, and solves back to the gravity state. The per-mode value is the
central difference about the gravity state scaled by `Γ_k Sa(T_k)` for each `DesignSpectrum`: a
`(T, Sa)` table or a vectorised callable `Sa(T)`, plus a g-to-model-units scale and a direction (the
scale is applied in Python). An extractor is any `extractor(ops) -> 1-D array` of fixed length,
optionally with `.labels`; `NodeDisp`, `NodeReaction` and `ElementResponse` are built in, and the
driver adds `"base_shear"` (total support reaction per direction) so `SpectrumModalResult.base_shear`
gives `V_t` per direction. `.combine` gives CQC (Der Kiureghian `ρ_ij`, one ratio or one per mode) or
SRSS per quantity, `.directional` 100/30 or SRSS over the directions, and `seismic_envelope` the
`S ± f·E` envelope for a gravity combination; `gravity_values` keeps each extractor at the gravity
state. `responseSpectrumAnalysis -mode k` is not used: it replaces the committed displacement with the
modal one and commits, so the gravity state is lost. Element mass is refused (the modal force is built
from nodal masses; `g.masses` / `mass_from_model` is the supported route), and the driver checks the
domain returns to the gravity state. On a linear frame the per-mode displacements, reactions and column
forces and their CQC meet a numpy reference to `4e-13`, the Python CQC meets the fork's `-combine CQC`
to `3e-13`, and every mode's base shear is `M_eff Sa` to `4e-16`; runs on stock and fork.
