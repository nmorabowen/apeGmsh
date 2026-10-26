### ADDED — `spectrum_after_gravity`: per-mode response spectrum about the gravity state, combined per quantity (ADR 0124 D4–D7)

`apeSees.spectrum_after_gravity(num_modes, *, spectra, extractors, p_delta=True, solver=...)` runs the
`modal_after_gravity` sequence and then, on the same domain, applies each mode's probe force
`± a_k M φ_k` as a nodal pattern, takes one `Linear` step on the tangent the modes came from (so each
leg moves the domain by exactly `± a_k φ_k / λ_k`, the state `responseSpectrumAnalysis -mode k` imposes,
PDelta included), calls every response extractor on the committed state, and solves back to the gravity
state with the declared algorithm. The probe moves the largest nodal translation by `1e-8` of the model
extent whatever the spectrum; modes with negligible effective mass are skipped; a mode whose two legs
meet different stiffnesses (second difference of the resisting forces above `1e-3` of the first: a
no-tension brace at zero strain, `Concrete01` on its backbone) is refused, and so is a model with
declared recorders (every leg commits). The per-mode value is the
central difference about the gravity state scaled by `Γ_k Sa(T_k)` for each `DesignSpectrum`: a
`(T, Sa)` table, any object with a vectorised `.at(T)` (a codeSpectra `Spectrum`, never imported) or a
vectorised callable, plus a g-to-model-units scale and a direction (the scale is applied in Python). An extractor is any `extractor(ops) -> 1-D array` of fixed length,
optionally with `.labels`; `NodeDisp`, `NodeReaction` and `ElementResponse` are built in, and the
driver adds `"base_shear"` (total support reaction per direction) so `SpectrumModalResult.base_shear`
gives `V_t` per direction, and `scale_to_base_shear(v_target, names)` lifts the named combined forces
by `max(1, v_target/V_t)` per direction (code-agnostic: ASCE 7-22 §12.9.1.4.1, NEC, NCh targets are the
caller's; drifts are not scaled). `.combine` gives CQC (Der Kiureghian `ρ_ij`, one ratio or one per mode) or
SRSS per quantity, `.directional` 100/30 or SRSS over the directions, and `seismic_envelope` the
`S ± f·E` envelope for a gravity combination; `gravity_values` keeps each extractor at the gravity
state. `responseSpectrumAnalysis -mode k` is not used: it replaces the committed displacement with the
modal one and commits, so the gravity state is lost. Element mass is refused (the modal force is built
from nodal masses; `g.masses` / `mass_from_model` is the supported route), and the driver checks the
domain returns to the gravity state. On a linear frame the per-mode displacements, reactions and column
forces and their CQC meet a numpy reference to `4e-13`, the Python CQC meets the fork's `-combine CQC`
to `3e-13` (`4e-14` on the PDelta frame), and every mode's base shear is `M_eff Sa` to `4e-16`; runs on
stock and fork.
