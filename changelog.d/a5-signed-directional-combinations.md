### ADDED — signed and directional RSA combinations, per-mode drifts and storey forces from modal inertia (Dynamics program A5)

`spectrum_after_gravity` results now combine any signed linear quantity,
not only an extractor's own values, and give the storey forces a design
check needs. Every combination still works on one quantity's per-mode
values; none is taken from combined displacements.

- **Any signed per-mode quantity.** `SpectrumModalResult.linear(name,
  weights)` forms a linear quantity per mode (a sum of reactions, a
  difference of displacements, a storey force from element forces), and
  `combine_values(per_mode, rule=, damping=)` combines it by CQC or SRSS
  with the modes' own frequencies. `combine_modes` keeps its signature;
  its correlation is the fork's `rhoCQC` formula, and the hook for the
  rigid term of A3-5 (Sa-period overrides summed into one mode before the
  rule) is in place but not implemented.
- **Directional combinations.** `combine_directions` takes per-mode sets
  (`rule=`, `omega=`, `damping=`, shape `(n_directions, n_modes, …)`) and,
  with `permutations=True`, returns a `DirectionalCases` with every signed
  100/30 case (`n · 2^n` cases; 8 for two directions), whose `envelope` is
  the scalar 100/30 value. Existing calls return the same values. CQC3 is
  not offered.
- **Per-mode drifts.** `SpectrumModalResult.drifts(name, pairs, dof=,
  heights=)` takes each joint pair's drift per mode from a `NodeDisp`
  extractor. Combine it with `combine_values`. Differencing two combined
  displacements is not offered.
- **Storey forces from modal inertia.** The result carries each mode's
  inertia force `m φ Γ Sa` at every massed node (`inertia`,
  `inertia_nodes`, `inertia_coords`). `storey_forces(elevations,
  vertical=, about=)` sums the shear and the overturning moment above each
  cut, rotational inertia included, as a `StoreyForces` of per-mode
  values.
- **Inertia / base-reaction check.** `inertia_balance()` compares each
  mode's total inertia with its total base reaction. `spectrum_after_gravity`
  warns `InertiaBalanceWarning` above `1e-3`, and `storey_forces` (and
  `check_inertia_balance()`) raises there.
- **Cancellation diagnostics.** `modal_cancellation(per_mode, …)` and
  `SpectrumModalResult.cancellation(name, index)` return a
  `ModalCancellation` with each mode's signed term, its signed share of
  the combined value, the modes' principal-axis angles (from the
  participation factors) and the ratio `combined / Σ|terms|`. Nothing is
  printed.

The storey forces are computed in Python from the eigenvectors the
per-mode loop already reads, on stock OpenSees and on the fork alike. The
fork's `responseSpectrumPerMode` (P1) is not used here.
