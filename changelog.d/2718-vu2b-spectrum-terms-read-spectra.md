### Spectrum terms of a combination read `/spectra` (ADR 0130 slice 2b)

`Results.combinations()` now reads `spectrum` and `reference` terms from the
set's `.user.h5` `/spectra` through `UserSpectra`. With no provider passed, these
now evaluate: a `cqc`, `srss` or `abs_sum` over a case, a signed increment, the
reference state, and the `directional` and `envelope` combinations built on
them. A nodal component reads the one case column whose recorder spelling the
readers translate to it (`DISPLACEMENT` / `Uy` is `displacement_y`).
`ids=` selects the case's rows as it selects a stage's, and an id the case does
not hold is refused. `evaluate(state, "base_shear", level="domain")` reads a
domain result by its label, every column; `ids=` there raises. Its `stage` and `mode` terms are refused `unresolved`,
because `Results` records no domain value. A quantity the case does not hold,
an element level, or a missing case is refused `unresolved`, and a case the
reader refused is `refused`. A `/spectra` zone whose model hashes differ, or
that has no model to compare with, is refused `stale` unless
`evaluate(..., allow_stale=True)` (A6).
