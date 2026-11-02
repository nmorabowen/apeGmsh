### Spectrum terms of a combination read `/spectra` (ADR 0130 slice 2b)

`Results.combinations()` now reads `spectrum` and `reference` terms from the set's
`.user.h5` `/spectra` through `UserSpectra`, so a `cqc`, `srss` or `abs_sum` over a
case, a signed increment, the reference state, and the `directional` and
`envelope` combinations built on them evaluate with no provider passed. A nodal
component reads the one case column whose recorder spelling the readers translate
to it (`DISPLACEMENT` / `Uy` is `displacement_y`). `UserSpectra` also reads a
domain result by its label (`("domain", "base_shear")`, every column), though
`evaluate()` still takes only the recorded levels. A quantity the case does not
hold, an element level, or a missing case is refused `unresolved`; a case the
reader refused is `refused`. A `/spectra` zone whose model hashes differ, or
that has no model to compare with, is refused `stale` unless the provider is
made with `allow_stale=True` (A6).
