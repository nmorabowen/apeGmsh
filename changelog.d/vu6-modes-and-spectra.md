### `<stem>.user.h5`: the `/modes` table and the `/spectra` cases (ADR 0130 slice 6)

`apeGmsh.results.user` now writes and reads the modal table and the spectrum cases.
`mode_table(properties, host=...)` turns a `modalProperties -return` dict into the
`/modes` table: periods, participation factors, mass ratios as fractions, and the
generalised mass, stored as effective mass over `Γ²` on the largest `|Γ|`.
`spectrum_cases(result, names, damping=...)` turns an `apeSees.spectrum_after_gravity`
result into one case per spectrum, holding each mode's `λ`, `Γ`, `Sa`, `Sa` period and
damping, the per-mode increments and the reference state's values. Domain quantities
are written by default. A `NodeDisp` field is written as `ON_NODES/DISPLACEMENT` only
when it is named in `nodal=`, and a nodal extractor named neither there nor in `omit=`
is refused. `write_user_file(modes=..., spectra=...)` replaces each zone whole.
`staleness` now checks `/modes` row by row: a row is stale when its `λ` differs from
the `.ladruno`'s by more than `1e-9` relative. The `modes` and `spectra` zones join
the schema registry at 1.0.0.
