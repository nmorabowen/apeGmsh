### FIXED — a `consistent_tan` contact refuses a symmetric solver at build time (#1273, B4-d #1629)

`g.constraints.contact(..., consistent_tan=True)` and `edge_consistent_tan=True`
emit the fork's non-symmetric consistent friction tangent. A deck that solved it
on `ProfileSPD` (or any half-storage / diagonal system, a symmetric `Pardiso` or
`Mumps`, or no `system` at all — the OpenSees default is ProfileSPD) built
silently and returned a wrong solve with rc 0. `BuiltModel.emit` now runs
`validate_consistent_tan_solver` (new `opensees/_internal/contact_solver_gate.py`)
beside the u-p gate, with the same scope rules: fail-loud `BridgeError` naming
the allowed solvers (`UmfPack`, `SparseGeneral`, `FullGeneral`, `BandGeneral`,
unsymmetric `Pardiso` / `Mumps`), a declared symmetric system refused on every
emit, a missing system refused only on a deck that analyses, the partitioned
no-system deck allowed (ADR 0027 auto-emits a general solver), and each stage
checked on its own. An archival `apeSees.h5` records the verdict as
`consistent_tan_solver` in `@solve_refusals`. Decks without the flag are
unchanged.
