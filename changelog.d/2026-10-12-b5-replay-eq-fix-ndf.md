### FIXED — deck replay keeps `equationConstraint` rows and per-node `fix` widths (B5-g, #1798)

`OpenSeesModel.from_h5(...).build("tcl" | "py" | "live")` used to drop every
`apeSees.equation_constraint` row, and gave no warning when it did.

- **Equation constraints.** The archive already stores each row in
  `/opensees/decl_params`. Replay now reads those rows and emits them again after
  the regions, which is where the bridge emits them, on flat and staged archives
  alike. It runs the bridge's own node and DOF checks.
  - If a declaration has no readable row, replay raises `MalformedH5Error`.
  - If the archive has no `/opensees/decls` at all (older than opensees 2.25.0),
    replay still runs, but raises `H5LedgerWarning` to say the rows cannot be
    checked.
- **Fix widths.** Replay used to pad a mixed-ndf node's `fix` to the envelope ndf,
  so `fix 1 1 1` came back as `fix 1 1 1 0`. OpenSees accepts both forms, so
  results did not change.
  - `from_h5` now trims each global and per-stage row to the node's ndf in
    `/opensees/nodes_ndf`.
  - It raises when that map has no entry for a fixed node, or when a non-zero
    flag lies past the node's own DOFs.

In the R1b oracle, `_ledgered_frame` now round-trips and its row is gone
(`N_WAIVED_ORACLE` 36 → 35). Three mixed-ndf rows drop from class C to B.
`_broad_frame` stays C and now cites #1814: its replayed deck declares a
`Parallel` material before the materials it combines.

The oracle now derives each row's class. It enforces the re-pin carve-out
against the ledger at the merge base with `origin/main`: a row may be deleted,
or C may drop to A or B, and nothing else.
