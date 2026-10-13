### FIXED — `ops.h5` no longer warns that `apeSees.equation_constraint` rows are dropped (#1805)

Since #1801 each `apeSees.equation_constraint` row is archived as an
`EquationConstraintRecord` row of `/opensees/decl_params`, and
`OpenSeesModel.from_h5(...)` replays it. Even so, `ops.h5` still raised
`H5FeatureDeferredWarning` ("NOT archived"), and the automatic write
counted the row in `H5LedgerWarning`. Both warnings were false. Neither
fires for these rows now. `H5LedgerWarning` still counts the
`equationConstraint` rows of an `enforce="equation"` interface tie,
because deck replay skips those.
