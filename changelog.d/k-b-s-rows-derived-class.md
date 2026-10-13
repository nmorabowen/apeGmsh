### CHANGED — R1b oracle: five FEMStub-built rows replay on a real FEMData (#1885, #1797)

Tests only. `test_h5_apesees_cuts::_build_minimal_ops` and the
`test_h5_decls` two-quad models (`_staged`, `_param_stage_model`, and
`test_h5_decl_params::_broad_staged`, which builds on `_staged`) now build on a
real `FEMData`, so their archives carry a neutral zone and replay. Their five
class-S rows in the round-trip ledger re-pin to the class the oracle derives:
three A (integral floats only) and two B (declaration order). The oracle has a
new transition rule for this, approved by the maintainer on #1797. An S row may
re-pin to A or B, never C, and only when its reason cites the derived evidence as
`derived <class> (#1885): <what the diff is>`. A row without that citation is
refused. `N_WAIVED_ORACLE` is unchanged.
The cuts fixture's 3-D fiber section now carries `GJ`, because stock OpenSees
refuses one without it, and the live net runs every B row's deck.
