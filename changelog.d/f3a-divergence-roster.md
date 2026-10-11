### ADDED — fork divergence roster with a per-row oracle, and the fork-option effect-test rule (program F3-a, #1666)

`architecture/fork-divergence-roster.md` lists every known place where
stock OpenSees and the Ladruno fork give a different answer: tet10 1/6
stiffness, the DruckerPrager tension cutoff, the ASDPlastic apex, the LAPACK
zero pivot reported as success, `-hall` tangents under BackwardEuler,
Newmark1 and Collocation, fork-only integrators accepted silently by stock,
the fork's three unknown-token policies, the `-crackedNu`/`-betaC` flags,
and the SANISAND cross-compiler difference (#1523, pending). Each row names
the correct engine, the oracle that decides it, and the tests that pin it.
`tests/opensees/unit/test_fork_divergence_roster.py` locks the table: a
cited test that is renamed or deleted, a blank correct-engine cell, or a new
unpinned row fails the suite.

`architecture/testing.md` gains the rule for fork-option effect tests: a
fork-only option apeGmsh emits ships with a live test that asserts its
effect, not just its acceptance.
