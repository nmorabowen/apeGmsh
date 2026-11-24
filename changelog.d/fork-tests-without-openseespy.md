### TESTS — fork tests no longer sit behind an openseespy skip, and live-fork can run py decks

Fourteen `ladruno_fork` tests in seven files were gated by
`pytest.importorskip("openseespy…")`. live-fork installs the fork as
`opensees` and has no openseespy, so the skip fired there. live-stock skips
the same tests as not-fork, so they ran in no lane. The gates now check for
any backend (`backend_available()`), or are gone where the conftest
`ladruno_fork` hook already resolves the fork.

Three of these tests also read results from `openseespy` by name. Beside
a fork build that name is the stock binding, a second, empty domain, so
they failed on a fork machine: the two augment-sweep tests and the live
equation-tie solve. They now read the module the bridge drives
(`get_ops()`, `LiveOpsEmitter._ops`).

An emitted py deck runs `import openseespy.opensees`, so live-fork now
installs a small `openseespy` shim that binds that name to the fork module,
and checks that it does. The `.ladruno` route test runs its deck there.
Elsewhere it skips with a reason when the deck's interpreter does not bind
the fork. A new quirk rule, `openseespy-gates-fork`, flags an openseespy
`importorskip` that can skip a `ladruno_fork` test. An `autouse` fixture now
counts as reaching every test of its module, for this rule and for
`openseespy-unmarked`.
