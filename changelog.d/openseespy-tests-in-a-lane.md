### TESTS — every openseespy test now runs in a CI lane, and the viewer's fork tests run in live-fork

Some tests called `pytest.importorskip("openseespy…")` with no `live`
mark. `suite` installs no openseespy, so they skipped there, and
`live-stock` selects `-m live`, so it never ran them. They passed only on a
developer machine. Each one is now `live`, or `ladruno_fork` when it needs
the fork. A skip that was stale is gone, so its tests now run in `suite`:
the demo, reader-parity, every-door and v1.1-B controller modules and the
shared composed-artifacts fixture.

Five fork tests (mortar tie, the Monitor recorder, the tie-force query, the
BezierTri6 run gate) probed fork features through `openseespy` by name.
Beside a fork build that is the stock binding, a second domain, so they
skipped even on the fork. They now take the module from `get_ops()`.
`tests/results/test_monitor.py` called `importorskip` at module level
halfway down the file, which also skipped the pure-Python `tail_monitor`
tests above it. The call is now inside its one live test.

A new quirk rule, `openseespy-unmarked`, flags an openseespy `importorskip`
that can skip a test no `live`, `ladruno_fork` or `subprocess` mark covers.
The nightly `live-fork` workflow also runs the viewer's `npm test` with
`APEGMSH_OPENSEES` set to the fork's OpenSees, so the
`apeGmshViewer/test/materialTest` tests run, and it fails if any of them
skips. `suite` prints skip reasons (`-rs`).
