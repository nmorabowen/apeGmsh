### ADDED — retirement ledger with a dated CI expiry (X4-a, #1806)

Every retiring thing (V4 KEEP items, known gaps, the grandfathered fork
members, the studio spin-out, the prototypes folder, deprecated aliases,
shadow paths, the xfail ledger) now has a dated row in
`internal_docs/program/retirement_ledger.toml`. `tests/test_retirement_ledger.py`
runs in `lock-tests` and fails when a row's date has passed while its target
still exists, when a target has vanished but its row stayed, or when a row is
malformed or undated. The ledger is a `land_pr.py` baseline file, so editing it
needs the `human-gate` label. ADR 0116 gains a Proposed amendment: trame and
`show_web`/`serve_web` leave when V4 covers interactive results.
