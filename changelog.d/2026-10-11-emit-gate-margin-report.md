### Emit-cost gate reports its margin on every run (#1658)

`test_emit_regression_gate.py` now prints one `EMIT-GATE cell=... ratio=... ceiling=... baseline=... margin=... headroom=... emit=... parse=...` line per cell on a pass as well as a fail, and appends it to `$GITHUB_STEP_SUMMARY` when set. The nightly `Benchmarks` step runs with `-s` so the lines reach its log. Thresholds, baselines and pass/fail logic are unchanged.
