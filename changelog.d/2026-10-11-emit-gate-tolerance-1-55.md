### CHANGED — emit-cost gate tolerance tightened 1.7 -> 1.55 (#1677)

`RATIO_TOLERANCE` in `tests/benchmarks/test_emit_regression_gate.py` drops to 1.55, so the hex cell catches the #876 shape with about 10% margin while the worst observed EPYC 7763 run keeps about 20% headroom. A deterministic test pins both sides, and the docstring now says a `write_emit_gate_baseline` dispatch always passes against the file it just wrote.
