### FIXED — nightly Benchmarks: stale ADR 0100 R1 anchor after X1-b (#1650)

`tests/benchmarks/emit_throughput_profile.py` dropped its `node_idx = node_index_lookup(` R1 anchor, left behind when X1-b (#1505) removed the `split=` mode, and re-verified every other R-term anchor against `apesees.py`/`build.py`. The `test_emit_profile_smoke.py` bench lane is green again.
