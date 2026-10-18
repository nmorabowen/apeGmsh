### Fixed

- The nightly `Benchmarks` lane was red from 2026-10-15: the ADR 0100 G0 R7 anchor (`class_chunks` in `infer_node_ndf`) was removed by #1925. R7 is retired in `emit_throughput_profile.py` (it reads 0 by construction); the other R-term anchors are unchanged (#2144).
