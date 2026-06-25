# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: 2026-06-25 05:46:58 UTC

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1500.0`

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | 1_014 | 0.005 | 0.007 | 380.6 | — |
| 100 | 2 | hex_host_line_embed | 1_414 | 0.005 | 0.009 | 385.5 | — |
| 100 | 4 | tet_host_line_embed | 1_020 | 0.004 | 0.006 | 385.5 | — |
| 100 | 4 | hex_host_line_embed | 1_420 | 0.005 | 0.009 | 385.6 | — |
| 100 | 8 | tet_host_line_embed | 1_032 | 0.004 | 0.006 | 385.6 | — |
| 100 | 8 | hex_host_line_embed | 1_432 | 0.005 | 0.009 | 386.6 | — |
| 1_000 | 2 | tet_host_line_embed | 10_014 | 0.027 | 0.075 | 444.4 | — |
| 1_000 | 2 | hex_host_line_embed | 14_014 | 0.036 | 0.103 | 475.7 | — |
| 1_000 | 4 | tet_host_line_embed | 10_020 | 0.028 | 0.071 | 475.7 | — |
| 1_000 | 4 | hex_host_line_embed | 14_020 | 0.036 | 0.103 | 480.7 | — |
| 1_000 | 8 | tet_host_line_embed | 10_032 | 0.031 | 0.071 | 480.7 | — |
| 1_000 | 8 | hex_host_line_embed | 14_032 | 0.041 | 0.102 | 487.1 | — |
| 10_000 | 2 | tet_host_line_embed | 100_014 | 0.366 | 0.788 | 1060.9 | — |
| 10_000 | 2 | hex_host_line_embed | 140_014 | 0.468 | 1.107 | 1328.7 | — |
| 10_000 | 4 | tet_host_line_embed | 100_020 | 0.273 | 0.793 | 1328.7 | PASS |
| 10_000 | 4 | hex_host_line_embed | 140_020 | 0.377 | 1.115 | 1335.8 | PASS |
| 10_000 | 8 | tet_host_line_embed | 100_032 | 0.311 | 0.797 | 1335.8 | — |
| 10_000 | 8 | hex_host_line_embed | 140_032 | 0.533 | 1.104 | 1335.8 | — |
| 100_000 | 2 | tet_host_line_embed | 1_000_014 | 2.967 | 8.060 | 7081.3 | — |
| 100_000 | 2 | hex_host_line_embed | 1_400_014 | 4.243 | 11.458 | 9820.9 | — |
| 100_000 | 4 | tet_host_line_embed | 1_000_020 | 3.184 | 8.252 | 9820.9 | — |
| 100_000 | 4 | hex_host_line_embed | 1_400_020 | 4.738 | 11.707 | 9827.2 | — |
| 100_000 | 8 | tet_host_line_embed | 1_000_032 | 3.506 | 8.235 | 9827.2 | — |
| 100_000 | 8 | hex_host_line_embed | 1_400_032 | 5.097 | 11.642 | 9832.5 | — |

## Decision gate status

- `deck_emit_sec`     pass: **True**
- `deck_parse_py_sec` pass: **True**
- `deck_lines`        pass: **True**
- `peak_rss_mb`       pass: **True**

**Overall: PASS** — proceed to Phase 2 (full feature).
