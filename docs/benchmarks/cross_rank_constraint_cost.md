# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: 2026-06-25 05:12:07 UTC

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1500.0`

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | 1_014 | 0.005 | 0.007 | 381.5 | — |
| 100 | 2 | hex_host_line_embed | 1_414 | 0.004 | 0.009 | 386.6 | — |
| 100 | 4 | tet_host_line_embed | 1_020 | 0.004 | 0.007 | 386.6 | — |
| 100 | 4 | hex_host_line_embed | 1_420 | 0.005 | 0.009 | 386.6 | — |
| 100 | 8 | tet_host_line_embed | 1_032 | 0.004 | 0.006 | 386.6 | — |
| 100 | 8 | hex_host_line_embed | 1_432 | 0.005 | 0.008 | 386.8 | — |
| 1_000 | 2 | tet_host_line_embed | 10_014 | 0.026 | 0.070 | 446.6 | — |
| 1_000 | 2 | hex_host_line_embed | 14_014 | 0.034 | 0.101 | 477.6 | — |
| 1_000 | 4 | tet_host_line_embed | 10_020 | 0.028 | 0.070 | 477.6 | — |
| 1_000 | 4 | hex_host_line_embed | 14_020 | 0.036 | 0.099 | 482.8 | — |
| 1_000 | 8 | tet_host_line_embed | 10_032 | 0.031 | 0.071 | 482.8 | — |
| 1_000 | 8 | hex_host_line_embed | 14_032 | 0.039 | 0.101 | 487.8 | — |
| 10_000 | 2 | tet_host_line_embed | 100_014 | 0.348 | 0.784 | 1062.0 | — |
| 10_000 | 2 | hex_host_line_embed | 140_014 | 0.444 | 1.093 | 1331.7 | — |
| 10_000 | 4 | tet_host_line_embed | 100_020 | 0.275 | 0.778 | 1331.7 | PASS |
| 10_000 | 4 | hex_host_line_embed | 140_020 | 0.373 | 1.089 | 1343.8 | PASS |
| 10_000 | 8 | tet_host_line_embed | 100_032 | 0.403 | 0.772 | 1343.8 | — |
| 10_000 | 8 | hex_host_line_embed | 140_032 | 0.499 | 1.105 | 1343.8 | — |
| 100_000 | 2 | tet_host_line_embed | 1_000_014 | 2.943 | 8.448 | 7081.9 | — |
| 100_000 | 2 | hex_host_line_embed | 1_400_014 | 4.215 | 12.335 | 9825.7 | — |
| 100_000 | 4 | tet_host_line_embed | 1_000_020 | 3.233 | 8.108 | 9825.7 | — |
| 100_000 | 4 | hex_host_line_embed | 1_400_020 | 4.453 | 11.858 | 9828.4 | — |
| 100_000 | 8 | tet_host_line_embed | 1_000_032 | 3.555 | 8.163 | 9828.4 | — |
| 100_000 | 8 | hex_host_line_embed | 1_400_032 | 4.859 | 11.925 | 9833.4 | — |

## Decision gate status

- `deck_emit_sec`     pass: **True**
- `deck_parse_py_sec` pass: **True**
- `deck_lines`        pass: **True**
- `peak_rss_mb`       pass: **True**

**Overall: PASS** — proceed to Phase 2 (full feature).
