# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: 2026-08-05 05:05:23 UTC

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1500.0`

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | 1_014 | 0.007 | 0.006 | 445.2 | — |
| 100 | 2 | hex_host_line_embed | 1_414 | 0.006 | 0.010 | 448.9 | — |
| 100 | 4 | tet_host_line_embed | 1_020 | 0.006 | 0.007 | 448.9 | — |
| 100 | 4 | hex_host_line_embed | 1_420 | 0.006 | 0.009 | 449.2 | — |
| 100 | 8 | tet_host_line_embed | 1_032 | 0.006 | 0.005 | 449.2 | — |
| 100 | 8 | hex_host_line_embed | 1_432 | 0.008 | 0.009 | 450.2 | — |
| 1_000 | 2 | tet_host_line_embed | 10_014 | 0.036 | 0.075 | 508.0 | — |
| 1_000 | 2 | hex_host_line_embed | 14_014 | 0.052 | 0.105 | 539.1 | — |
| 1_000 | 4 | tet_host_line_embed | 10_020 | 0.040 | 0.071 | 539.1 | — |
| 1_000 | 4 | hex_host_line_embed | 14_020 | 0.047 | 0.107 | 540.8 | — |
| 1_000 | 8 | tet_host_line_embed | 10_032 | 0.050 | 0.074 | 540.8 | — |
| 1_000 | 8 | hex_host_line_embed | 14_032 | 0.059 | 0.108 | 543.1 | — |
| 10_000 | 2 | tet_host_line_embed | 100_014 | 0.420 | 0.856 | 1117.9 | — |
| 10_000 | 2 | hex_host_line_embed | 140_014 | 0.466 | 1.197 | 1406.7 | — |
| 10_000 | 4 | tet_host_line_embed | 100_020 | 0.410 | 0.875 | 1406.7 | PASS |
| 10_000 | 4 | hex_host_line_embed | 140_020 | 0.578 | 1.212 | 1456.4 | PASS |
| 10_000 | 8 | tet_host_line_embed | 100_032 | 0.467 | 0.817 | 1456.4 | — |
| 10_000 | 8 | hex_host_line_embed | 140_032 | 0.595 | 1.226 | 1456.4 | — |
| 100_000 | 2 | tet_host_line_embed | 1_000_014 | 4.031 | 9.158 | 7121.4 | — |
| 100_000 | 2 | hex_host_line_embed | 1_400_014 | 5.631 | 16.238 | 9864.3 | — |
| 100_000 | 4 | tet_host_line_embed | 1_000_020 | 5.390 | 10.967 | 9864.3 | — |
| 100_000 | 4 | hex_host_line_embed | 1_400_020 | 8.329 | 16.512 | 9868.0 | — |
| 100_000 | 8 | tet_host_line_embed | 1_000_032 | 7.065 | 11.755 | 9868.0 | — |
| 100_000 | 8 | hex_host_line_embed | 1_400_032 | 8.508 | 16.049 | 9872.5 | — |

## Decision gate status

- `deck_emit_sec`     pass: **True**
- `deck_parse_py_sec` pass: **True**
- `deck_lines`        pass: **True**
- `peak_rss_mb`       pass: **True**

**Overall: PASS** — proceed to Phase 2 (full feature).
