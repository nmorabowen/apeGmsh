# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: 2026-08-19 16:16:59 UTC

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1500.0`

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | 1_014 | 0.011 | 0.013 | 484.7 | — |
| 100 | 2 | hex_host_line_embed | 1_414 | 0.011 | 0.013 | 489.8 | — |
| 100 | 4 | tet_host_line_embed | 1_020 | 0.010 | 0.009 | 489.8 | — |
| 100 | 4 | hex_host_line_embed | 1_420 | 0.012 | 0.014 | 490.1 | — |
| 100 | 8 | tet_host_line_embed | 1_032 | 0.018 | 0.008 | 490.1 | — |
| 100 | 8 | hex_host_line_embed | 1_432 | 0.015 | 0.016 | 490.1 | — |
| 1_000 | 2 | tet_host_line_embed | 10_014 | 0.075 | 0.148 | 549.9 | — |
| 1_000 | 2 | hex_host_line_embed | 14_014 | 0.088 | 0.193 | 580.6 | — |
| 1_000 | 4 | tet_host_line_embed | 10_020 | 0.078 | 0.137 | 580.6 | — |
| 1_000 | 4 | hex_host_line_embed | 14_020 | 0.094 | 0.200 | 583.6 | — |
| 1_000 | 8 | tet_host_line_embed | 10_032 | 0.281 | 0.146 | 583.6 | — |
| 1_000 | 8 | hex_host_line_embed | 14_032 | 0.116 | 0.209 | 585.0 | — |
| 10_000 | 2 | tet_host_line_embed | 100_014 | 0.786 | 1.761 | 1162.8 | — |
| 10_000 | 2 | hex_host_line_embed | 140_014 | 0.973 | 1.867 | 1446.6 | — |
| 10_000 | 4 | tet_host_line_embed | 100_020 | 0.760 | 1.324 | 1446.6 | PASS |
| 10_000 | 4 | hex_host_line_embed | 140_020 | 1.198 | 1.884 | 1446.6 | PASS |
| 10_000 | 8 | tet_host_line_embed | 100_032 | 0.907 | 1.353 | 1446.6 | — |
| 10_000 | 8 | hex_host_line_embed | 140_032 | 1.074 | 1.920 | 1478.1 | — |
| 100_000 | 2 | tet_host_line_embed | 1_000_014 | 7.840 | 15.311 | 7163.6 | — |
| 100_000 | 2 | hex_host_line_embed | 1_400_014 | 9.676 | 21.228 | 9920.8 | — |
| 100_000 | 4 | tet_host_line_embed | 1_000_020 | 8.053 | 8.349 | 9920.8 | — |
| 100_000 | 4 | hex_host_line_embed | 1_400_020 | 10.171 | 20.186 | 9921.0 | — |
| 100_000 | 8 | tet_host_line_embed | 1_000_032 | 9.083 | 14.093 | 9921.0 | — |
| 100_000 | 8 | hex_host_line_embed | 1_400_032 | 11.144 | 20.219 | 9921.0 | — |

## Decision gate status

- `deck_emit_sec`     pass: **True**
- `deck_parse_py_sec` pass: **True**
- `deck_lines`        pass: **True**
- `peak_rss_mb`       pass: **True**

**Overall: PASS** — proceed to Phase 2 (full feature).
