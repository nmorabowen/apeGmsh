# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: 2026-08-04 15:49:47 UTC

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1500.0`

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | 1_014 | 0.015 | 0.008 | 429.9 | — |
| 100 | 2 | hex_host_line_embed | 1_414 | 0.008 | 0.017 | 435.0 | — |
| 100 | 4 | tet_host_line_embed | 1_020 | 0.012 | 0.011 | 435.0 | — |
| 100 | 4 | hex_host_line_embed | 1_420 | 0.012 | 0.018 | 435.1 | — |
| 100 | 8 | tet_host_line_embed | 1_032 | 0.012 | 0.011 | 435.1 | — |
| 100 | 8 | hex_host_line_embed | 1_432 | 0.015 | 0.019 | 436.7 | — |
| 1_000 | 2 | tet_host_line_embed | 10_014 | 0.069 | 0.134 | 494.6 | — |
| 1_000 | 2 | hex_host_line_embed | 14_014 | 0.085 | 0.175 | 525.5 | — |
| 1_000 | 4 | tet_host_line_embed | 10_020 | 0.090 | 0.133 | 525.5 | — |
| 1_000 | 4 | hex_host_line_embed | 14_020 | 0.093 | 0.159 | 528.8 | — |
| 1_000 | 8 | tet_host_line_embed | 10_032 | 0.097 | 0.103 | 528.8 | — |
| 1_000 | 8 | hex_host_line_embed | 14_032 | 0.066 | 0.134 | 533.1 | — |
| 10_000 | 2 | tet_host_line_embed | 100_014 | 0.479 | 1.041 | 1110.0 | — |
| 10_000 | 2 | hex_host_line_embed | 140_014 | 0.680 | 1.918 | 1376.3 | — |
| 10_000 | 4 | tet_host_line_embed | 100_020 | 0.627 | 1.017 | 1376.3 | PASS |
| 10_000 | 4 | hex_host_line_embed | 140_020 | 0.887 | 1.806 | 1413.7 | PASS |
| 10_000 | 8 | tet_host_line_embed | 100_032 | 0.618 | 1.006 | 1413.7 | — |
| 10_000 | 8 | hex_host_line_embed | 140_032 | 0.646 | 1.641 | 1468.2 | — |
| 100_000 | 2 | tet_host_line_embed | 1_000_014 | 4.462 | 10.987 | 7108.4 | — |
| 100_000 | 2 | hex_host_line_embed | 1_400_014 | 6.719 | 16.143 | 9851.2 | — |
| 100_000 | 4 | tet_host_line_embed | 1_000_020 | 5.427 | 12.020 | 9851.2 | — |
| 100_000 | 4 | hex_host_line_embed | 1_400_020 | 8.136 | 21.691 | 9855.7 | — |
| 100_000 | 8 | tet_host_line_embed | 1_000_032 | 7.723 | 15.968 | 9855.7 | — |
| 100_000 | 8 | hex_host_line_embed | 1_400_032 | 10.863 | 21.133 | 9860.7 | — |

## Decision gate status

- `deck_emit_sec`     pass: **True**
- `deck_parse_py_sec` pass: **True**
- `deck_lines`        pass: **True**
- `peak_rss_mb`       pass: **True**

**Overall: PASS** — proceed to Phase 2 (full feature).
