# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: 2026-08-04 16:26:27 UTC

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1500.0`

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | 1_014 | 0.007 | 0.006 | 429.3 | — |
| 100 | 2 | hex_host_line_embed | 1_414 | 0.006 | 0.008 | 434.9 | — |
| 100 | 4 | tet_host_line_embed | 1_020 | 0.005 | 0.006 | 434.9 | — |
| 100 | 4 | hex_host_line_embed | 1_420 | 0.006 | 0.009 | 434.9 | — |
| 100 | 8 | tet_host_line_embed | 1_032 | 0.006 | 0.006 | 434.9 | — |
| 100 | 8 | hex_host_line_embed | 1_432 | 0.007 | 0.008 | 435.6 | — |
| 1_000 | 2 | tet_host_line_embed | 10_014 | 0.036 | 0.072 | 493.2 | — |
| 1_000 | 2 | hex_host_line_embed | 14_014 | 0.045 | 0.101 | 524.3 | — |
| 1_000 | 4 | tet_host_line_embed | 10_020 | 0.038 | 0.070 | 524.3 | — |
| 1_000 | 4 | hex_host_line_embed | 14_020 | 0.047 | 0.100 | 527.9 | — |
| 1_000 | 8 | tet_host_line_embed | 10_032 | 0.044 | 0.069 | 527.9 | — |
| 1_000 | 8 | hex_host_line_embed | 14_032 | 0.052 | 0.100 | 530.9 | — |
| 10_000 | 2 | tet_host_line_embed | 100_014 | 0.352 | 0.956 | 1112.6 | — |
| 10_000 | 2 | hex_host_line_embed | 140_014 | 0.624 | 1.635 | 1395.5 | — |
| 10_000 | 4 | tet_host_line_embed | 100_020 | 0.733 | 1.169 | 1395.5 | PASS |
| 10_000 | 4 | hex_host_line_embed | 140_020 | 0.672 | 1.810 | 1395.5 | PASS |
| 10_000 | 8 | tet_host_line_embed | 100_032 | 0.721 | 1.247 | 1395.5 | — |
| 10_000 | 8 | hex_host_line_embed | 140_032 | 0.782 | 1.738 | 1443.4 | — |
| 100_000 | 2 | tet_host_line_embed | 1_000_014 | 5.129 | 12.278 | 7110.1 | — |
| 100_000 | 2 | hex_host_line_embed | 1_400_014 | 6.793 | 17.353 | 9854.5 | — |
| 100_000 | 4 | tet_host_line_embed | 1_000_020 | 4.277 | 12.184 | 9854.5 | — |
| 100_000 | 4 | hex_host_line_embed | 1_400_020 | 6.027 | 16.267 | 9854.9 | — |
| 100_000 | 8 | tet_host_line_embed | 1_000_032 | 6.529 | 11.006 | 9854.9 | — |
| 100_000 | 8 | hex_host_line_embed | 1_400_032 | 8.463 | 15.770 | 9873.6 | — |

## Decision gate status

- `deck_emit_sec`     pass: **True**
- `deck_parse_py_sec` pass: **True**
- `deck_lines`        pass: **True**
- `peak_rss_mb`       pass: **True**

**Overall: PASS** — proceed to Phase 2 (full feature).
