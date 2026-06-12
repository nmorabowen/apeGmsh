# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: 2026-06-12 02:28:50 UTC

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1500.0`

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | 1_014 | 0.007 | 0.011 | 218.8 | — |
| 100 | 2 | hex_host_line_embed | 1_414 | 0.007 | 0.014 | 222.2 | — |
| 100 | 4 | tet_host_line_embed | 1_020 | 0.010 | 0.013 | 222.2 | — |
| 100 | 4 | hex_host_line_embed | 1_420 | 0.011 | 0.016 | 222.6 | — |
| 100 | 8 | tet_host_line_embed | 1_032 | 0.008 | 0.010 | 222.6 | — |
| 100 | 8 | hex_host_line_embed | 1_432 | 0.009 | 0.018 | 222.8 | — |
| 1_000 | 2 | tet_host_line_embed | 10_014 | 0.060 | 0.132 | 277.4 | — |
| 1_000 | 2 | hex_host_line_embed | 14_014 | 0.259 | 0.196 | 303.2 | — |
| 1_000 | 4 | tet_host_line_embed | 10_020 | 0.056 | 0.134 | 303.2 | — |
| 1_000 | 4 | hex_host_line_embed | 14_020 | 0.090 | 0.189 | 306.5 | — |
| 1_000 | 8 | tet_host_line_embed | 10_032 | 0.067 | 0.126 | 306.5 | — |
| 1_000 | 8 | hex_host_line_embed | 14_032 | 0.292 | 0.184 | 306.5 | — |
| 10_000 | 2 | tet_host_line_embed | 100_014 | 0.775 | 1.449 | 841.2 | — |
| 10_000 | 2 | hex_host_line_embed | 140_014 | 1.346 | 1.925 | 1055.6 | — |
| 10_000 | 4 | tet_host_line_embed | 100_020 | 0.864 | 1.390 | 1055.6 | PASS |
| 10_000 | 4 | hex_host_line_embed | 140_020 | 1.322 | 1.884 | 1056.9 | PASS |
| 10_000 | 8 | tet_host_line_embed | 100_032 | 1.012 | 1.339 | 1056.9 | — |
| 10_000 | 8 | hex_host_line_embed | 140_032 | 1.588 | 2.014 | 1056.9 | — |
| 100_000 | 2 | tet_host_line_embed | 1_000_014 | 7.355 | 16.578 | 5475.4 | — |
| 100_000 | 2 | hex_host_line_embed | 1_400_014 | 9.852 | 19.505 | 8930.2 | — |
| 100_000 | 4 | tet_host_line_embed | 1_000_020 | 6.816 | 13.472 | 8930.2 | — |
| 100_000 | 4 | hex_host_line_embed | 1_400_020 | 13.815 | 21.425 | 9049.1 | — |
| 100_000 | 8 | tet_host_line_embed | 1_000_032 | 8.999 | 13.273 | 9049.1 | — |
| 100_000 | 8 | hex_host_line_embed | 1_400_032 | 11.651 | 18.049 | 9050.3 | — |

## Decision gate status

- `deck_emit_sec`     pass: **True**
- `deck_parse_py_sec` pass: **True**
- `deck_lines`        pass: **True**
- `peak_rss_mb`       pass: **True**

**Overall: PASS** — proceed to Phase 2 (full feature).
