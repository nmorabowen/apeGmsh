# Cross-rank constraint cost — ADR 0038 §"v1 scope gate"

Last run: no runs yet — populated by nightly `benchmarks.yml` workflow

## Thresholds (ADR 0038 §"v1 scope gate", 10k × 4 ranks)

- `deck_emit_sec     < 5.0`
- `deck_parse_py_sec < 2.0`
- `deck_lines        < 500_000`
- `peak_rss_mb       < 1_500`

The 10k × 4 cell (10 000 embedded line nodes × 4 ranks ×
`tet_host_line_embed`) is the gate. Smaller cells provide a scaling
trace; larger cells inform the 100k-fail fallback branch.

## Results

| interface_size | ranks | element_kind | deck_lines | deck_emit_sec | deck_parse_py_sec | peak_rss_mb | pass_at_10k×4 |
|---:|---:|---|---:|---:|---:|---:|:---:|
| 100 | 2 | tet_host_line_embed | — | — | — | — | — |
| 100 | 2 | hex_host_line_embed | — | — | — | — | — |
| 100 | 4 | tet_host_line_embed | — | — | — | — | — |
| 100 | 4 | hex_host_line_embed | — | — | — | — | — |
| 100 | 8 | tet_host_line_embed | — | — | — | — | — |
| 100 | 8 | hex_host_line_embed | — | — | — | — | — |
| 1_000 | 2 | tet_host_line_embed | — | — | — | — | — |
| 1_000 | 2 | hex_host_line_embed | — | — | — | — | — |
| 1_000 | 4 | tet_host_line_embed | — | — | — | — | — |
| 1_000 | 4 | hex_host_line_embed | — | — | — | — | — |
| 1_000 | 8 | tet_host_line_embed | — | — | — | — | — |
| 1_000 | 8 | hex_host_line_embed | — | — | — | — | — |
| 10_000 | 2 | tet_host_line_embed | — | — | — | — | — |
| 10_000 | 2 | hex_host_line_embed | — | — | — | — | — |
| 10_000 | 4 | tet_host_line_embed | — | — | — | — | — |
| 10_000 | 4 | hex_host_line_embed | — | — | — | — | — |
| 10_000 | 8 | tet_host_line_embed | — | — | — | — | — |
| 10_000 | 8 | hex_host_line_embed | — | — | — | — | — |
| 100_000 | 2 | tet_host_line_embed | — | — | — | — | — |
| 100_000 | 2 | hex_host_line_embed | — | — | — | — | — |
| 100_000 | 4 | tet_host_line_embed | — | — | — | — | — |
| 100_000 | 4 | hex_host_line_embed | — | — | — | — | — |
| 100_000 | 8 | tet_host_line_embed | — | — | — | — | — |
| 100_000 | 8 | hex_host_line_embed | — | — | — | — | — |

## Decision gate status

PENDING — awaits first nightly run.

Per ADR 0038 §"v1 scope gate":

- **All thresholds pass at 10k × 4** → proceed to Phase 2 (full feature).
- **10k × 4 passes but 100k × 8 fails** → proceed with
  `WARN_INTERFACE_SIZE = 50_000` hard warning at compose time.
- **Any threshold breaches at 10k × 4** → abandon the full feature in
  v1; ship Phase 2 + stripped Phase 3 (mesh-cache-only `g.compose()`;
  any cross-module MP-constraint raises `ComposeUnsupportedError`).
