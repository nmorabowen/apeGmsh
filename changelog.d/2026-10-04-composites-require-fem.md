### FIXED — Results `pg=` / `label=` / `selection=` name the zone below its floor (#1371)

On a native results file whose embedded `/model` is below its ADR 0113
floor and with no `fem=` supplied, `results.nodes.get(..., pg=...)` and the
element-side `pg=` / `label=` / `selection=` resolution raised a generic
"Pass fem=" `RuntimeError`. Both resolvers in `results/_composites.py` now go
through `_require_fem`, so they raise the `SchemaVersionError` that names the
zone (D9) and keeps the `fem=` hint, like the other composites.
`tests/results/test_composites_flagged_fem.py` covers both selector paths
and the `plot.vector_glyph` gate.
