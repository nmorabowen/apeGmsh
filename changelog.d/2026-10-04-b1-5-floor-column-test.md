### CHANGED — the h5-schema doc-registry test holds the Floor column too (program slice B1-5, #1400; closes #1375)

`test_h5_schema_doc_registry_matches_writer_constants` now also asserts that the **Floor** column of the
`h5-schema.md` zone registry equals `reader_floor(zone)` for the neutral, opensees, results, geometry and
provenance zones (ADR 0113 INV-3), so a floor cell can no longer drift silently. Five round-trip tests named
`*_within_window` are renamed `*_at_or_above_floor`, and the dated viewer measurement note no longer calls it
the reader window. No behaviour change.
