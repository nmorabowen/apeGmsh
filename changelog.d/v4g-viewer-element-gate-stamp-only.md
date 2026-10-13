### CHANGED — apeGmshViewer: element row ids are verified only by the `element_index_space` stamp (V4g, #1768)

The app's results reader no longer treats `source_type = "domain_capture"`
as proof that a file's `_element_index` rows are FEM element ids. A file's
element mapping is verified if and only if its root attribute
`element_index_space` is the string `"fem"` (`architecture/h5-schema.md`,
§`element_index_space`). An absent stamp, a non-string one, or any other
value reads as `unverified`, whatever the `source_type` and version. The
reader now targets results zone 1.2.0 (floor 1.0.0), so a 1.2.0 file opens
without a newer-minor warning. No view draws element data yet, so nothing
on screen changes. A new fixture, `fixtures/results_stamped.h5`, is written
by `NativeWriter.stamp_element_index_space("fem")`
(`scripts/make_viewer_results_fixtures.py --stamped`).
