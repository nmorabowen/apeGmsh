### FIXED — a bridge with a `ComputedSection` writes its automatic `model.h5` again (#2567)

The bridge digests every primitive by pickling it before it writes the
automatic `model.h5` (ADR 0112 D1, rule P5). A `ComputedSection`'s
`SectionProperties` analyzer held its materials in a read-only
`mappingproxy`, which does not pickle. The write was skipped with only a
`BridgeArtifactWarning` ("cannot pickle 'mappingproxy' object"), so no
`<model_name>.h5` was left beside the run.

`SectionProperties` now pickles. It thaws the material view into a dict
for the pickle and refreezes it on load, and its memoised analyses travel
with it. The automatic write succeeds, and the archive carries
`/opensees/computed_sections`.
