### CHANGED — apeGmshViewer labels a renamed `.ladruno`'s stages by their display names and still matches them by program name (#2843, X W4d)

A `.ladruno` stamped with a `stage_names=` rename (W4c, #2823) carries
`INFO/PROVENANCE@stage_display_names` beside the program's `stage_names`.
The viewer now reads it. Each results stage carries two names:

- `name`, the label every site shows: the display name;
- `programName`, the program's stage name, used by every match against the
  model. That covers `/sequence` activation, the loads overlay, span-load
  reconstruction and the variant difference.

Before, the viewer showed the program names only. Matching a display name
against the model would have read a renamed stage as `absent`, drawn its
inactive elements and lost its loads.

A display list that is malformed, miscounted, or has no usable `stage_names`
to bind by shows the program names and warns. Span-load reconstruction
refuses a results stage that carries no program name.

A view spec names a stage by either name, so a spec written against a
program name keeps resolving after a rename. When the two names point at
different stages, the spec is refused and asked to name the stage by its
index. A snapshot writes the program name.

Not changed here: a partitioned set takes its display names from part 0
without checking that the parts agree. `stage_names` has the same gap.
