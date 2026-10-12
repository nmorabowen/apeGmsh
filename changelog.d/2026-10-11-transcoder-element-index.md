### FIXED — recorder emits record the elements the spec names, under sequential tags (program slice B5-c, #1727, closes #1711)

A `ResolvedRecorderSpec` names elements by FEM id, but its three emit
routes, `emit_recorders`, `to_tcl_commands` and `to_python_commands`, wrote
those ids into the `-ele` list as they were. Under
`element_tags="sequential"` (the bridge default) an element's OpenSees tag
is not its FEM id, so OpenSees recorded other elements, and
`Results.from_recorders` labelled their rows with the requested FEM ids
without a warning. On a 2-D portal whose beam was declared before its
columns, the beam's transcoded `globalForce` rows held column forces (Fy
57034 and 102966 against the beam's -2965.7). This hit every element-scoped
record: `elements`, `gauss` and `line_stations`. Node records were not
affected, because the bridge emits every node under its FEM id. The MPCO
route records the whole domain and its reader already translates.

The three routes now take a keyword-only `model=`, the built
`OpenSeesModel`, and translate each `-ele` list through the model's
`(fem_eid, tag)` pairing. The transcoder is unchanged: it labels rows with
the spec's FEM ids, which are now the elements that were recorded. An
`element_tags="fem"` build emits exactly the commands it emitted before.

**Changed (behaviour):** direct callers of
`ResolvedRecorderSpec.emit_recorders`, `to_tcl_commands` or
`to_python_commands` (and of `LiveRecorders`) whose spec holds an
`elements`, `gauss` or `line_stations` record must now pass `model=`.
Without it the emit raises a `ValueError` that names `model=`, because the
spec carries no tag mode or pairing, so identity cannot be proven from it.
That includes callers whose ids happen to equal the tags (an
`element_tags="fem"` build, or a hand-written deck numbered by FEM id). A FEM
id the model has no element for raises too. The live route refuses at
`__enter__`, before any recorder is issued. Node-only specs, MPCO emits, and
recorders declared on the bridge (`ops.recorder.declare`,
`ops.recorder.Element`) are unaffected: the bridge build already emits them
under OpenSees tags.