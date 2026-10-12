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

**Changed (behaviour):** an element-scoped recorder emit without `model=`
now refuses with a `ValueError` that names `model=`, and so does a FEM id the
model has no element for. Identity is never assumed. The live route refuses
at `__enter__`, before any recorder reaches the domain. Node-only specs need
no `model=`.
