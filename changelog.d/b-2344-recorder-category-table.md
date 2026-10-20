### FIXED — a recorder component in the wrong category refuses by name (program slice B, #2344)

A canonical component declared under a category that does not record it used to be accepted. It then
recorded nothing, or recorded another response under its name. For example, `elements` + `stress_xx`
emitted `globalForce`, and `gauss` + `displacement_x` emitted no recorder at all and gave no warning.

Now `ops.recorder.declare(...)`, `RecorderRecord` and `ResolvedRecorderRecord` refuse such a component
when the record is built, and name the category that records it. All three read one table,
`apeGmsh._vocabulary.CATEGORY_COMPONENTS`. Before this, two tables kept by hand disagreed on 20
components. DomainCapture derives its own table from the same one, minus what it cannot capture in
process.

A bare-string `components="displacement_x"` is now read as one component. It used to be split into
letters, and the refusal named `'d'`. For every pair the table holds, the emitted recorder commands are
unchanged.
