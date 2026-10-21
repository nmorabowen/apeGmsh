### CHANGED — the docs and the skill teach the default results route (program slice X-mig, #1989)

The tutorials, examples, how-tos, the bridge concept page and the two skill references
(`opensees-bridge.md`, `ladruno.md`) no longer teach `DomainCaptureSpec` /
`ops.domain_capture`. An in-process run is recorded with
`ResolvedRecorderSpec.emit_default(path, model_h5=...)` and read with `run.results()`: the fork
records `.ladruno`, a stock build with MPCO records `.mpco`, and pip openseespy records classic
`.out` files. The loop steps the module `get_ops()` returns. Modes come from
`ops.eigen(n).results()` (or `to_native` / `to_ladruno`). Every changed page's script was run on
the fork and on pip openseespy and prints the numbers the page shows. Two results that had
drifted on `main` are corrected along the way: the tie example now shows its current top
displacement (0.12831 mm), and the multipart example keeps its `Results` open for the plot that
follows. `DomainCapture` stays as the legacy route, labelled as such in the API reference and
the design page, until its removal slice.
