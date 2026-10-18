### ADDED — apeGmshViewer has a `.ladruno` reader, not yet wired into the app (ADR 0122 slice 4a, #2126)

`apeGmshViewer/src/reader/ladruno.ts` opens one fork-written `.ladruno` with
h5wasm and indexes it in the shape the native results reader gives: stages,
time, nodal components, one step by hyperslab, and one node's history. It
checks `INFO@GENERATOR == "Ladruno"` and `INFO@FORMAT_VERSION` before it reads
anything else, and refuses any other value with a typed `LadrunoError`.
`tests/test_app_ladruno_gate_drift.py` keeps its version set equal to Python's
`LADRUNO_SUPPORTED_FORMAT_VERSIONS`. Components keep the file's spelling
(`DISPLACEMENT/Ux`), with no token table (ADR 0122 D7). A `KIND=eigen` stage
reads as static, and each recorded mode becomes its own `mode` stage. The
lowercase `INFO/PROVENANCE` attributes are reported one by one as present,
absent or malformed, and the reader never coerces one. A nodal result whose
datasets disagree is refused by name, and element results are listed as
unread. A `<stem>.part-<N>.ladruno` is refused. The app does not open
`.ladruno` files yet: the wiring and the pairing are the next slice.
