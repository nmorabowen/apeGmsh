### FIXED — the EOS case-study and archive notebooks record their runs on the default results route, not `DomainCapture` (#1771, G13)

Twelve notebooks under `examples/EOS Examples/` (ten in `case_studies/`, two
in `archive/`) imported the removed `Recorders` helper and called
`DomainCapture(..., ndm=, ndf=)`, so they raised on `main`. Each now re-runs
its hand-built deck inside `ResolvedRecorderSpec.emit_default(...)`: the
Ladruno fork records a `.ladruno` file, a stock build with MPCO records
`.mpco`, and pip openseespy records classic `.out` files. `run.results()`
reads the run back. The decks tag nodes and elements by FEM id, so the
FEMData archive (`fem.to_h5`) is the model archive, with the identity
element pairing. Each recorded cell asserts that the last recorded step
equals the live domain's displacements. `chevron_brace_retrofit_sine` also
records line stations and checks its end stations against the live
`localForce`.

The notebooks also move to the current FEMData API, which had broken their
build cells as well: `fem.nodes.select(...)` replaces `get` / `get_ids` /
`get_coords`, `fem.elements.select(...).groups()` replaces `get`, and
`g.model.select(...)` replaces `g.model.selection`. Interactive viewer calls
are skipped under `APEGMSH_SKIP_VIEWER`.

`scripts/wire_eos_notebook.py` writes the new cell. It replaces a
notebook's V1 `DomainCapture` cells in place and refuses an eigen-only
build, because no default route records modes.

Not fixed here:

- `ltb_shell` still stops at `g.loads.surface.force_resultant_center_mass`
  on a curve group, which surface loads now refuse.
- `plate_viewer_v2` still stops at the removed `Results.to_vtu` chain.
- On pip openseespy the `.out` route refuses the FEMData archive, which
  declares no `ndm`.
