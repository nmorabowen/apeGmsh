### FIXED — the EOS case-study and archive notebooks record their runs without `DomainCapture`: the default route on the fork, a native file on stock (#1771, G13)

Twelve notebooks under `examples/EOS Examples/` (ten in `case_studies/`, two
in `archive/`) imported the removed `Recorders` helper and called
`DomainCapture(..., ndm=, ndf=)`, so they raised on `main`. Each now re-runs
its hand-built deck and records it on a route chosen by the build the deck
ran on (`backend_info_of(ops)`):

- **Ladruno fork:** `ResolvedRecorderSpec.emit_default(...)` records the
  run with the fork's own recorder, a `.ladruno` file.
- **Stock openseespy:** the cell samples `ops.nodeDisp` after every step and
  writes a native file with `NativeWriter`. `emit_default` cannot record
  these decks on a stock build without MPCO: its `.out` route needs a bridge
  model archive that declares `ndm`, and a hand-built deck has none.

Both files are read with `Results.from_fem(fem, path)`, which raises no
warning. The decks tag nodes and elements by FEM id, so the FEMData snapshot
is the model. Each recorded cell asserts that the recorded node set is the
one asked for, and that the last recorded step equals the live domain's
displacements. On the fork, `chevron_brace_retrofit_sine` also records line
stations. It asserts that the recorded element set is the requested one and
that its end stations match the live `localForce`.

The notebooks also move to the current FEMData API, which had broken their
build cells as well: `fem.nodes.select(...)` replaces `get` / `get_ids` /
`get_coords`, `fem.elements.select(...).groups()` replaces `get`, and
`g.model.select(...)` replaces `g.model.selection`. Interactive viewer calls
are skipped under `APEGMSH_SKIP_VIEWER`.

`scripts/wire_eos_notebook.py` writes the new cell. It replaces a
notebook's V1 `DomainCapture` cells in place and refuses two builds: an
eigen-only build, because no route here records modes, and an
`ops.analyze` call that the step hook cannot follow.

Not fixed here:

- `ltb_shell` still stops at `g.loads.surface.force_resultant_center_mass`
  on a curve group, which surface loads now refuse.
- `plate_viewer_v2` still stops at the removed `Results.to_vtu` chain.
