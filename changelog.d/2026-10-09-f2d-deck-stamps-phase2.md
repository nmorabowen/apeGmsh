### CHANGED — deck headers stamp the target they are built for (F2-d phase 2, #1511)

`apeSees.tcl()`, `.py()` and both `modal_deck()` solvers, and
`OpenSeesModel.from_h5(...).build("tcl" / "py")`, now pass a backend to the
deck emitters, so the deck's second line can read
`# apeGmsh <version>; backend <fork|stock>[; build <sha>]`. The backend comes
from `apeGmsh.opensees.emitter.tcl.deck_backend(target)`:

- `OpenSeesTarget(mode="fork")` or `mode="stock"` always stamps that kind.
  The build is added only when the live resolver has already answered the
  same kind in this process.
- `mode="auto"`, or no target (the archive path carries none), stamps the
  live resolver's verdict when it has already answered, and stamps nothing
  when it has not. Writing a deck never imports openseespy to find out.

`ModelData.recorder_commands` stays unstamped: recorder fragments are not
decks. The golden builder pins `OpenSeesTarget(mode="stock")`, so every deck
golden gains `# apeGmsh <VERSION>; backend stock`. The `require_fork`
refusal now names the `ladrunoBuild()` signal instead of
`criticalTimeStep`, and `DECK_BANNER` is exported from `emitter.tcl`.
