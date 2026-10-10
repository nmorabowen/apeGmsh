### CHANGED — deck headers stamp the target they are built for (F2-d phase 2, #1511)

`apeSees.tcl()`, `.py()` and both `modal_deck()` solvers now pass a backend
to the deck emitters, through `apeGmsh.opensees.emitter.tcl.deck_backend`.
A bridge built with `OpenSeesTarget(mode="fork")` or `mode="stock"` (or
`require_fork=True`) writes `# apeGmsh <version>; backend <fork|stock>` as
the deck's second line. With `mode="auto"` or no target, the deck is
unstamped and byte-identical to before. The stamp never reads the live
resolver: that verdict describes the in-process module, not the binary a
deck runs on, and reading it would make the same model's deck depend on
what ran earlier in the process. So no build is stamped on this path.

`OpenSeesModel.from_h5(...).build("tcl" / "py")` goes through the same seam.
The archive carries no target, so it is the auto case and stays unstamped.
`ModelData.recorder_commands` stays unstamped: recorder fragments are not
decks. The golden builder pins `OpenSeesTarget(mode="stock")`, so every deck
golden gains `# apeGmsh <VERSION>; backend stock`. The `require_fork`
refusal now names the `ladrunoBuild()` signal instead of `criticalTimeStep`,
and `DECK_BANNER` is exported from `emitter.tcl`.
