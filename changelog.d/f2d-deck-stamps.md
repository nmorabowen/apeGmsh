### ADDED — provenance stamps in Tcl and openseespy deck headers (F2-d, #1511)

`TclEmitter(backend=...)` and `PyEmitter(backend=...)` take an optional
`BackendInfo`. When one is given, the deck header carries a second comment
line after the banner, `# apeGmsh <version>; backend <fork|stock>`, plus
`; build <sha>` when the fork build is known
(`apeGmsh.opensees.emitter.tcl.provenance_stamp`). Anything that is not a
`BackendInfo` is refused with `TypeError`. With no backend (the default)
both headers are byte-identical to before, so existing decks and the golden
corpus do not change. The golden builder masks the stamp's version and build
(`<VERSION>` / `<BUILD>`) so a release or a fork rebuild never rewrites a
golden; `backend` stays visible. `apeSees.tcl()` / `.py()` do not pass a
backend yet; that is phase 2 of the slice.
