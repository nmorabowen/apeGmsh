### FIXED — Deck replay writes rebar cells after the reinforce ties (#1828)

For a model built with `g.rebar.place(emit_elements=True)`,
`OpenSeesModel.from_h5(...).build("tcl" | "py" | "live")` wrote the bar's
`CorotTruss` cells before the host elements. The bridge writes them after the
`LadrunoEmbeddedRebar` ties. The replay now writes them after the ties too, so
the replayed deck matches the bridge deck byte for byte. A neutral zone that
names a bar cell the archive's element store does not hold now raises
`ValueError`. The H5 rewrite keeps its element order.
