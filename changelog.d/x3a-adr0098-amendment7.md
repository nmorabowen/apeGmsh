### DOCS — ADR 0098 Amendment 7 (proposed): slot `clim`/`cmap`, the undeformed ghost, snapshot v2 without `path` plots (X3-a, #1731)

Amendment 7 to ADR 0098 records the maintainer's KEEP ruling on X3: the six
slotless diagram kinds stay Python-only through `show_web().director` until V4,
and the slot catalog stays closed at seven. It specifies three smaller changes
for the slices that follow. Colour-mapped slots gain `clim` and `cmap`
(`None` keeps today's defaults; a map is validated at assignment; two slots of
the same field with conflicting explicit values raise). `MeshView.overlay`
draws the view's scoped, stage-activated cells at scale 0 in the Outlines
style. The session snapshot moves to version 2, the reader keeps accepting
version 1, and the never-drawable `path` plot kind is removed: an old snapshot's
`path` pane is dropped with a restore notice. Amendment 2's expiry trigger now
fires at V4, because X3 keeps `show_web` and the director. No code changes in
this entry; the amendment awaits the maintainer's acceptance.
