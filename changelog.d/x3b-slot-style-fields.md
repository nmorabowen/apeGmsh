### CHANGED (behaviour) — session slots carry `clim`/`cmap`; snapshot v2; the `path` plot kind is gone (ADR 0098 A7, X3-b, #1732)

Added: the four colour-mapped slot records (`Contour`, `Vector`, `Gauss`,
`Sand`) take `clim=(low, high)` and `cmap=`. `None`, the default, keeps each
kind's existing range and colormap, so no default changed. `cmap` accepts any
matplotlib name, `'jet'` included, and an unknown name or a malformed `clim`
raises a `ValueError` when the record is built. An explicit `cmap` needs
matplotlib (the `plot` extra) and raises an `ImportError` without it; a saved
`cmap` restored without matplotlib falls back to the default with a notice. Two occupied slots of one
quantity that set different values raise a `ValueError` (one field, one
scale). `MeshView.legends()` reports the values a slot set, and
`slot_style(record)` reads them. The session stores, validates and saves the
fields; the window and `render()` do not apply them yet (X3-c).

Changed: the session snapshot writer writes schema version 2. The reader takes
versions 1 and 2. A version-1 slot restores with no `clim`/`cmap`, and a
version-1 file that carries them refuses. `PLOT_KINDS` is now
`("history", "xy")`: `PlotView(kind="path")` raises, and a `path` plot pane in
a version-1 snapshot is dropped with a restore notice. `path` plots were never
drawable. An unknown slot category still refuses in both versions.
