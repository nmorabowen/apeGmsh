### ADDED — session slots draw their `clim`/`cmap`; `overlay` draws the undeformed ghost (ADR 0098 A7.3/A7.5, X3-c, #1733)

Added: a colour-mapped slot's explicit `clim` and `cmap` now reach the
picture in every realize: `session.render()`, `results.render()`, the
session window and `export_animation`. The scale bar shows the same range
and map. A slot left at `None` beside a slot of the same quantity that sets
them draws with that slot's values. With both fields unset nothing changes.

Added: `MeshView.overlay = True` draws the undeformed ghost, which is the
view's own cells at scale 0 in the Outlines style. It follows the view's
scope, the active elements of the stage being drawn, and its section planes.
It draws whether or not the Outlines button is on, and it is never
colour-mapped and adds no scale bar. Setting `overlay` used to raise
`NotImplementedError`.
