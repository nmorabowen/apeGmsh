### ADDED — apeGmshViewer tool windows hold tiles (ADR 0137 H4)

A tool window now holds several tool tiles side by side and one above the other, with splitters
between them to resize. Each tile has its title and a × to close it, and the window's "+ Add tile"
picker adds more. Spotlight (Ctrl+K) offers **Add Tile: Step Readout** and **Add Tile: Node
History**, into the newest tool window (or a new one when none is open), and **New Tool Window** for
another monitor. The picked node's history chart is now its own tool, **Node History**, so two of
them can chart different components side by side; Step Readout keeps the step, the step buttons and
the component's range at the step. Up to eight tiles in a window are live; more wait as paused tabs,
and showing one pauses the tile shown longest ago.
