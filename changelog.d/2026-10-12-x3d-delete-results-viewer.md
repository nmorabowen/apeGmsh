### REMOVED — the legacy Qt `ResultsViewer` (X3-d, #1734)

The retired Qt results window is deleted: `apeGmsh.viewers.results_viewer`,
its three side panels (fiber-section scatter, layer-thickness profile,
isochrone profile chart) and the 23 private modules only it reached
(outline tree, settings tabs, plot pane, time scrubber, output dock and log
router, pick-readout and empty-state HUDs, colour-map editor, File → Open
dialog, scope gizmo, section-plane panel and cut-face caps). It had no
public import path since ADR 0098 S6a. Open results with `results.viewer()`
(a `ResultsSession` window) or `python -m apeGmsh.viewers <results.h5>`;
`Results.show_web` and `Results.export_animation` are unaffected. The
fiber-section, layer-stack and isochrone-profile diagrams now inherit the
base `make_side_panel` hook, which returns `None`. The public-named modules
that only the old window used (`core/opacity_controller`, `core/results_pick`,
`core/results_pick_engine`, `overlays/local_axes_overlay`,
`overlays/probe_overlay`, `viewers/animation.py`) stay until X3-e.
