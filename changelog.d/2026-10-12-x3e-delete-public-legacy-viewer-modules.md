### REMOVED — five public-named modules only the legacy Qt window used (X3-e, #1769)

With the Qt `ResultsViewer` gone (X3-d, #1734), five modules that only it
reached are deleted: `apeGmsh.viewers.core.opacity_controller`
(`OpacityController`), `apeGmsh.viewers.core.results_pick`
(`install_results_pick`, `ResultsPickController`, `PickResult`,
`BoxPickResult`), `apeGmsh.viewers.core.results_pick_engine`
(`PickInventory`), `apeGmsh.viewers.overlays.local_axes_overlay`
(`LocalAxesOverlay`) and `apeGmsh.viewers.overlays.probe_overlay`
(`ProbeOverlay` and its point/line/plane probe results). None was
documented. Node and Gauss-point picking, including box picks, is on the
`ResultsSession` path (`results.viewer()`); per-geometry opacity, the
local-axes triads and the point/line/plane probes are accepted gaps of
ADR 0098 whose home is apeGmshViewer. `apeGmsh.viewers.animation.export_animation`
(the bring-your-own plotter + director animation loop) stays until the
`ResultsDirector` is deleted at V4; `Results.export_animation` is
unaffected.
