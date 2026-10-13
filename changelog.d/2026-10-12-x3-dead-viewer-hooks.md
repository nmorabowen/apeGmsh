### REMOVED — dead viewer hooks left by X3-d/X3-e (#1782)

`FEMSceneData.pick_engine` and `FEMSceneData.opacity_controller` (always
`None` since the Qt `ResultsViewer` and its pick/opacity classes were
deleted) and `GaussPointDiagram._register_pick` / `_unregister_pick` (a
no-op against `scene.pick_engine`) are removed, along with stale
docstrings that still named the deleted modules. No behaviour change.
