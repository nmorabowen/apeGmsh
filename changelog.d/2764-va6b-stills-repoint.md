### CHANGED — on Windows the stills come from apeGmshViewer through `apegmsh-view/1`; Linux keeps VTK (#2764, ADR 0128 slice 6)

On Windows, `results.render`, `results.render_pack`, `results.assess(figures=True)`,
`fem.render`, `python -m apeGmsh.viewers render` and the studio `render(path=)`
door now draw their PNGs with apeGmshViewer, through
`apeGmsh.agent_view.render_stills`. Each `view=` token becomes one
`apegmsh-view/1` shot, as the parity plan
(`internal_docs/plan_viewer_agent_stills_parity.md`) rules:

- The canned pack, and so `assess(figures=True)`, is one spec: `mesh`,
  `contour` and `deformed`, plus the app's `manifest.json` and the matplotlib
  `history.png`. `view="reactions"` is dropped (G1), and so is a velocity or
  acceleration warp (G4). A `deformed` still with no scale takes the old
  auto scale: 0.12 of the model diagonal over the largest |u| (G3).
- A `Results` is drawn from the `.ladruno` it was read from, with the
  `<stem>.h5` model beside it. `Results.path` is the new read-only accessor
  for that file. Every other source is refused with a message naming
  `.ladruno`: an `.mpco`, a native `.h5`, recorders, a Results with no file,
  a `to_units` view, and a `fem=` or `bind()` override that is not the
  `<stem>.h5` mesh (R22b). The stills draw the render canvas only
  (`chrome: "model"`), not the app's panels.
- An in-memory `FEMData` is written to a temporary `model.h5` first (G7).
- No app raises `FileNotFoundError`, and a refused shot raises
  `AgentViewError` (G9). `APEGMSH_SKIP_VIEWER=1` still skips, for every
  source: the skip comes before the `.ladruno` check.
- A stored Gauss component (R16b) is still refused; follow-up slice #2794
  adds it.

Linux and macOS keep the VTK stills unchanged (G2).
