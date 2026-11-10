### ADDED — apeGmshViewer draws a state-free user result as a contour; footfall's map is the first (#2800, ADR 0130 D7)

A state-free `ON_NODES` entry of the set's `<stem>.user.h5` is now picked in
the Contour card's User Results group and drawn as a contour, one column at a
time, at whatever phase is on screen. Footfall's ratio map
(`FootfallResult.to_results`, #2782) is the first: `footfall`, with
`footfall_ap`, `footfall_ratio` and `footfall_fdom`.

- Values are joined by node id; a node the file leaves out or holds as `NaN`
  is drawn as no data. The colour range is the result's own unless fixed.
- The app reads the values on request through a new main-process door,
  `user:readResult`, confined to the open set's user file (as `user:read`).
  Main judges A6 staleness first: a stale entry is refused with its reason
  and its `DATA` is never read. The contour, the banner and the manifest
  say why.
- A user result is a contour only: the Deformed card is refused while it is
  shown, and it is never subtracted across variants or drawn over a User
  Combination. A per-state user result, or one not on the nodes, is still
  listed only, and its tooltip says why.
- `--render` names one with `"contour": {"field": "footfall", "component":
  "footfall_ratio", "source": "user"}` and no stage; the manifest's contour
  says `"source": "user"`.
- `examples/footfall_two_bay_shell.py` writes `footfall_ratio.png` through
  the app's `--render` (`apeGmsh.agent_view.render_stills`). It falls back to
  matplotlib only when no viewer is found, and says so.
