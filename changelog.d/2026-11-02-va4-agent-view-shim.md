### ADDED — `apeGmsh.agent_view.render_stills`: apeGmshViewer stills plus a manifest in one call (ADR 0128 slice 4, #2653)

`render_stills(spec, out)` takes every shot of an `apegmsh-view/1` view spec with the installed
apeGmshViewer and returns its `manifest.json` as a dict. It finds the executable (`$APEGMSH_VIEWER`, else
the per-user install under `%LOCALAPPDATA%\Programs\apegmsh-viewer`), writes the spec, and runs
`apeGmshViewer --render <spec> --out <dir>`. It imports nothing from the app and does not check the
spec: the app's validator is the only one. Any refusal raises `AgentViewError`, which names every refused
key or shot and carries this run's manifest. A `$APEGMSH_VIEWER` that names no file raises instead of
falling back to the install, and a `manifest.json` that was in `out` before the run is never read. The
skill's `assess.md` documents the call and the ADR 0128 D8 voice rule for captions and summaries.
