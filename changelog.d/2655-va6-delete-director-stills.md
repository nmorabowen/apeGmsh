### REMOVED — the director stills path in `apeGmsh.viewers.render`: `render_results` and `render_pack` (#2655, ADR 0128 slice 6)

`results.render` and `results.render_pack` have drawn their stills on the
session path (`viewers/session/_results_stills.py`) since X2-d. The old
director path they replaced, `apeGmsh.viewers.render.render_results` and
`render_pack` with their private helpers, had no caller left in `src/`. It is
deleted, so `viewers/render.py` no longer imports `apeGmsh.viewers.diagrams`
and leaves that guard's allowlist.

Nothing a user calls changes. `results.render`, `results.render_pack`,
`results.assess(figures=True)`, `fem.render`, `g.model.render`,
`g.mesh.render` and the `python -m apeGmsh.viewers render` command work as
before. Code that imported `render_results` or `render_pack` from
`apeGmsh.viewers.render` directly now gets an `ImportError`; call the
`Results` methods instead.
