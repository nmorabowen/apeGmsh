### Fixed — a stage-scoped still draws that stage (X2-g, #1742)

`results.stage(s).render()`, `.render_pack()` and `.assess(figures=True)`
drew the last stage's step and active elements whatever stage `s` was, a
silently wrong picture on any multi-stage file. They now draw stage `s`, as
`export_animation` already did: the scope first, then the only stage. An
unscoped multi-stage Results still draws its last stage, as documented, and
the pack's default contour component and reactions still follow the scope. A
scope that names no stage in the file raises instead of drawing another one.
