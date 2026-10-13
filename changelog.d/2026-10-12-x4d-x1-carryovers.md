### CHANGED — `APEGMSH_QUIET` dropped; two EOS notebooks retargeted; ADR 0116 D3/D4 quirk lint (program X4-d, #1808)

- `APEGMSH_QUIET` has done nothing since X1-c removed the import banner, and
  apeGmsh no longer sets it: `doctor`, the studio MCP and index build, the
  habitat template (`cursor/mcp.json`, `scripts/_habitat.py`), the Cursor
  config, the launch scripts, the skill and the tests. A user who still sets
  it sees no error; it simply has no effect. `LADRUNO_OPENSEES_QUIET` (the
  fork's stdout splash) is unchanged and still needed for MCP stdio.
- Two `examples/EOS Examples` notebooks called APIs removed in X1-c.
  `chevron_brace_kink_imperfection` now draws `m.plot.geometry().label_nodes()`
  (matplotlib) instead of `m.model.preview()` / `m.mesh.preview()`;
  `frame3D_slab_opensees_manual_results` replaces its `GeomTransfViewer` cell
  with a note (ADR 0116 D2).
- `scripts/check_quirks.py` gains `vtk-import` (pyvista, pyvistaqt or vtk in
  `src/` outside `src/apeGmsh/viewers/`, ADR 0116 D3) and `render-tech-import`
  (plotly, bokeh, k3d, pythreejs or trame anywhere in `src/`, D4). Neither has
  a per-site waiver. The one existing site, the trame backend of `WebViewer`,
  is in `scripts/quirks_render_allowlist.txt`, which may only shrink.
