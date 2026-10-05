### REMOVED — plotly preview, `GeomTransfViewer`, `apeGmsh.sensitivity` and the import banner; ADR 0116 on render technologies (program X1-c, #1483)

Four surfaces that the expert panel cut are gone:

- the plotly notebook preview: `apeGmsh.preview`, `g.model.preview()`,
  `g.mesh.preview()` and the module `apeGmsh.viz.NotebookPreview`. For
  pictures in a notebook use `g.plot` (matplotlib) or a `render(...)` still;
- the three.js `GeomTransfViewer` (`apeGmsh.viewers.geom_transf_viewer`) and
  its `apeGmsh.viewers` re-export. A transform-orientation view returns, if
  wanted, as an apeGmshViewer window;
- the finite-difference driver `apeGmsh.sensitivity`;
- the ASCII banner that `import apeGmsh` printed to stderr. `APEGMSH_QUIET`
  no longer changes what `import apeGmsh` prints; the Ladruno fork's splash
  silencer (`LADRUNO_OPENSEES_QUIET`) is unchanged.

New ADR 0116 (Proposed) records the render-technology rule from ADR 0112:
three.js inside apeGmshViewer and matplotlib for new code, with VTK in
sunset inside the frozen Qt viewers.
