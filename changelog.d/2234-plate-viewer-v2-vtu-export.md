### FIXED — the `plate_viewer_v2` archive notebook exports to ParaView with `write_vtu` and probes with pyvista, so it runs end to end (#2234)

`examples/EOS Examples/archive/plate_viewer_v2` stopped at cell 11, which
called `Results.to_vtu`, removed by the Results rebuild. The cells after it
also used the removed `Results.to_mesh_data` and the retired Python viewer's
`ProbeEngine` and `ViewportRenderer`.

- **Export:** cell 11 writes the mesh and the section-1 fields (displacement,
  radial displacement, nodal-averaged and element stresses, von Mises) to one
  `.vtu` with `write_vtu` from `apeGmsh.viz.VTKExport`. It reads the file back
  and asserts that every array round-trips.
- **Probes and stills:** the point, line and plane probes use pyvista on that
  `.vtu` (`sample`, `sample_over_line`, `slice`). The offscreen stills use a
  `pyvista.Plotter`.
- **Lamé check:** the analytical radial displacement had the wrong sign on its
  `B/r` term, so its curve disagreed with the FEM. It now reads
  `u_r = (1+ν)/E·[(1-2ν)·A·r − B/r]` (plane strain). The line-probe cell
  asserts the FEM `u_r` is within 1% of it: the error is 0.6%, against 221%
  with the old formula.

The record cells from #1771 are unchanged. The notebook runs on the Ladruno
fork and on stock openseespy.
