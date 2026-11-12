### CHANGED — on Windows a stored Gauss component contour draws through apeGmshViewer (#2794, ADR 0128 slice 6)

On Windows, `results.render(view="contour" | "deformed", component=...)` with a
stored Gauss component such as `stress_xx` now draws through `apegmsh-view/1`
instead of being refused (parity plan row R16b, ruled a repoint):

- The shot names the app's element field: the `.ladruno` token and the file's
  own column label (`stress` and `stress/sigma11`), the column Python reads as
  that component. A component the file holds in two columns, or one named by
  the response catalog rather than the file, is refused by name.
- The still keeps v1's averaged picture: `cell_averaged` for one Gauss point,
  `node` (extrapolated to the corners and averaged) for more.
- A derived scalar such as `von_mises_stress` is still refused (R16c, G10).
- `python -m apeGmsh.viewers render` now skips under `APEGMSH_SKIP_VIEWER=1`
  for any source, as in-process `render` and `assess` do; it used to exit 2 for
  a source that is not a `.ladruno`.

The six stills test files that VA6b pinned to the VTK backend no longer force a
backend: their VTK-scene tests run where VTK is the platform's still path
(Linux and macOS, G2), and app-path tests on a `.ladruno` source sit beside them.
