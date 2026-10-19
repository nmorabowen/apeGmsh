### ADDED — apeGmshViewer contours Gauss-point results from a `.ladruno` (#2187)

The app now reads the Gauss-point buckets of a fork-written `.ladruno`, the
`RESULTS/ON_ELEMENTS/<result>/<classTag>-<ClassName>[...]` groups whose
`COLUMN_MAP` lists one block per Gauss point (fork schema §7.2). A continuum
element's stress or strain bucket becomes a Gauss group, as a native
`results.h5` group does (checked on a fork-written quad plate). Its components keep the file's own names, spelled
`<result>/<label>` (for example `stress/sigma11`), so they never collide with a
nodal component of the same label. The contour lists them as the field
`stress` with the component `sigma11`, as it lists nodal results. The natural coordinates of the Gauss points
come from the file's `QUADRATURE/GP_PARAM`, never from a table in the app. A
Gauss contour and its extrapolation to nodes work on a paired `.ladruno` as on
a native file. Element values are served only when `INFO/PROVENANCE` stamps
`element_index_space = "fem"`.

A Gauss bucket that disagrees with itself is refused by name, with the reason,
and is never read. That covers a `GAUSS_ID` count that differs from
`GP_PARAM`, a `NUM_COLUMNS` that is not the sum of `MULTIPLICITY * NUM_COMP`, a
`DATA` step axis that disagrees with `TIME` and `STEP`, and a bucket with no
Gauss point. Fibers, layers, element-level rows, beam section stations and the
domain and region buckets stay listed as unread. Beams keep their line-station
path. Natural coordinates are offered for the extrapolation only on
`PARAM_DOMAIN = "[-1,1]"`; a Bezier or simplex domain reads its Gauss values but
warns that it cannot extrapolate them to nodes.

The Gauss and the line-station reads now share one `COMP_NAMES` parser. An
empty label (for example a trailing comma) refuses the bucket; the line-station
read used to drop it and read the bucket. No committed fixture has one.
