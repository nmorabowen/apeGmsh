### FIXED - apeGmshViewer places `.ladruno` section stations only where `PARAM_DOMAIN` says, and warns when it spaces them evenly (#2472)

The viewer's `.ladruno` line-station read took the station xi of a
`section.force` / `section.deformation` bucket from column 0 of
`QUADRATURE/GP_PARAM` without reading the group's `@PARAM_DOMAIN`. When the
`GAUSS_ID`s did not index GP_PARAM, or the group had no GP_PARAM, it fell
back to evenly spaced stations with no warning. A force-based beam's
Legendre or Lobatto points then drew at the wrong places along the beam,
and nothing said so.

Now:

- a GP_PARAM with no `PARAM_DOMAIN`, or with any domain other than
  `"[-1,1]"`, is refused by name with a `LadrunoError`, as Python's
  `_gp_param_for` refuses it (`"[0,1]"` and `"bary"` included);
- a GP_PARAM row that is not a finite xi in [-1, 1] is refused by name;
- every evenly spaced fallback warns, naming the bucket and the reason. The
  warning goes through `readLineStations`' `warningsOut`, and through
  `availableLineStations` at open, which is the path the app's warnings
  reach. At open, a refused section bucket is left out and says why;
  element-level buckets with more than two stations also warn at open.

Reads of every committed `.ladruno` are byte-identical before and after.
