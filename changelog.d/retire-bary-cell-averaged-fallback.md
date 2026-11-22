### CHANGED — Windows Gauss stills of BezierTet10 and BezierTri6 fields extrapolate to the nodes again (#3085)

From #2942 and #2950, `results.render` on Windows asked apeGmshViewer for the
element mean (`cell_averaged`) whenever a Gauss contour held only BezierTet10
or BezierTri6 groups, because the app had no natural coordinates for their
barycentric (`"bary"`) Gauss points. Since #2706 the app converts those
points. The still now asks for `node` whenever any group of the field has
several Gauss points, whatever its element class. A BezierTet10 or BezierTri6
field is drawn extrapolated to the corners, like any other multi-point field.

The app decides the fallback of a group it has no natural coordinates for:
a group with no `QUADRATURE/GP_PARAM`, or a `PARAM_DOMAIN` or `"bary"` frame
it cannot convert. It draws that group as its element mean and names it in
the manifest's `topology_fallback` and in the banner, as the owner ruled on
#2924 item 2 (2026-11-13). This holds in a mixed field and in a field made
only of such groups.

One picture keeps its values but gains the label. A BezierTet10 or
BezierTri6 field whose file has no `GP_PARAM` used to be asked as
`cell_averaged` and drawn `element_mean_averaged` with no label. It is now
asked as `node` and drawn as the same element means, labelled as the
fallback.
