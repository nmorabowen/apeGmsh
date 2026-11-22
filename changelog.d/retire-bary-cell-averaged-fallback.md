### CHANGED — Windows Gauss stills of BezierTet10 and BezierTri6 fields extrapolate to the nodes again (#3085)

From #2942 and #2950, `results.render` on Windows asked apeGmshViewer for the
element mean (`cell_averaged`) whenever a Gauss contour held only BezierTet10
or BezierTri6 groups. The reason was that the app had no natural coordinates
for their barycentric (`"bary"`) Gauss points. Since #2706 the app converts
those points, so the still now picks its mode by whether the app has natural
coordinates for each group, not by the element class. A field of BezierTet10
or BezierTri6 groups now asks for `node` and is drawn extrapolated to the
corners, like any other multi-point field.

A group with no natural coordinates is handled as the owner ruled on #2924
item 2 (2026-11-13). Such a group has no `QUADRATURE/GP_PARAM`, or a
`PARAM_DOMAIN` or `"bary"` frame the app cannot convert; SixNodeTri and
LadrunoLST write no `GP_PARAM`, for example. In a mixed field the app still
draws only that group as its element mean and names it in the manifest's
`topology_fallback`. A field made only of such groups asks for
`cell_averaged`, as BezierTet10 and BezierTri6 fields did before. For a
SixNodeTri-only field, the still asked for `node` and the app fell back,
so the picture is the same and the manifest no longer reports a fallback.
