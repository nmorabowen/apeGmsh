### CHANGED — apeGmshViewer draws a `node` contour of Gauss points with no natural coordinates as `cell_averaged`, and says so (#2940)

A Gauss result whose points have no natural coordinates the app can read, such
as a BezierTet10's barycentric points in a `.ladruno` (PARAM_DOMAIN `"bary"`),
was refused when asked as a `node` contour: the field was drawn as no data.
Now, when that is the only reason, the field is drawn `cell_averaged` (each
element's mean, averaged at the nodes), as the owner ruled on #2924. The
notices banner says that `gauss_node` could not be drawn and why, and the
manifest's `contour.topology` reads `element_mean_averaged`, with
`contour.topology_fallback` naming the topology asked and the reason. Every
other refusal stays a refusal: an element type with no corner shape
functions, natural coordinates with too few columns or a non-finite value,
and `node_discrete`, which asks for no averaging.
