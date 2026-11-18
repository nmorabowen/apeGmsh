### CHANGED — apeGmshViewer draws a `node` contour's Gauss groups with no natural coordinates as element means, per group, and says so (#2940)

A Gauss result whose points have no natural coordinates the app can read, such
as a BezierTet10's barycentric points in a `.ladruno` (PARAM_DOMAIN `"bary"`),
was refused when asked as a `node` contour: the whole field was drawn as no
data. Now, when that is the only reason, each such group draws
`cell_averaged`: each element gives its mean to its corners, averaged at the
nodes. This follows the owner's rulings on #2924. Every other group of the
field is still extrapolated (`gauss_node`), so a field mixing hex or quad
groups with BezierTet10 groups keeps its extrapolated values where it can.
The notices banner names the groups drawn as means and why. The manifest
keeps `contour.topology` `gauss_node` and adds `contour.topology_fallback`
with the topology asked, the one drawn, the groups and the reason. Every
other refusal stays a refusal: an element type with no corner shape
functions, natural coordinates with too few columns or a non-finite value,
and `node_discrete`, which asks for no averaging.
