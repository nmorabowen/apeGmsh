### ADDED — Windows stills draw `von_mises_stress` / `von_mises_strain`, and a tet10 Gauss still asks `cell_averaged` (#2942)

On Windows, `results.render(..., component="von_mises_stress")` and
`component="von_mises_strain"` were refused, although apeGmshViewer draws
both from a Gauss tensor. The still now names the derived scalar as a
component of the element field that holds the tensor, for example
`{field: "stress", component: "von_mises_stress", source: "element"}`.
The field must hold every recorded component of the tensor at one Gauss
mode, or the still is refused. A stage that derives the scalar at the
nodes is also refused, and so are the other derived names, such as
`tresca_stress`.

A BezierTet10 Gauss still asked for `gauss: "node"`, which the app refuses
because the element's Gauss points are barycentric. BezierTet10 groups now
ask for `cell_averaged`: the mean of each element's Gauss points, averaged
at the nodes (owner ruling 2026-11-13, #2924 item 2).
