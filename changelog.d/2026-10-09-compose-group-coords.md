### FIXED — compose places group, label, selection and phantom coordinates (#1592)

After `compose(translate=, rotate=)`, and so after `Assembly.bridge` with a
placed instance, `fem.nodes.physical.node_coords(...)`,
`fem.nodes.labels.node_coords(...)` (and their element-side twins) and a mesh
selection's `node_coords` returned the module's source coordinates while
`nodes.coords` returned the placed ones. A v1 `anchor=` on a translated module
resolved to the wrong centroid. A node-to-surface constraint's
`phantom_coords` was also left at the source position, so the bridge declared
its phantom nodes there. The merge now moves all of these caches by the same
`R x + t` it applies to the node table.
