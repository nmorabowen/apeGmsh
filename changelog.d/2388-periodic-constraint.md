### ADDED — `g.constraints.periodic(master, slave, axis=...)`: tie two opposite boundaries node for node (#2388)

`equal_dof` ties co-located nodes only. A 1-D site-response column
therefore tied its left and right sides with one
`ops.equation_constraint` per elevation and DOF. `periodic` pairs the
nodes of two sides (labels, physical groups or parts) separated along
`axis`. Each slave node pairs with the master node whose other
coordinates match within `tolerance`, and each pair emits one
`equalDOF master slave *dofs`, through the same records `equal_dof`
produces. This covers the tied sides of a column (`axis="x"`) and two
opposite faces of a 3-D box.

The pairing must be complete and one-to-one. A node of either side with
no partner (sides meshed differently), a master matched twice, or a node
on both sides raises at `get_fem_data` and names the first offenders.
