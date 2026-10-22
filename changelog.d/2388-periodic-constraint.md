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

Ties along two directions (the sides of a square, the faces of a box)
meet at corner and edge nodes. They are merged so that each group of tied
nodes gets one master; a corner is never constrained twice, and a slave is
never chained to another slave. Under the Transformation handler either
of those would solve the whole model wrong without a warning. A cycle,
or ties on different DOF sets that meet, raises. A periodic tie added
after the first `get_fem_data` that shares a node with an earlier tie
also raises.
