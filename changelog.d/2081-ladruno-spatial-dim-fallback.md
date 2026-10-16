### FIXED — a `.ladruno` without `INFO/SPATIAL_DIM` takes its dimension from the node coordinates, or refuses the reads that need it (#2081)

`LadrunoReader` used to read a file that lacked `INFO/SPATIAL_DIM` as 3-D.
A 2-D truss in a frame model (ndf 3) writes `globalForce` as `P<node>_<dof>`,
so its `P1_3` is the node-1 rotation slot. The reader listed and read that
column as `nodal_resisting_force_z`, and `opensees_model()` reported
`ndm = 3`. The pinned fork always writes the attribute, so this reached only
files from other writers.

The reader now takes the dimension from `INFO/SPATIAL_DIM`, else from the
width of `MODEL/NODES/COORDINATES` when every stage agrees on 1, 2 or 3.
When neither gives it, the `nodal_resisting_*` element reads raise
`NodalForceNotRecorded`, the element component listing leaves those names out
with a `UserWarning`, and `opensees_model()` raises `ValueError`. Each message
names the file. Raw token reads such as `globalForce` and `basicForce` are
unchanged.
