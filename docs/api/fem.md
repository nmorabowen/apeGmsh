# FEM Broker — `FEMData`

Solver-agnostic snapshot returned by `g.mesh.queries.get_fem_data(dim)`.
Composite of `.nodes`, `.elements`, `.info`, `.inspect`, `.mesh_selection`.
The `.physical`, `.loads`, `.masses`, `.constraints`, and `.sp` views
are reached through `.nodes.*` / `.elements.*` rather than living
directly on `FEMData`.

Also surfaces `snapshot_id` plus `to_native_h5` / `from_native_h5` /
`from_mpco_model` for binding and round-tripping with Results.

::: apeGmsh.mesh.FEMData
