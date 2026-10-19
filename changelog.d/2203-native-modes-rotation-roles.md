### FIXED — native mode files no longer write a u-p pore pressure as a rotation (#2203)

`ParallelModalResult.to_native(path, fem, *, model_h5=None)` and `EigenResult.to_native(path=None, *, model_h5=None)` now write a node's `rotation_*` only where the model's DOF roles say that slot holds a rotation. They follow the rule `to_ladruno` follows (#2178).

- **Before:** a 3-D u-p node in a deck padded to ndf 6 had its pore pressure written as `rotation_x`. A 2-D frame lost its real `rotation_z`, and `EigenResult` wrote rotations on element-less nodes of a u-p model.
- **With `model_h5=`:** `ParallelModalResult.to_native` lays each node's columns out by its own ndf and keeps a rotation only where the roles allow it. So a 2-D frame keeps `rotation_z`, and a u-p node writes 0.0 there. `EigenResult.to_native` always has the roles: by default it takes them from the bridge's own archive.
- **Without `model_h5=` (`ParallelModalResult` only), a plain 3-D frame writes exactly what it wrote before.** The rules, in order:
  1. The archive `modal_deck` leaves at `default_artifact_dir()/<fem.model_name>.h5` is used when its `/meta/lineage` `fem_hash` is `fem.snapshot_id` and its elements touch every fem node on a solid cell. Any other file there is ignored, with one `StaleModelArchiveWarning`. That includes an archive another bridge on the same mesh and `model_name` wrote last.
  2. A 3-D ndf-6 deck on a fem with no solid (3-D) cell writes DOFs 4-6 as rotations, since without a solid there is no u-p element.
  3. Otherwise each node's own ndf comes from `fem`. A node nothing settles writes no rotation, with one `RotationRolesUnavailableWarning` that says to pass `model_h5=`. That covers a 2-D node of ndf 3 (frame or u-p), and a 3-D node beside solids whose ndf `fem` does not carry.
