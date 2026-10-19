### FIXED — native mode files no longer write a u-p pore pressure as a rotation (#2203)

`ParallelModalResult.to_native(path, fem, *, model_h5=None)` and `EigenResult.to_native(path=None, *, model_h5=None)` now write a node's `rotation_*` only where the model's DOF roles say that slot holds a rotation. They follow the rule `to_ladruno` follows (#2178).

- **Before:** a 3-D u-p node in a deck padded to ndf 6 had its pore pressure written as `rotation_x`. A 2-D frame lost its real `rotation_z`, and `EigenResult` wrote rotations on element-less nodes of a u-p model.
- **With `model_h5=`:** `ParallelModalResult.to_native` lays each node's columns out by its own ndf and keeps a rotation only where the roles allow it. So a 2-D frame keeps `rotation_z`, and a u-p node writes 0.0 there. `EigenResult.to_native` always has the roles: by default it takes them from the bridge's own archive.
- **Without `model_h5=` (`ParallelModalResult` only):** when the deck's DOF list reaches a rotation slot (2-D ndf 3, 3-D ndf 6), no `rotation_*` is written. One `RotationRolesUnavailableWarning` says to pass `model_h5=`. 3-D frame decks exported without the archive lose their rotations until it is passed.
