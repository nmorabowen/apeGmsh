### Added

- `EigenResult.to_ladruno(path=None)` and `ParallelModalResult.to_ladruno(path, model_h5=)` write the modes as a stamped `.ladruno` (#2178).
  - **Layout:** one `KIND="eigen"` stage in the fork recorder's own `modesOfVibration` layout. `MODEL/NODES` comes from the bridge's fem, and `INFO/PROVENANCE` pairs the file with the model archive. `EigenResult` writes that archive beside the file as `<stem>.h5`. A target that is the archive, or any `.h5`, is refused.
  - **Rotation slots:** these follow the model's DOF roles, so a u-p node's pore pressure is never written as a rotation. A slot the roles cannot settle is written as 0.0, with one `RotationSlotUnresolvedWarning`.
  - **Reading back:** `Results.from_ladruno(path, model_h5=)` reads the modes back as `mode` stages.
  - **No element buckets:** none are written, and `PROVENANCE@unsupported` names `elements`.
