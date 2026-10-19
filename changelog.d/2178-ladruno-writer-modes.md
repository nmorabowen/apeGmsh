### Added

- `EigenResult.to_ladruno(path=None)` and `ParallelModalResult.to_ladruno(path, model_h5=)` write the modes as a stamped `.ladruno`: one `KIND="eigen"` stage in the fork recorder's own `modesOfVibration` layout, with `MODEL/NODES` from the bridge's fem and `INFO/PROVENANCE` paired with the model archive (`EigenResult` writes it beside the file as `<stem>.h5`). `Results.from_ladruno(path, model_h5=)` reads them back as `mode` stages. No element buckets are written; `PROVENANCE@unsupported` names `elements` (#2178).
