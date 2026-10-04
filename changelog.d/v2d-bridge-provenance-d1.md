### ADDED — bridge declaration provenance in `model.h5` (ADR 0112 D3, program slice V2d, #1307)

`apeSees._register` now records where in the user's source each primitive
was declared, through the V2c capture helper: one record per user call,
keyed `opensees/<kind>/<name|#k>` where `<kind>` is the OpenSees command
(`element`, `uniaxialMaterial`, `pattern`, ...). A primitive the bridge
synthesises inside a verb the user called (the HOLD series of `s.support`,
the series and pattern of `imposed_displacement`) adds no record of its
own. `_compose_model_h5` writes `/provenance` after `/opensees`: the
snapshot's records (the session's, which `FEMData.from_h5` carries forward
from a source file) followed by the bridge's, files and sites deduplicated
and `seq` continued. `model_hash` and `fem_hash` read neither table, so
they are unchanged. The replay writers (`OpenSeesModel.to_h5`,
`ModelData.write`) copy `/provenance` and `session_id` forward: byte-equal
into the source's directory, re-based to the new file's own `@base_dir`
elsewhere. A `mass_from_model()` model passes this path. The unconditional
bridge write (D1) is not in this change: it needs V2b's artifact path.
