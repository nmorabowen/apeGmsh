### ADDED — bridge declaration provenance in `model.h5` (ADR 0112 D3, program slice V2d, #1307, #1378)

`apeSees._register` now records where in the user's source each primitive
was declared, through the V2c capture helper: one record per user call,
keyed `opensees/<kind>/<name|#k>` where `<kind>` is the OpenSees command
(`element`, `uniaxialMaterial`, `pattern`, ...). The series and patterns
the bridge synthesises inside a verb the user called (the HOLD series and
pattern of `s.support`, the series and pattern of `imposed_displacement`,
which gains a `name=` kwarg) get records of their own under
`<verb>:<owner>[/<role>]` keys (`opensees/timeSeries/support:<stage>/hold`,
`opensees/pattern/imposed_displacement:<name>`), pointing at the verb call,
never taking a `#k` number, with the new `records/origin = "synthesised"`
column (`/provenance` schema 1.1.0, additive; a 1.0.0 file reads as
`origin = "user"`). `_compose_model_h5` writes `/provenance` after
`/opensees`: the snapshot's records (the session's, which `FEMData.from_h5`
carries forward from a source file) followed by the bridge's, files and
sites deduplicated and `seq` continued. The bridge owns the `opensees/`
zone: a snapshot reloaded from a bridge-written file drops that file's
`opensees/` records before the new bridge's are appended, so the
"analyse" script (`FEMData.from_h5` → `apeSees` → `.h5()`) neither
collides on a repeated name nor keeps a stale record. `model_hash` and
`fem_hash` read neither table, so they are unchanged. The replay writers
(`OpenSeesModel.to_h5`, `ModelData.write`) copy `/provenance` and
`session_id` forward: byte-equal into the source's directory, re-based to
the new file's own `@base_dir` elsewhere. A `mass_from_model()` model
passes this path. A stub snapshot (no neutral zone) writes no
`/provenance`. The unconditional bridge write (D1) is not in this change:
it waits for V2b's artifact path (#1331).
