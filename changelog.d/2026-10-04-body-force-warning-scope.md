### FIXED — `WarnBodyForceDoubleCount` fires only on a body-force case, not on a collinear footing load (#1338)

The self-weight double-count guard warned whenever a `p.from_model(case)`
import put a nodal load collinear with a continuum element's `body_force=`
on the same nodes. A vertical footing line load on a soil block that carries
its weight through `body_force` is exactly that, yet nothing is counted
twice; two readability-workshop writers restructured their models to
silence it. Resolved nodal loads carried no record of what produced them, so
collinearity was all the guard could read.

`NodalLoadRecord` now carries `source`, the `kind` of the definition the
resolver reduced it from (`gravity`, `body`, `line`, `surface`, `point`,
`point_closest`, `face_load`; the vocabulary is
`apeGmsh._kernel.records.NodalLoadSource`). The guard counts a collinear
overlap only for a body source (`g.loads.gravity`, `g.loads.volume`) and
stays silent for a boundary load; the message names the source. A record
whose source is unknown — a `model.h5` older than neutral 2.35.0, or a
record synthesized without a definition — still warns, as before, and the
message says why; a source outside the vocabulary raises `BridgeError`.

The column is persisted: neutral schema **2.35.0** adds `source` to
`/loads/nodal/{pattern}` (additive, presence-probed on read; older files
decode `None`), with the outgoing 2.34 and the new 2.35 corpus files.
