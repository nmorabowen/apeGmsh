### ADDED — `/geometry` and `/provenance` zone keys, `FEMData.session_id` (ADR 0112 V2a, #1304)

The `/geometry` and `/provenance` root zones are registered with their own
version keys, `/meta/geometry_schema_version` and
`/meta/provenance_schema_version` (both `1.0.0`), and their layouts are
specified in `architecture/h5-schema.md`, with an int32-only integer policy
whose writers refuse on overflow. The legacy envelope never stands in for
these keys. `FEMData` now carries a uuid4 `session_id`, which every
neutral-zone writer stamps as `/meta/session_id` and `FEMData.from_h5` reads
back, so a `model.h5` pairs with its sibling `<stem>.geometry.h5`. No hash
reads it: adding or deleting the new zones, or changing `session_id`, leaves
`snapshot_id`, `fem_hash` and `model_hash` unchanged. An unpickled snapshot
from before this change gets a fresh id instead of silently losing its neutral
zone on `apeSees.h5()`. The two new zones start at their own 1.0.0.
