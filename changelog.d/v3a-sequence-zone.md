### ADDED — `/sequence`: the readable stages of a `model.h5` (ADR 0112 D2b, V3a #1640)

Every `model.h5` written through the composer (`apeSees.h5`,
`OpenSeesModel.to_h5`, `ModelData.to_h5`) now carries a root `/sequence`
zone with two flat tables. `stages` has one row per stage: its name, the
analysis chain (`analysis`, `integrator`, `algorithm`, `test`, `system`,
`numberer`, `handler` and their arguments as JSON), `steps`, `dt`,
`set_time` and `domain_change`. `changes` has one row per change a stage
makes (`add pattern`, `add fix`, `remove element`, `set rayleigh`, ...),
with the declaration's name and a count. A deck without stages gets one
implicit stage (`@staged` 0). Each row names the `/opensees` path it was
read from, because the zone is derived by reading the just-written
`/opensees` back, never from a second registry; a rewrite re-derives it.
Its version key is `/meta/sequence_schema_version = "1.0.0"`. No hash
reads it. Integers are int32, and a larger value refuses loudly.
