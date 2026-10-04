### CHANGED — old model.h5 files open again: a compatibility floor per zone replaces the two-version window (ADR 0113, #1303)

The readers (`FEMData.from_h5`, `g.compose`, `OpenSeesModel.from_h5` and
`Results` on a native results file, including its embedded `/model` and
`/opensees`) used to accept only the current and the previous minor of each
zone, so every schema bump expired files they could still parse. Each zone now
has a floor, and a file opens when it is on the reader's major and its minor
lies between that floor and the current one; the patch is ignored. The floors
are neutral 2.10.0, opensees 2.11.0 and results 1.0.0, and the new `/geometry`
and `/provenance` zones start at 1.0.0. Files the window had expired, such as
neutral 2.32.x and opensees 2.19.x or 2.20.x, are readable again, back to the
floors. A file below the floor is refused as "too old" with a message naming
the supported range, for example "supports 2.10.x–2.34.x"; a newer minor is
still refused (ADR 0023 INV-4). The floors are writer-owned constants
(`NEUTRAL_SCHEMA_FLOOR`, `emitter.h5.SCHEMA_FLOOR`, `RESULTS_SCHEMA_FLOOR`),
read through `schema_version.reader_floor(zone)`.
