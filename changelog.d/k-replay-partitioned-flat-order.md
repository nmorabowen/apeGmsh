### FIXED — a partitioned archive replays in the bridge's flat order: series above the `geomTransf`, the MPCO region just above its recorder (#2277, K R1b-S3b)

`OpenSeesModel.from_h5(path).build("tcl" | "py")` on a partitioned archive
used to replay in the fixed category order. That archive's
`/opensees/program` records the ADR 0027 partitioned emit, which writes
every declaration above the partition blocks, so it could not merge by emit
index the way #2193 does for a single-domain archive. The replay wrote the
`geomTransf` above the `timeSeries` and the MPCO `-R` region above the
`Node` and `Element` recorders, where `apeSees.tcl(flat=True)` writes the
opposite.

A partitioned archive's replay now sorts each unit by the section of
`_emit_flat` that writes it, then by emit index: nodes, the 4a
declarations, the gated elements, the builder-scoped declarations, the
`geomTransf`, the elements, the fixes, masses, regions and commands, the
constraints and ties, the initial stress, then the patterns and recorders.
Two units take their section from the archive:

- An analysis-chain component is a user declaration (4a) when the program
  wrote it above the first partition block. Otherwise it is the
  auto-emitted constraint handler (7c).
- A region whose declaration is a recorder (an MPCO filter region) takes
  that recorder's emit index, so it replays just above its recorder.

The replay of the staged, partitioned recording frame (the real-FEMData
twin of `_broad_partitioned`) now equals `ops.tcl(flat=True)` line for
line. A unit kind that no section places, a recorder region whose recorder
no call wrote, and a program that opens partition blocks only inside
stages each raise.

A single-domain archive replays exactly as before. A test pins every deck
target over every unpartitioned round-trip model and every `schema_corpus`
file.

Not fixed here, and replayed as before:

- `_emit_flat` writes a global pattern after the recorders (a pattern
  waits on its series in the topological order). A partitioned program
  writes it inside the rank blocks, above them, and no emit index recovers
  that order, so such a deck still replays the pattern first.
- When a stage claims an MPCO recorder, `_emit_flat` writes its region
  inside the stage block. A partitioned archive stores that region with
  the global regions, so the replay writes it there.
