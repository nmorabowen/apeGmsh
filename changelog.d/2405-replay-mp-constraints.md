### FIXED — `OpenSeesModel.from_h5(path).build(...)` replays the archive's MP constraints instead of running untied; phantom-node and equation-route ties refuse (#2405, #2411, #2416)

The replay of an `ops.h5(path)` archive left out every global `equalDOF`,
`rigidLink`, `rigidDiaphragm` and `ASDEmbeddedNodeElement` line that the
bridge's own deck carried. The writer had always stored these lines as rows of
`/opensees/constraints/*`, but the reader looked for them only inside stage
blocks. The replay warned (`ReplaySkippedStreamWarning`) and then wrote an
untied deck.

- `H5Model.mp_constraints()` reads the global rows. The deck targets (`tcl`,
  `py`, `live`) emit them where the bridge did, after the regions and before
  the equation constraints, with each row's name comment. They keep the
  bridge's line order, so the R1b round-trip oracle sees identical bytes on
  flat and partitioned archives. A partitioned archive stores a replicated
  constraint once, and the replay emits it once.
- `to_h5` rewrites keep the rows.
- An archive that lacks the rows still warns, naming the stream.
- Some models still cannot be replayed from the archive. On the `tcl`, `py`
  and `live` targets they now raise a `BridgeError` that names each kind
  and its issue, instead of replaying untied. The refusal comes before any
  emitter is made, so a refused `build("live")` leaves the domain as it
  was. Emit such a model through `apeSees(FEMData.from_h5(path))`.
  - A `node_to_surface` or `node_to_surface_spring` coupling, or a
    mixed-ndf interface (#2411). Its lines tie through phantom nodes, and
    the archive stores no phantom-node coordinates.
  - An `enforce="equation"` tie, distributing coupling or embedded node
    (#2416). The archive does not store its `equationConstraint` lines.
- The R1b corpus gains 16 rows with FEMData-side MP constraints. Its replay
  now treats `ReplaySkippedStreamWarning` as an error.
