### FIXED — `OpenSeesModel.from_h5(path).build(...)` replays the archive's MP constraints instead of running untied; phantom-node couplings refuse (#2405, #2411)

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
- A model with a `node_to_surface` or `node_to_surface_spring` coupling, or
  with a mixed-ndf interface, now raises `BridgeError` on deck replay, citing
  #2411. Its lines tie through phantom nodes, and the archive stores no
  phantom-node coordinates. Emit such a model through
  `apeSees(FEMData.from_h5(path))`.
- The R1b corpus gains 16 rows with FEMData-side MP constraints. Its replay
  now treats `ReplaySkippedStreamWarning` as an error.
