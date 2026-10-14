### FIXED — `from_h5(...).build()` declares materials before the wrappers that use them (#1814)

`OpenSeesModel.from_h5(...).build("tcl" | "py" | "live")` emitted the
materials in the archive's group order, which HDF5 iterates by name, so
`uniaxialMaterial Parallel 3 1 2` came before `Steel01 1` and the replayed
deck aborted ("no existing material with tag 1"). Replay now emits each
family's materials in the order the bridge emitted them: the archive's
`/opensees/program` records every material call and its declaration
(opensees 2.31.0), and `/opensees/decls` names the declaration's tag. A
material no call joins to (minted by a side pass, with no declaration)
follows the joined ones in tag order. An archive below 2.31.0 records no
joins and replays in tag order, the bridge's registration order; only a
wrapper `register`-ed before its members cannot be recovered there. The
`materials()` accessor and the `h5` rewrite keep the store order.
