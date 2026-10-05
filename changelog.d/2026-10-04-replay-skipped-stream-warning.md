### FIXED — deck replay warns when it leaves a stream out (program slice B2-1, D8, #1412)

`OpenSeesModel.from_h5(path).build("tcl"|"py"|"live")` re-emits only the
`/opensees` deck records and the `g.reinforce` ties. The neutral-zone streams
that the forward bridge fans out at emit time were dropped without a word:
equalDOF, rigidLink, rigidDiaphragm and node-to-surface couplings, penalty and
equation ties, `g.embed` ties, contacts and contact planes, and the phantom node
and equalDOF of a mixed-ndf interface. A replayed deck that needed any of them
ran with the nodes unconstrained. Replay now raises one
`ReplaySkippedStreamWarning` (in `apeGmsh.opensees._internal.compose`) before it
emits anything, naming each skipped stream with its count, for example
`fem.nodes.constraints (equal_dof: 1)`. Constraints that the forward emit writes
as `element` lines (RBE2 `kinematic_coupling`, `rigid_body` with `as_element`,
`distributing` RBE3, `penalty_al` ties) replay and do not warn. Neither do
constraints a staged archive's stage blocks replay (matched by kind and nodes,
not by name, since a `tied_contact` slave row carries no name), or the `build("h5")`
target, whose neutral zone keeps them. A model without these streams replays
silently. For a faithful deck, load `FEMData.from_h5(path)` and emit it through
`apeSees(fem)`.
