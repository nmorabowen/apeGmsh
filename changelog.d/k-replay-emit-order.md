### CHANGED — `OpenSeesModel.build` replays a deck in the bridge's emit order (#2193)

The `tcl`, `py` and `live` targets used to write a fixed category order:
materials, sections, transforms, integrations, series, dampings, elements,
fixes, commands, regions, and the analysis chain last. The bridge interleaves
these. It can put a section between two materials, a `damping` above a
`geomTransf`, a user-declared `system` and `analysis` above every
declaration, and `rayleigh` between two regions. A replayed deck could
therefore list the same lines as the bridge deck in another order.

The replay now merges each declaration record, each `/opensees/commands`
row, each analysis-chain component and each block (nodes, elements, fixes,
masses, ties, initial stress) by the emit index `/opensees/program` records
for it. This applies to an archive whose `/opensees/decls@program_decl` is
`"complete"` (an opensees 2.31.0 emit or later). The stage blocks' re-declared
builder-scoped kinds follow the same order. Some archives keep the fixed order:

- a file below 2.31.0, or a `to_h5` rewrite of one;
- a partitioned archive, whose program records the partitioned emit;
- a model built from the emitter's buffers.

Two cases fail closed. An archive whose program puts a builder-scoped
declaration above the gated-element bracket raises `ValueError` (ADR 0099
INV-1). An analysis-chain component that no program call wrote raises
`LookupError`.

In the R1b round-trip oracle, all 15 class-B ledger rows clear. Fourteen
re-pin to A, because only integral float text is left, and
`test_h5_solve_stamp::_frame_bridge[analysis=True]` now replays byte for
byte. `N_WAIVED_ORACLE` goes from 33 to 32.

`OpenSeesModel.build` also checks the target before `flat`: a call like
`build("tlc", flat=True)` now names the unknown target.
