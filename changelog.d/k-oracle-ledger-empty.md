### CHANGED — the R1b round-trip oracle ledger is empty: the last S row's model now has a real-FEMData twin (#2195, K R1b-S3)

The oracle's last row was class S. `test_h5_decl_params::_broad_partitioned`
was the golden-corpus model `two_column_frame_partitioned / staged_partitioned
/ recording`, built on `FEMStub`, so its archive had no neutral zone and
could not be replayed. The builder now builds the same model on a real
`FEMData`: the same four nodes and two columns, the same two-rank split, both
stages and the node, element and MPCO recorders. Its archive replays, and
`OpenSeesModel.from_h5(...).build("tcl", flat=True)` equals the bridge's
`tcl(flat=True)` byte for byte. The row is deleted, so `N_WAIVED_ORACLE`
goes from 1 to 0, ahead of the KC4 deadline (2026-12-11).

The twin's non-flat deck equals the golden cell's deck except for one line.
The golden pins `OpenSeesTarget(mode="stock")` and writes the `# apeGmsh …;
backend stock` stamp. The twin pins no target and writes no stamp, because
the archive does not carry the target mode (#2278). Mutations that drop a
stage or swap the ranks change the twin's deck. Dropping a partition makes
the bridge refuse its serial `system Mumps`. A second twin adds a lumped mass
and declares `RCM` / `UmfPack`. With the mass, the corpus fails when the fix
and mass sections of the flat replay order are swapped (the #2277 review
finding). The S, orphan-row and add-row rules are now held by synthetic rows
on the `FEMStub` model.

The live net cannot run the golden twin on stock: it must keep the cell's
`ParallelRCM` / `Mumps`, and stock has neither. The net runs the second twin
in its place (`_STAND_INS`), and the known-stock-failure table stays at five
rows. A live test pins three things: the two decks differ only in the solver
pair and the `mass` lines, the stand-in runs, and the twin's own deck aborts
on that pair. Each live deck directory now holds an empty `out/`, because an
MPCO recorder writing into a missing directory aborts the deck. Tests only;
no library code changes.
