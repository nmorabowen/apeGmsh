### ADDED — `OpenSeesModel.build(..., flat=True)` replays a partitioned archive as the bridge's flat deck (#1832)

A partitioned archive records the ADR 0027 INV-5 auto-chain: the
`numberer ParallelPlain` / `RCM` and `system Mumps` / `UmfPack` runtime
conditionals, stored with `numberer_runtime_fallback` /
`system_runtime_fallback` attrs. `OpenSeesModel.build("tcl" | "py")` replays
them, so its deck matched the partitioned bridge deck and never
`apeSees.tcl(flat=True)`. The new keyword-only `flat: bool = False` mirrors
the bridge's `flat=True`: it drops exactly the numberer and system lines that
carry a recorded fallback, in the global chain and in every stage's. A
numberer or system the user declared is kept, and an archive with no
fallback replays the same deck either way. The default output is unchanged.
`flat=True` is refused with `ValueError` for the `"live"` and `"h5"` targets.

The R1b round-trip oracle now replays with `build("tcl", flat=True)`, and its
`test_h5_partitions_roundtrip::_build_bridge` row moves from class C to class
A (integral float text only).
