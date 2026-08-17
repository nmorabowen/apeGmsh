# partitioned-frame

A 2-storey x 2x2-bay rigid-jointed space frame, partitioned into 4
ranks via Gmsh-native METIS, with one rigid-floor diaphragm per storey
(cross-partition replication, ADR 0027 INV-1/INV-2), emitted as an
OpenSeesMP-ready Tcl deck.

Teaches:

- `sess.mesh.partitioning.partition(n_parts=...)` (Gmsh-native METIS).
- Injecting a `NodeGroupRecord` (rigid diaphragm) directly into the
  broker when the physical-group source isn't a `Part`.
- `apeSees(fem).tcl(...)` auto-emitting the parallel numberer/system
  pair (`ParallelPlain`/`RCM`, `Mumps`/`UmfPack`) once
  `len(fem.partitions) > 1`.

Requires: `mesh-only` — emit-only, no solver needed to verify.

Run:

```
python partitioned_frame.py            # emit only
python partitioned_frame.py --run      # also try mpiexec, if present
python verify.py
```

`verify.py` re-builds and re-emits the deck itself (mesh-only, fast)
and checks node/element/partition counts plus that the Tcl file
exists. See `manifest.json` for the oracle metrics.
