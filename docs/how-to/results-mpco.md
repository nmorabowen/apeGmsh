# Get results via MPCO (STKO)

Record an OpenSees run into a STKO `.mpco` HDF5 file and read it back with
`Results.from_mpco`. Reach for this when you live in the **STKO ecosystem**
(or run big parallel jobs) and want STKO's battle-tested recorder writing
the database — including fibers, shell layers, and modal shapes, which the
classic `.out` recorders can't carry.

## Recipe

Declare a resolved recorder spec, drive the analysis under `spec.emit_mpco`,
then open the file. `emit_mpco` issues a single `ops.recorder('mpco', ...)`
on entry and flushes the HDF5 on exit — no per-stage `begin_stage` /
`end_stage` ceremony.

```python
import openseespy.opensees as ops
from apeGmsh.opensees import apeSees, OpenSeesModel
from apeGmsh.results import Results
from apeGmsh.results.spec import ResolvedRecorderSpec, ResolvedRecorderRecord

# fem = g.mesh.queries.get_fem_data(dim=3)   # from the meshed session

# Typed bridge: declare model, materials, elements, supports, patterns.
# MP constraints declared via g.constraints.* auto-emit. Loads are opt-in:
# import each g.loads case into a pattern with p.from_model("<case>").
# Masses and support fixities ARE re-declared on the bridge (ops.mass / ops.fix).
ops_bridge = apeSees(fem)
ops_bridge.model(ndm=3, ndf=3)
# ... materials, elements, fix, mass, pattern ...

# Persist the canonical two-zone model.h5 — from_mpco needs it (model_h5=).
ops_bridge.h5("model.h5")

# Resolve WHAT to record (target by PG name, never raw tags).
spec = ResolvedRecorderSpec(
    fem_snapshot_id=fem.snapshot_id,
    records=(
        ResolvedRecorderRecord(
            category="nodes", name="top",
            components=("displacement_x", "displacement_y", "displacement_z"),
            dt=None, n_steps=None,
            node_ids=fem.nodes.get_ids(pg="Top"),
        ),
        ResolvedRecorderRecord(
            category="gauss", name="body",
            components=("stress_xx",),
            dt=None, n_steps=None,
            element_ids=fem.elements.get_ids(pg="Body"),
        ),
    ),
)

# Drive the run; MPCO writes ONE file with every stage inside.
with spec.emit_mpco("run.mpco"):
    ops.analysis("Transient")
    for _ in range(n_steps):
        ops.analyze(1, dt)

# Read it back — model_h5= is REQUIRED (a sibling path, not the in-memory
# model object). Omitting it raises TypeError.
results = Results.from_mpco("run.mpco", model_h5="model.h5")

disp = results.nodes.get(pg="Top", component="displacement_z")
sigma = results.elements.gauss.get(pg="Body", component="stress_xx")
```

**Prefer to run under STKO instead of in-process?** Export the deck and let
STKO write the file, then read identically:

```python
ops_bridge.tcl("model.tcl", recorders=spec, mpco=True)   # run with STKO loaded
results = Results.from_mpco("run.mpco", model_h5="model.h5")
```

The read-side `Results` API is **identical** to every other strategy —
`results.nodes.get(...)`, `results.elements.{gauss,fibers,layers}.get(...)`,
`results.stages`, `results.modes`. Nothing about querying changes because the
data came from MPCO.

## Notes / gotchas

- **`model_h5=` is required and is a path.** `from_mpco` loads the broker via
  `OpenSeesModel.from_h5(model_h5)`; MPCO files carry no `/opensees/` zone, so
  there's nothing to auto-resolve from. Omitting `model_h5=` raises `TypeError`.
- **Write `.mpco` to local disk, never to a SeaDrive / cloud-synced folder.**
  Closing the file collides with the sync client and aborts the kernel (leaving
  a stale open-for-write flag). Write to a local temp path; copy afterward if
  you must.
- **MPCO needs an STKO-built openseespy.** Vanilla `openseespy` distributions
  don't ship the MPCO recorder; `emit_mpco.__enter__` raises a `RuntimeError`
  with a remediation pointer. If you don't have STKO's bundled Python, use
  native domain **capture** (`spec.capture(...)` → `Results.from_native`) for
  the same fibers/layers/modal coverage without MPCO.
- **Loads are opt-in.** `g.loads.*` cases do not auto-emit; import each into a
  pattern with `p.from_model("<case>")`. Because nothing auto-emits, there is no
  double-count trap.
- **Multi-partition runs:** pass any one part to `from_mpco("run.part-0.mpco",
  model_h5="model.h5")` — `.part-N` siblings are auto-discovered and merged.
  Pass `merge_partitions=False` to read only the named partition.
- **MPCO vs native capture:** use MPCO for STKO interoperability and parallel
  runs; use `spec.capture(...)` when you control a plain openseespy build and
  want apeGmsh's native HDF5 (same read API, broadest coverage, no STKO build).

## See also

- Concept: [Obtaining results — five strategies](../internal_docs/guide_obtaining_results.md)
  (Strategy C₁/C₂) and the [recorder reference](../internal_docs/guide_recorders_reference.md).
- How-to: [Choose a results strategy](index.md#results) ·
  [Plot a deformed shape or contour](../internal_docs/guide_results.md).
- API: [`apeGmsh.results.Results`](../api/results.md) — `from_mpco`, the
  composite query surface, and slab shapes.
