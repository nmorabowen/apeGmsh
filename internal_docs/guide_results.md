# Results — Post-processing Container

## What Results is

After the solver finishes, you have arrays of displacements, stresses, and reactions sitting in some on-disk file format. The `Results` class is the bridge between those raw outputs and the rest of the apeGmsh ecosystem: probes, plots, the interactive viewer, programmatic queries.

`Results` is a **backend-agnostic post-processing container**. It mirrors the `FEMData` composite shape — `results.nodes`, `results.elements.gauss`, `results.elements.fibers` — and uses the same `pg=` / `label=` / `ids=` selection vocabulary, so once you know how to query a mesh you already know how to query results on that mesh. The container itself does not own arrays; it lazily reads from a backing reader (native HDF5, MPCO, or a transcoded recorder cache).

The class lives at `apeGmsh.results.Results` and is re-exported from the package root:

```python
from apeGmsh import Results
```

For the analysis side — *how the file got written* — see [`guide_obtaining_results.md`](guide_obtaining_results.md). For *how to slice and query* an open `Results`, see [`guide_results_filtering.md`](guide_results_filtering.md). This guide covers construction, stages, modes, binding, and the lifecycle.


## Construction

There are three constructors, one per supported backend:

### `Results.from_native` — apeGmsh HDF5

For files written by domain capture (Strategy B) or by the recorder transcoder (Strategy A₁/A₂/A₃ via the cache).

```python
results = Results.from_native("run.h5")
results = Results.from_native("run.h5", fem=fem)   # explicit bind override
```

A native file embeds a frozen `FEMData` snapshot under `/model/`. If `fem=` is omitted, that snapshot is the bound FEMData. If `fem=` is provided and the file embeds a snapshot, the two `snapshot_id` hashes must match.

### `Results.from_mpco` — STKO HDF5

For files written by the STKO MPCO recorder (Strategy C₁/C₂).

```python
results = Results.from_mpco("run.mpco")

# Multi-partition (parallel runs) — auto-discover .part-N siblings
results = Results.from_mpco("run.part-0.mpco")

# Or pass an explicit partition list
results = Results.from_mpco(["run.part-0.mpco", "run.part-1.mpco"])
```

MPCO doesn't embed a full FEMData; the reader synthesises a partial one from the file's `MODEL/` group. Pass `fem=` to bind your session-side FEMData (recommended — it carries labels and Parts that the synthesised one doesn't).

### `Results.from_recorders` — classic OpenSees recorders

For `.out` / `.xml` files written by `g.opensees.export.tcl(..., recorders=spec)`, `g.opensees.export.py(..., recorders=spec)`, or `spec.emit_recorders(...)` (Strategy A₁/A₂/A₃).

```python
results = Results.from_recorders(spec, "out/", fem=fem)

# Multi-stage — pick the stage you want by id
gravity = Results.from_recorders(spec, "out/", fem=fem, stage_id="gravity")
```

`from_recorders` transcodes the recorder files into a cached HDF5 (`writers/_cache.py`), keyed on file mtimes + spec `snapshot_id`, and then opens that through `from_native`. Subsequent calls with unchanged inputs return the cached HDF5 directly. `fem=` is required — the spec's `snapshot_id` must match.


## Stage scoping

Every results file holds one or more **stages**. A stage is one analysis segment (a gravity push, a transient run, a single eigenmode) tagged with a `kind` (`"static"`, `"transient"`, `"mode"`) and a name.

A top-level `Results` carries all stages. Reads disambiguate automatically when there is exactly one stage; with multiple stages, you have to pick:

```python
results.stages                    # list[StageInfo]

gravity = results.stage("gravity")     # stage-scoped Results
sigma = gravity.elements.gauss.get(component="stress_xx", pg="Body")

# Stage-scoped instances expose stage metadata as properties:
gravity.kind        # "static"
gravity.name        # "gravity"
gravity.n_steps     # int
gravity.time        # ndarray
```

Calling a stage-only property on an unscoped `Results` raises `AttributeError` with a hint to call `.stage(...)` first.


## Modes

Modes are stages with `kind="mode"`. The `.modes` accessor filters them and returns each as a mode-scoped `Results`:

```python
for mode in results.modes:
    print(mode.mode_index, mode.frequency_hz, mode.period_s)
    shape = mode.nodes.get(component="displacement_z")

# Stable order by mode_index
for mode in sorted(results.modes, key=lambda m: m.mode_index):
    ...
```

Mode-scoped instances additionally expose `.eigenvalue`, `.frequency_hz`, `.period_s`, and `.mode_index`. These raise on non-mode-scoped reads.


## FEM access and binding

The bound `FEMData` snapshot is available as `.fem`:

```python
results.fem            # FEMData | None
```

`.bind(other_fem)` swaps in a different FEMData (typically your session-side one, which carries labels and Parts that the embedded snapshot doesn't):

```python
results = Results.from_native("run.h5").bind(fem)
```

**No hash validation is performed by `bind`** — pairing the FEMData with the right results file is the user's responsibility. This is intentional; see the bind-contract memory note for the design rationale.


## Lifecycle

`Results` holds an open file handle through its underlying reader. Two ways to release it:

```python
# Explicit close
results = Results.from_native("run.h5")
# ... use it ...
results.close()

# Context manager
with Results.from_native("run.h5") as results:
    # ... use it ...
    pass
# closed on exit
```

On Windows in particular, an open HDF5 handle blocks the writer from re-creating the same file in a re-run loop — close before re-running a capture script that overwrites the path.


## Visualisation

```python
results.viewer()                       # blocking, in-process (default)
results.viewer(blocking=False)         # subprocess; notebook keeps running
results.viewer(title="Gravity push")
results.viewer(restore_session=False)  # ignore any saved viewer-session.json
results.viewer(save_session=False)     # don't auto-save on close
```

The blocking path opens `ResultsViewer` in-process and blocks the caller until the window closes. The non-blocking path spawns `python -m apeGmsh.viewers <path>` so the kernel can keep running; this requires that the Results was opened from disk (in-memory Results raise).

When non-blocking spawns, the parent reader is closed automatically — this lets you re-run the capture script (which deletes/recreates the file) without hitting Windows' "file in use" error. To keep querying after the spawn, re-open with `Results.from_native(path)`.

The `restore_session` flag controls whether a sibling `<results>.viewer-session.json` is loaded:
- `True` — restore silently
- `False` — ignore
- `"prompt"` (default) — open a yes/no dialog if a matching session exists

**CI / nbconvert escape hatch.** Set the `APEGMSH_SKIP_VIEWER`
environment variable to make `viewer(...)` print a skip marker and
return `None` immediately — useful when running notebooks under
`jupyter nbconvert --execute` or in CI without a display.


## Reading fields

Field reads go through the composite tree and use the same selection vocabulary as the FEM broker. A small taste:

```python
# Nodal: displacement at the top surface
disp = results.nodes.get(component="displacement_z", pg="Top")

# Element-level: Gauss-point stress in the body
stress = results.elements.gauss.get(component="stress_xx", pg="Body")

# Spatial selectors
near_load = results.nodes.get(component="displacement_z").nearest_to((0, 0, 5))
in_box = results.nodes.get(component="displacement_z").in_box(
    lo=(0, 0, 0), hi=(1, 1, 1)
)
```

Full coverage of selectors, slab dataclass shapes, time slicing, and the `.available_components()` discovery API lives in [`guide_results_filtering.md`](guide_results_filtering.md). The vocabulary (`displacement_z`, `stress_xx`, `section.fiber.stress`, etc.) is documented in [`guide_recorders_reference.md`](guide_recorders_reference.md).


## Practical workflow

```python
from apeGmsh import apeGmsh, Results
import openseespy.opensees as ops

with apeGmsh(model_name="slab") as g:
    # ... geometry, mesh, physical groups ...
    g.opensees.set_model(ndm=3, ndf=3)
    # ... materials, elements, fix ...
    fem = g.mesh.queries.get_fem_data(dim=3)
    g.opensees.build()

    # Declare what to record
    g.opensees.recorders.nodes(components=["displacement"], pg="Top")
    g.opensees.recorders.gauss(components=["stress"], pg="Body")
    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    # Run with domain capture (Strategy B)
    with spec.capture(path="run.h5", fem=fem, ndm=3, ndf=3) as cap:
        cap.begin_stage("gravity", kind="static")
        # ... static analysis ...
        for _ in range(10):
            ops.analyze(1, 0.1)
            cap.step(t=ops.getTime())
        cap.end_stage()

# Open the file
results = Results.from_native("run.h5", fem=fem)

# Query
gravity = results.stage("gravity")
disp = gravity.nodes.get(component="displacement_z", pg="Top")
sigma = gravity.elements.gauss.get(component="stress_xx", pg="Body")

# Visualise
results.viewer()
results.close()
```


## Inspection

```python
results.inspect.summary()    # multi-line human-readable summary
print(repr(results))         # same as .inspect.summary()
```

The summary lists stages, kinds, step counts, available components, and the bound FEMData. It's the first thing to print when a results file behaves unexpectedly.


## See also

- [`guide_obtaining_results.md`](guide_obtaining_results.md) — five strategies for writing results files
- [`guide_results_filtering.md`](guide_results_filtering.md) — selectors, slabs, time slicing
- [`guide_recorders_reference.md`](guide_recorders_reference.md) — what categories and components exist
- [`guide_fem_broker.md`](guide_fem_broker.md) — the `FEMData` shape that `Results` mirrors
