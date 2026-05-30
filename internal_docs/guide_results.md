# Results — Post-processing Container

## What Results is

After the solver finishes, you have arrays of displacements, stresses, and reactions sitting in some on-disk file format. The `Results` class is the bridge between those raw outputs and the rest of the apeGmsh ecosystem: probes, plots, the interactive viewer, programmatic queries.

`Results` is a **backend-agnostic post-processing container**. It mirrors the `FEMData` composite shape — `results.nodes`, `results.elements.gauss`, `results.elements.fibers` — and uses the same `pg=` / `label=` / `ids=` selection vocabulary, so once you know how to query a mesh you already know how to query results on that mesh. The container itself does not own arrays; it lazily reads from a backing reader (native HDF5, MPCO, or a transcoded recorder cache).

The class lives at `apeGmsh.results.Results` and is re-exported from the package root:

```python
from apeGmsh import Results
```

For the analysis side — *how the file got written* — see [`guide_obtaining_results.md`](guide_obtaining_results.md). For *how to slice and query* an open `Results`, see [`guide_results_filtering.md`](guide_results_filtering.md). This guide covers construction, stages, modes, binding, and the lifecycle.

## Tasks on this page

- [Open an apeGmsh HDF5 file](#resultsfrom_native-apegmsh-hdf5) · [Open an STKO MPCO file](#resultsfrom_mpco-stko-hdf5) · [Open classic recorder output](#resultsfrom_recorders-classic-opensees-recorders)
- [Pick a stage](#stage-scoping) · [Iterate eigenmodes](#modes) · [Bind a session FEMData](#fem-access-and-binding)
- [Open the interactive viewer](#interactive-qt-viewer) · [Show the kernel-safe web viewer](#web-jupyter-viewer-kernel-safe) · [Read fields](#reading-fields) · [Inspect a file](#inspection)


## Construction

There are three constructors, one per supported backend:

### `Results.from_native` — apeGmsh HDF5

For files written by domain capture (Strategy B) or by the recorder transcoder (Strategy A₁/A₂/A₃ via the cache).

```python
from apeGmsh.opensees import OpenSeesModel

model = OpenSeesModel.from_h5("run.h5")             # the broker (ADR 0020)
results = Results.from_native("run.h5", model=model)
results = Results.from_native("run.h5", model=model, fem=fem)   # explicit bind override
```

`model=` is **required** (ADR 0020 INV-1) — omitting it raises `TypeError`. The model carries the `/opensees/` zone broker; on a Composed file the model path is often the same path as the results file, as above. A native file also embeds a frozen `FEMData` snapshot under `/model/`. If `fem=` is omitted, that snapshot is the bound FEMData (reachable via `results.model.fem`). If `fem=` is provided it is preferred (it typically carries richer apeGmsh-specific labels and provenance than the embedded snapshot). The `snapshot_id` hash is computed and stored as metadata, but bind never enforces equality — pairing a FEMData with a results file from the same run is the user's responsibility.

### `Results.from_mpco` — STKO HDF5

For files written by the STKO MPCO recorder (Strategy C₁/C₂). For an end-to-end recipe see [Read results from an MPCO file](../how-to/results-mpco.md).

```python
results = Results.from_mpco("run.mpco", model_h5="model.h5")

# Multi-partition (parallel runs) — auto-discover .part-N siblings
results = Results.from_mpco("run.part-0.mpco", model_h5="model.h5")

# Or pass an explicit partition list
results = Results.from_mpco(
    ["run.part-0.mpco", "run.part-1.mpco"], model_h5="model.h5"
)
```

`model_h5=` is **required** (ADR 0020 INV-1) — omitting it raises `TypeError`. It points at the sibling apeGmsh `model.h5` (written by `apeSees(fem).h5(...)`); the broker is rehydrated via `OpenSeesModel.from_h5` and held in memory (INV-3 — no derived results h5 is written). MPCO doesn't embed a full FEMData; the reader synthesises a partial one from the file's `MODEL/` group. Pass `fem=` to bind your session-side FEMData (recommended — it carries labels and Parts that the synthesised one doesn't).

### `Results.from_recorders` — classic OpenSees recorders

For `.out` / `.xml` files written by `ops.tcl(..., recorders=spec)`, `ops.py(..., recorders=spec)`, or `spec.emit_recorders(...)` (Strategy A₁/A₂/A₃). Here `ops` is the `apeSees(fem)` bridge.

```python
results = Results.from_recorders(spec, "out/", fem=fem, model=model)

# Multi-stage — pick the stage you want by id
gravity = Results.from_recorders(
    spec, "out/", fem=fem, model=model, stage_id="gravity"
)
```

`from_recorders` transcodes the recorder files into a cached HDF5 (`writers/_cache.py`), keyed on file mtimes + spec `snapshot_id`, and then opens that through `from_native`. Subsequent calls with unchanged inputs return the cached HDF5 directly. Both `fem=` and `model=` are required (omitting either raises `TypeError`): `fem`'s `snapshot_id` participates in the cache key, and `model`'s `/opensees/` zone is embedded into the transcoded native h5 (the Composed-file pattern) so the broker is auto-resolved on the downstream `from_native`. No hash equality is enforced against the resulting file.


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

A constructed `Results` always holds an `OpenSeesModel` broker, and the FEM is reached *through* it — `results.model` is never `None` (ADR 0020 INV-1):

```python
results.model          # OpenSeesModel        — the broker, always present
results.model.fem      # FEMData              — chain-forward to the mesh snapshot
results.fem            # FEMData | None       — shorthand for the bound snapshot
```

`results.model.fem` is the canonical broker path; `results.fem` is the directly-bound snapshot (which `from_native`/`from_mpco` populate from the embedded/synthesised FEMData, or from your `fem=` override).

`.bind(other_fem)` swaps in a different FEMData (typically your session-side one, which carries labels and Parts that the embedded snapshot doesn't):

```python
results = Results.from_native("run.h5", model=model).bind(fem)
```

**No hash validation is performed by `bind`** — pairing the FEMData with the right results file is the user's responsibility. This is intentional; see the bind-contract memory note for the design rationale. (`BindError` was deleted; there is no exception to catch.)

### Lineage — warn, never raise

`results.lineage` returns a `Lineage` — a git-style `fem_hash → model_hash → results_hash` chain (ADR 0021). Each layer's hash folds in its parent's, so the chain is tamper-evident. Mismatches between stored and recomputed hashes surface as `[lineage] ...` strings in `lineage.warnings`; reading the property **never raises** (INV-2):

```python
lin = results.lineage
lin.fem_hash, lin.model_hash, lin.results_hash   # hex digests
lin.warnings                                      # tuple[str, ...] — empty when clean

lin.assert_clean()    # opt-in escalation: raises LineageError if warnings present
```

Call `lin.assert_clean()` when you *want* drift to be fatal (e.g. in a regression check); the default is to warn and continue.


## Lifecycle

`Results` holds an open file handle through its underlying reader. Two ways to release it:

```python
# Explicit close
results = Results.from_native("run.h5", model=model)
# ... use it ...
results.close()

# Context manager
with Results.from_native("run.h5", model=model) as results:
    # ... use it ...
    pass
# closed on exit
```

On Windows in particular, an open HDF5 handle blocks the writer from re-creating the same file in a re-run loop — close before re-running a capture script that overwrites the path.


## Visualisation

### Interactive Qt viewer

```python
results.viewer()                       # blocking, in-process (DEFAULT)
results.viewer(blocking=False)         # subprocess; notebook keeps running
results.viewer(title="Gravity push")
results.viewer(restore_session=False)  # ignore any saved viewer-session.json
results.viewer(save_session=False)     # don't auto-save on close
```

!!! danger "`viewer()` is blocking by default — it crashes the Jupyter kernel"
    `results.viewer()` opens the Qt/VTK `ResultsViewer` **in-process** and blocks the caller. Inside a notebook this kills the ipykernel. In a notebook either spawn the subprocess (`results.viewer(blocking=False)`) or, better, use the kernel-safe web viewer `results.show_web()` (below). The blocking path is fine from a plain terminal script.

The non-blocking path spawns `python -m apeGmsh.viewers <path>` so the kernel can keep running; this requires that the Results was opened from disk (in-memory Results raise).

When non-blocking spawns, the parent reader is closed automatically — this lets you re-run the capture script (which deletes/recreates the file) without hitting Windows' "file in use" error. To keep querying after the spawn, re-open with `Results.from_native(path, model=model)`.

The `restore_session` flag controls whether a sibling `<results>.viewer-session.json` is loaded:
- `True` — restore silently
- `False` — ignore
- `"prompt"` (default) — open a yes/no dialog if a matching session exists

**CI / nbconvert escape hatch.** Set the `APEGMSH_SKIP_VIEWER`
environment variable to make `viewer(...)` print a skip marker and
return `None` immediately — useful when running notebooks under
`jupyter nbconvert --execute` or in CI without a display.

### Web / Jupyter viewer (kernel-safe)

`results.show_web()` renders the FEM substrate plus any director diagrams through a `pyvista.trame` backend — the kernel-safe replacement for the blocking Qt viewer in a notebook (ADR 0042 R-C). It is view-only (picking is deferred), with a step slider and per-layer visibility checkboxes when `ipywidgets` is present.

```python
results.show_web()                       # inline in the notebook
results.show_web(stage="gravity")        # activate a specific stage
results.show_web(controls=False)         # bare view, no ipywidgets panel
results.show_web(render_mode="server")   # render on the kernel, stream images
viewer = results.show_web(show=False)    # get the WebViewer handle, add diagrams first
```

`render_mode` is `"client"` (default — renders in the browser via WebGL, fast camera), `"server"` (renders on the kernel and streams images; most VTK-feature-complete, for very large models), or `"hybrid"` (a local/remote toggle in the toolbar).

For a standalone (non-Jupyter) app, `results.serve_web()` builds a vuetify3 single-page app and serves it at a local URL, opening a browser tab and blocking until Ctrl-C:

```python
results.serve_web()                      # auto-picked port, opens a browser tab
results.serve_web(port=8080, title="Gravity push")
```

Both require the `[viewer]` extra (`pip install "apeGmsh[viewer]"` — pulls `trame` + `ipywidgets`).

No results file handy? `Results.demo()` returns a zero-setup cantilever-pushover sample (a real `apeSees`-emitted model with a synthetic ramped tip deflection, no OpenSees solve) — ideal for trying the viewers:

```python
from apeGmsh import Results

Results.demo().show_web()                # instant sample render
```


## Reading fields

Field reads go through the composite tree and use the same selection vocabulary as the FEM broker. For a focused recipe see [Read a node's displacement and reactions](../how-to/read-results.md); to pick a write-side strategy first see [Choose a results strategy](../how-to/choose-results-strategy.md). A small taste:

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

**Tri31 strain gap.** The Tri31 element has no element-level `"strains"` response branch in OpenSees (only `"stresses"`). Domain capture routes around this by collecting strain per Gauss-point material via `ops.eleResponse(eid, "material", "<gp>", "strain")` for any class listed in `PER_MATERIAL_STRAIN_CLASSES` (`solvers/_element_response.py:2072`); see `capture/_domain.py:870-874` for the per-material query path. From the read side this is invisible — `results.elements.gauss.get(component="strain_xx", pg="Body")` works the same as for any other continuum class.


## Practical workflow

```python
from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees, OpenSeesModel
import openseespy.opensees as ops

with apeGmsh(model_name="slab") as g:
    # ... geometry, mesh, physical groups ...
    fem = g.mesh.queries.get_fem_data(dim=3)

# OpenSees — post-session, explicit declarations.
bridge = apeSees(fem)
bridge.model(ndm=3, ndf=3)
# ... materials, elements, fix, patterns (re-declare explicitly) ...

# Persist the canonical two-zone model.h5 once, then rehydrate the
# read-side broker — every Results constructor below requires it.
bridge.h5("model.h5")
model = OpenSeesModel.from_h5("model.h5")

# Declare recorders and resolve the spec — see guide_obtaining_results.md
# for the full five-strategy walkthrough.  The recorder declaration API
# is `ops.recorder.Node(...)` / `ops.recorder.Element(...)` on the apeSees
# bridge; resolve() returns a ResolvedRecorderSpec used by the strategies below.
spec = ...   # see guide_obtaining_results.md for declaration + resolve

# Run with domain capture (Strategy B)
with spec.capture(path="run.h5", fem=fem, ndm=3, ndf=3) as cap:
    cap.begin_stage("gravity", kind="static")
    # ... static analysis ...
    for _ in range(10):
        ops.analyze(1, 0.1)
        cap.step(t=ops.getTime())
    cap.end_stage()

# Open the file
results = Results.from_native("run.h5", fem=fem, model=model)

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

The summary lists stages, kinds, step counts, available components, and the bound FEMData. It's the first thing to print when a results file behaves unexpectedly. When a specific component comes back empty, call `results.inspect.diagnose("stress_xx")` for a per-level routing report that shows where the component lives or why it's missing.


## See also

- [`guide_obtaining_results.md`](guide_obtaining_results.md) — five strategies for writing results files
- [`guide_results_filtering.md`](guide_results_filtering.md) — selectors, slabs, time slicing
- [`guide_recorders_reference.md`](guide_recorders_reference.md) — what categories and components exist
- [`guide_fem_broker.md`](guide_fem_broker.md) — the `FEMData` shape that `Results` mirrors
