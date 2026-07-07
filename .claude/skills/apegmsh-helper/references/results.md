# Results — post-processing OpenSees output
<!-- skill-freshness: verified against apeGmsh main@8eeda7a3 (2026-07-06) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

`Results` reads an OpenSees run back into apeGmsh's label/query world.
All signatures below are read from `src/apeGmsh/results/Results.py`
(and `_composites.py`, `demo.py`, `viewers/web_viewer.py`).

Top-level import: `from apeGmsh import Results` (also
`from apeGmsh.results import Results, make_demo_results`).

## 1. Constructors — `model=` / `model_h5=` is REQUIRED

Every constructor needs the model broker. Omitting it raises
`TypeError` — this is the single biggest drift in older docs/snippets.

| Constructor | Required kwarg | Source |
|---|---|---|
| `Results.from_native(path, *, fem=None, model=...)` | `model=OpenSeesModel` | apeGmsh native HDF5 (Composed-file: results + `/opensees/` zone in one file) |
| `Results.from_mpco(path, *, fem=None, merge_partitions=True, model_h5=...)` | `model_h5=<path>` | STKO `.mpco` HDF5 (single or `part-N` partitions) |
| `Results.from_ladruno(path, *, fem=None, merge_partitions=True, model_h5=None)` | *(none — self-sufficient)* | OpenSees **Ladruno fork** `.ladruno` HDF5 (single or `part-N`); `model_h5=` optional, only for richer lineage |
| `Results.from_recorders(spec, output_dir, *, fem, model=...)` | `fem=` **and** `model=` | `.out`/`.xml` from a Tcl/Py recorder run; transcodes → native + caches |

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel        # verified: tests/test_results_bind.py::test_lineage_propagates_from_model

# Native apeGmsh HDF5 — model= is REQUIRED (Phase 8 / ADR 0020 INV-1).
# Often the SAME path as the results file (Composed-file pattern).
model = OpenSeesModel.from_h5("run.h5")
results = Results.from_native("run.h5", model=model)

# STKO .mpco — model_h5= (sibling model archive) is REQUIRED.
results = Results.from_mpco("run.mpco", model_h5="model.h5")

# A multi-partition run: pass any one part; siblings auto-discovered.
results = Results.from_mpco("run.part-0.mpco", model_h5="model.h5")

# Ladruno fork .ladruno — SELF-SUFFICIENT: no model_h5 needed (the file
# carries its own geometry/regions/local axes). Partitions auto-discovered.
results = Results.from_ladruno("run.ladruno")
results = Results.from_ladruno("run.part-0.ladruno")          # merged

# Recorder .out/.xml run — needs BOTH fem= and model=.
results = Results.from_recorders(spec, "out/", fem=fem, model=model)
```

Omitting the required kwarg:

```python
Results.from_native("run.h5")        # TypeError: model= is required
Results.from_mpco("run.mpco")        # TypeError: model_h5= is required
Results.from_recorders(spec, "out/", fem=fem)   # TypeError: model= is required
# verified: tests/test_results_bind.py::test_from_native_without_model_raises_typeerror
# verified: tests/test_results_bind.py::test_from_mpco_without_model_h5_raises_typeerror
# verified: tests/test_results_bind.py::test_from_recorders_without_model_raises_typeerror
```

Notes:
- `from_recorders` also requires `fem=` (a separate `TypeError`) — the
  spec's `snapshot_id` is part of the transcode cache key. Phase 6 v1
  supports **nodal records only**; element-level records in the spec
  are skipped with a note.
- `from_mpco` over a **composed** model (ADR 0038/0043) auto-attaches
  an `ElementTagTranslator` so element queries speak `fem_eid`, not the
  raw OpenSees ops tag. (`tests/test_results_mpco_composed_join.py::test_query_by_offset_fem_eid_returns_the_row`)
- `from_ladruno` is the **exception** to the model-required rule: a
  `.ladruno` is the fork's *canonical* recorder and self-describes its
  geometry, so the broker is built from the file (`model_h5=` optional,
  only for richer bridge records / lineage). Reader specifics:
  `results.elements.gauss`/`line_stations` speak the **neutral** vocabulary
  (`stress_xx`, `axial_force`, …) and work cross-backend; `results.elements`
  is **token-driven** — the component is the file's `ON_ELEMENTS/<token>`
  key (`basicForce`/`localForce`/`force`/`globalForce`) returning the raw
  block. Beam diagrams orient from the recorder's `MODEL/LOCAL_AXES`
  (`results.elements.local_axes(...)`; `line_force` uses it for true
  cross-section roll). Energy: `results.energy(region=)`. See
  `references/ladruno.md` for the fork-only details.
- Zero-setup: `Results.demo()` / `make_demo_results(...)` — see §6.

## 2. The three-broker chain — `model`, `model.fem`, `fem`

ADR 0019–0023. A constructed `Results` always carries an
`OpenSeesModel` broker; reach the neutral `FEMData` *through* it.

```python
osm   = results.model        # OpenSeesModel broker — ALWAYS non-None      # verified: tests/test_results_bind.py::test_lineage_propagates_from_model
fem   = results.model.fem    # FEMData neutral zone (chain-forward path)
local = results.fem          # locally-bound snapshot (Optional; may differ after .bind())
```

- `results.model` is non-None on any constructed `Results` (ADR 0020
  INV-1). It is the chain-forward handle.
- `results.model.fem` is the canonical FEMData. Prefer this for
  labels/PGs.
- `results.fem` is the *bound* snapshot — `None` until bound, and may
  diverge from `results.model.fem` after a `.bind()`.

## 3. Lineage — warns, never raises (BindError is gone)

`BindError` was **deleted** in Phase 8. There is no construction-time
hash rejection — pairing the right `fem`/model with a run is the
**user's** responsibility. Provenance surfaces through `results.lineage`.

```python
lin = results.lineage        # Lineage(fem_hash, model_hash, results_hash, warnings)
lin.warnings                 # tuple[str, ...] — "[lineage] ..." drift strings; () = clean
lin.assert_clean()           # opt-in: raises LineageError if warnings non-empty
# verified: tests/test_results_bind.py::test_lineage_propagates_from_model
```

- `results.lineage` **never raises** (ADR 0021 INV-2). A stale FEMData
  paired with a fresh results file shows up as a `"[lineage] ..."`
  string in `lineage.warnings`, not an exception.
- `Lineage` is a dataclass:
  `Lineage(fem_hash: str = "", model_hash: Optional[str] = None, results_hash: Optional[str] = None, warnings: tuple[str, ...] = ())`
  (`src/apeGmsh/opensees/_internal/lineage.py:114`).
- `LineageError` (subclass of `ValueError`,
  `src/apeGmsh/opensees/_internal/lineage.py:102`) is raised **only**
  by the explicit `lineage.assert_clean()` opt-in check.

Re-bind a fresh-session FEMData (to recover labels/Parts the embedded
snapshot lacks) — returns a NEW `Results`, no hash validation:

```python
r2 = results.bind(fem)       # no validation; pairing is your job   # verified: tests/test_results_bind.py::test_bind_accepts_mismatched_fem
disp = r2.nodes.get(pg="Top", component="displacement_z")           # verified: tests/test_results_bind.py::test_pg_query_works_after_bind
```

If results look wrong / PG selectors come back empty, suspect a stale
FEMData bound to a fresh results file — the deleted `BindError` would
have caught it; now it's on you.

## 4. Querying — stages, nodes, elements, modes

```python
results.stages               # list[StageInfo]
s = results.stage("dynamic") # scope to one stage -> a new Results    # verified: tests/test_results_modes.py::test_modes_accessor_returns_scoped_results
```

Node / element reads return a typed slab. Selectors are by **label /
PG / selection / id** (never raw tags); `component=` is keyword-only:

```python
slab = results.nodes.get(pg="Body", component="displacement_x")      # verified: tests/test_results_modes.py::test_mode_shape_is_single_step
slab.values, slab.time, slab.node_ids
```

`nodes.get` signature (`_composites.py:714`) — all selectors keyword-only:

```
get(*, pg=None, label=None, selection=None, ids=None,
    component, time=None, stage=None) -> NodeSlab
```

`results.elements.get(...)` mirrors it (→ `ElementSlab`) and additionally
owns the sub-composites
`results.elements.{gauss, fibers, layers, line_stations, springs}`
(each a `.get(...)` returning `GaussSlab` / `FiberSlab` / `LayerSlab` /
`LineStationSlab` / `SpringSlab`; `gauss`/`fibers`/`layers` also take
`gp_indices=` / `layer_indices=`).

**Modes / eigen:**

```python
results.modes                # list[Results], one per kind="mode" stage   # verified: tests/test_results_modes.py::test_modes_accessor_returns_scoped_results
sorted(results.modes, key=lambda m: m.mode_index)   # stable index order
results.eigen_modes          # list[EigenMode] lightweight snapshots
```

Component names expand DOF-aware (`"displacement"` → `displacement_x/y/z`
clipped to `ndm`; `"stress"` → 6 in 3D / 3 in 2D); unknown names raise
`ValueError`.

## 5. Plots and the desktop viewer

### Static matplotlib (`[plot]` extra)

```python
results.plot.contour("displacement_z", step=-1)          # ResultsPlot.contour, _plot.py:151
results.plot.deformed(step=-1, scale=50, component="stress_xx")
results.plot.history(node=412, component="displacement_x")
```

`results.plot` is the headless renderer. Methods: `mesh / contour /
deformed / history / vector_glyph / reactions / loads / line_force`.
`ImportError` if the `[plot]` extra (matplotlib) is absent.

### Interactive desktop viewer — `blocking=True` CRASHES Jupyter

```python
def viewer(self, *, blocking=True, title=None,
           restore_session="prompt", save_session=True, cuts=None)
```

- The **default `blocking=True`** runs the VTK+Qt event loop in-process
  and **native-crashes a Jupyter / VS Code kernel** even with a GPU.
- In a notebook use **`results.show_web()`** (§6, kernel-safe) or
  **`results.viewer(blocking=False)`** (spawns a subprocess; needs a
  Results opened from disk, raises `RuntimeError` for in-memory).
- `viewer()` already calls `.show()` — never chain
  `results.viewer().show()` (opens two windows).
- `viewer(cuts=...)` is **ignored** on the `blocking=False` subprocess
  path (live `SectionCutDef` objects don't survive the argv hop).
- `APEGMSH_SKIP_VIEWER=1` makes the viewer call return `None` — lets a
  cell survive `jupyter nbconvert --execute` / CI.
- **File → Open Results…** (#757) opens another results file in the running
  viewer, with a format-aware follow-up prompt for the model archive when the
  format needs one (`.mpco` → `model_h5=`; `.ladruno` is self-sufficient).
- Higher-order solids (tet10/20/27, tri6, quad8/9, prism15, Bézier fork
  elements) now **render in the results viewer** (#763) — drawn as their
  linear corner cell (mid-side nodes dropped); previously a quadratic /
  fork-coded mesh emitted 0 cells and left the viewport blank.

### Headless animation export — `results.export_animation(...)` (#755)

```python
results.export_animation("run.mp4", fps=30, step_stride=2,
    deform=("displacement", 50.0),        # or a bare number → "displacement" at that scale
    stage=None, camera=None, window_size=(1280, 720))
```

Renders the time history to **`.mp4`** (H.264 — needs the
`apegmsh[animation]` extra) or **`.gif`** (Pillow, no extra); the path
suffix picks the format. It builds the full results viewer off-screen, so
deformation/contours/camera are pixel-identical to the interactive viewer
(the window flashes briefly — the OpenGL context needs a realized surface —
but no blocking event loop runs). `step_stride=N` captures every N-th step
plus the last; `deform=None` (default) renders undeformed; `setup=` takes a
`callback(plotter, director)` invoked after the scene builds and before
capture (the escape hatch for contours / section cuts / camera work). The
same export lives on the viewer GUI as a button.
`# src/apeGmsh/results/Results.py:1030`

### Concurrent geometries — `director.geometries` (ADR 0058)

A **Geometry is now a scene instance**, not just a deformation preset: every
Geometry with `visible=True` renders **concurrently** at its own deform state,
spatial offset, and stage pin. Mostly a GUI feature, but the same actions are
scriptable on the open viewer's director (`viewer.director.geometries`, a
`GeometryManager`) — only valid **after** the viewer window is open.

```python
gm = viewer.director.geometries          # GeometryManager (src/apeGmsh/viewers/diagrams/_geometries.py)
g2 = gm.add(name="Stage B", make_active=True)            # also: outline "+" button
gm.set_offset(g2.id, (20.0, 0.0, 0.0))                   # side-by-side; never baked into coords (picking stays valid)
gm.set_stage_pin(g2.id, stage_id)                        # read deform from a pinned stage; None = follow active
gm.set_visible(g2.id, True)                              # eye icon; hidden geoms are gate-hidden
gm.set_deformation(field="displacement", scale=2.0)      # per-geometry deform field + scale
gm.add_reference_ghost(g2.id)                            # dimmed deform-OFF wireframe (GHOST_OPACITY=0.3)
viewer.director.duplicate_geometry(g2.id)                # clone + REBUILD every diagram (not runtime overrides)
```

- **GUI-equivalent:** "+" in the Outline header (add), eye icon (visible),
  GeometrySettingsPanel fields/combo (offset, stage pin, deform), right-click
  row (reference ghost, duplicate). Inline F2 renames (`gm.rename`).
- **Single global time cursor** — all geometries share the step index; staged
  comparison is "global step + per-geometry stage pin", not per-geometry cursors.
- **Deform-follow is mandatory** — every diagram tracks the deformed substrate
  (the old `deformed_shape` diagram kind was retired; the undeformed reference is
  the reference-ghost preset). Loading a legacy session drops a `deformed_shape`
  spec with a log line (S4 migration).
- **Duplicate** copies deform state + offset + stage pin and rebuilds diagrams
  from their `DiagramSpec`; live color-map edits / manual hides / probes are NOT
  copied.
- **Session schema is v7** (`SESSION_SCHEMA_VERSION = 7`,
  `viewers/diagrams/_session.py`) — adds per-geometry `stage_id`; older sessions
  load with sensible defaults (`stage_id=None`, `offset=(0,0,0)`, active-only
  visibility).

## 6. Web / Jupyter viewers (ADR 0042 R-C — newest)

Kernel-safe trame/pyvista. Need the **`[viewer]` extra** (trame +
trame-vuetify; `ipywidgets` for the inline controls).

```python
def show_web(self, *, stage=None, show=True, controls=True,
             render_mode="client")            # Results.py:832
def serve_web(self, *, stage=None, render_mode="client", port=None,
              open_browser=True, title="apeGmsh", **start_kwargs)   # Results.py:881
```

```python
from apeGmsh import Results

# Zero-setup sample: real apeSees-emitted model + synthetic cantilever push.
results = Results.demo(n_steps=6, tip_drift=2.0)        # verified: tests/test_results_demo.py::test_results_demo_classmethod

# In a notebook: inline trame view + ipywidgets step slider / layer toggles.
wv = results.show_web(render_mode="client")             # returns a WebViewer   # verified: tests/viewers/test_trame_backend.py::test_show_web_returns_viewer_without_display
wv.set_step(3)                                          # programmatic scrub + re-render   # verified: tests/viewers/test_trame_backend.py::test_web_viewer_step_clamps

# Outside a notebook: standalone vuetify3 web app (blocks until Ctrl-C).
# results.serve_web(render_mode="client", port=8080)
```

`render_mode` (a friendly alias, **not** the raw pyvista backend name) —
valid values are exactly:

| value | behaviour |
|---|---|
| `"client"` (default) | WebGL in the browser; fast camera |
| `"server"` | image-streamed from the kernel; laggy, for very large models |
| `"hybrid"` | toolbar toggle between the two |

Any other value raises `ValueError("Unknown render_mode ...")`
(`tests/viewers/test_trame_backend.py::test_unknown_render_mode_raises`).

`Results.demo(**kwargs)` delegates to `make_demo_results(*, length=10.0,
n_elements=8, n_steps=6, tip_drift=2.0, path=None)`
(`from apeGmsh.results import make_demo_results`). It emits a **real**
apeSees OpenSees model (satisfying the `model=` contract) but the
displacements are a **synthetic analytic cantilever shape** — no
OpenSees solve runs. `n_steps<1` / `n_elements<1` raise `ValueError`.
(`tests/test_results_demo.py::test_make_demo_results_shape`,
`::test_demo_rejects_bad_args`)

### Web-viewer gotchas

- `WebViewer` / `show_web` / `serve_web` are **not** top-level exports —
  reach them via `Results` methods or
  `from apeGmsh.viewers.web_viewer import WebViewer, show_web, serve_web`.
- A `WebViewer` requires a `Results` with **bound FEMData**, else
  `RuntimeError("WebViewer requires a Results with bound FEMData ...")`.
  Construct with `fem=` or call `results.bind(fem)` first.
- It is **view-only** at construction: it renders the substrate plus
  whatever diagrams the director already holds. `show_web`/`serve_web`
  do **not** add a deformed/contour layer for you — add one via
  `wv.director.registry.add(...)` before `show()`.
- `show(controls=True)` degrades **gracefully** if `ipywidgets` is
  missing (returns the bare view); `WebViewer.controls()` instead
  **raises** `RuntimeError` pointing at the `[viewer]` extra.
- Static fallback: if pyvista can't launch the in-notebook trame server
  (usually missing `nest_asyncio2`), `show()` emits a `RuntimeWarning`
  to `pip install nest_asyncio2` and degrades to a static image with no
  controls.
- `APEGMSH_SKIP_VIEWER` short-circuits both `show()` (returns `None`)
  and `serve()` (returns the *unstarted* server, never blocks) — the
  CI/headless guard.
- Picking is **off** on the web (`TrameBackend.supports_picking() ->
  False`); deferred to Phase R-D.

## 7. Quick reference

```python
# Construct (model= / model_h5= REQUIRED — except from_ladruno)
Results.from_native(path, *, fem=None, model=OpenSeesModel.from_h5(path))
Results.from_mpco(path, *, fem=None, merge_partitions=True, model_h5="model.h5")
Results.from_ladruno(path, *, fem=None, merge_partitions=True, model_h5=None)  # self-sufficient
Results.from_recorders(spec, out_dir, *, fem=fem, model=model)
Results.demo(n_steps=6, tip_drift=2.0)              # zero-setup sample

# Broker chain
results.model            # OpenSeesModel — never None
results.model.fem        # FEMData neutral zone (use this)
results.fem              # locally-bound snapshot (Optional)

# Provenance (never raises)
results.lineage.warnings ; results.lineage.assert_clean()
results.bind(fem)        # new Results; no hash validation

# Query (labels/PGs, not tags)
results.stages ; results.stage("dynamic")
results.nodes.get(pg=..., component="displacement_z", time=None, stage=None)
results.elements.get(...) ; results.elements.{gauss,fibers,layers,line_stations,springs}
results.modes ; results.eigen_modes

# Render
results.plot.contour("displacement_z", step=-1)    # [plot] extra
results.show_web()                                  # notebook-safe; [viewer] extra
results.serve_web(port=8080)                        # standalone web app
results.viewer(blocking=False)                      # desktop subprocess
# results.viewer()  -> blocking=True default CRASHES the Jupyter kernel
results.export_animation("run.mp4", fps=30)         # headless video/GIF (§5)
```

Locking tests (run green this session): `tests/test_results_bind.py`,
`tests/test_results_modes.py`, `tests/test_result_chain.py`,
`tests/test_results_demo.py`, `tests/viewers/test_trame_backend.py`.
