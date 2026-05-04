# Results — Architecture & Implementation Plan

> [!note] Status
> Design complete (April 2026). This is the **single canonical
> reference** for the Results module rebuild. Companion to
> [[apeGmsh_architecture]] — uses the same conventions: composites,
> `pg=`/`label=` selection, immutable snapshots.
>
> Part I covers the architecture. Part II is the phased
> implementation plan. Part III collects deferred work.

---

## Part I — Architecture

### Vision

A backend-agnostic, partition-aware, multi-stage results system that
the viewer and post-processing scripts can talk to without caring
whether the underlying data came from MPCO, OpenSees text/xml
recorders, or in-process domain introspection.

The user defines **what to record once** (declaratively, by physical
group / label, in the apeGmsh session), picks an **execution
strategy** (file recorders, in-process capture, or MPCO), and reads
results through a **single composite API** that mirrors `FEMData`.

### Design principles

1. **One declarative spec, three execution strategies.** Recorders
   are owned by apeGmsh, not by the user's solver script. The same
   `g.opensees.recorders` declaration drives Tcl/Python recorder
   commands, in-process introspection, or MPCO output.
2. **Data always lives on disk.** RAM is for queries, not storage.
   Every backend resolves to chunked HDF5; lazy slab reads keep
   memory bounded for million-DOF, multi-partition runs.
3. **Two HDF5 schemas, one reader contract.** apeGmsh native
   schema (we own it) covers Tcl/Python/domain paths. MPCO schema
   (STKO-defined) is read directly. Both readers expose the same
   protocol; the composite layer above never branches.
4. **Composite API mirrors FEMData.** Same `pg=`/`label=` selection
   vocabulary, same immutable-snapshot feel. A user who knows
   FEMData already knows Results.
5. **Soft FEM coupling, hash-tagged.** Results can be built without
   a FEMData (raw inspection works). PG/label resolution and viewer
   plotting require a bound FEM. Native HDF5 embeds the FEMData
   snapshot; MPCO files reconstruct a partial FEMData from their
   `MODEL` group. The `snapshot_id` hash is computed and stored as
   metadata, but bind never enforces equality — pairing a FEMData
   with a results file from the same run is the user's
   responsibility.
6. **Verbose, human-readable component names.** No abbreviations
   except universal mechanics tensor indices (`xx`, `xy`, …). Backend
   adapters translate to canonical names on the way in.

### Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ DECLARATION  (apeGmsh session, by PG/label, like loads/masses)       │
│                                                                      │
│   g.opensees.recorders.nodes("Top", components=["displacement"])    │
│   g.opensees.recorders.gauss("Body", components=["stress"])         │
│                                                                      │
│   spec = g.opensees.recorders.resolve(fem)   # → ResolvedSpec        │
│                                  (carries fem snapshot_id)          │
└──────────┬───────────────────────┬───────────────────────┬───────────┘
           │                       │                       │
           ▼                       ▼                       ▼
   ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
   │ TclWriter    │        │ DomainCapture│        │ MPCOWriter   │
   │ + parser     │        │ (in-process) │        │ (recorder    │
   │ (.out/.xml)  │        │ openseespy   │        │  mpco …)     │
   └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
          │                       │                       │
          ▼                       ▼                       ▼
   ┌────────────────────────────────────────┐    ┌──────────────────┐
   │ apeGmsh native HDF5 (run.h5)            │    │ STKO MPCO HDF5  │
   │  ├── attrs (schema_version, …)          │    │ (run.mpco)      │
   │  ├── time, stages/, partitions/         │    │  └── MODEL_STAGE│
   │  └── model/ (embedded FEMData snapshot, │    │       /MODEL,   │
   │             carries snapshot_id)        │    │        /RESULTS │
   └────────────────┬───────────────────────┘    └──────────┬──────┘
                    │                                       │
                    ▼                                       ▼
            ┌──────────────┐                       ┌──────────────┐
            │ NativeReader │                       │ MPCOReader   │
            └──────┬───────┘                       └──────┬───────┘
                   │                                       │
                   └───────────────┬───────────────────────┘
                                   ▼
                       ┌────────────────────┐
                       │ Results composite  │   ← user-facing API
                       │  .stages, .modes   │
                       │  .nodes            │   ← mirrors FEMData
                       │  .elements         │
                       │    .gauss          │
                       │    .fibers         │
                       │    .layers         │
                       │  .fem  (bound)     │
                       │  .inspect          │
                       └────────────────────┘
                                   │
                                   ▼
                          viewer / scripts
```

### Layer 1 — Declarative recorder spec

`g.opensees.recorders` is a pre-mesh / pre-build composite, parallel
to `g.loads` / `g.masses` / `g.constraints`. Same two-stage pipeline:
declare against PGs/labels, resolve through `FEMData`.

#### Categories (cover everything OpenSees supports)

| Method | Records | Schema target |
|---|---|---|
| `recorders.nodes(pg, components, ...)` | `disp`, `vel`, `accel`, `incrDisp`, `reaction`, `pressure` | `nodes/` |
| `recorders.elements(pg, components, ...)` | `globalForce`, `localForce` (per element-node) | `elements/nodal_forces/` |
| `recorders.line_stations(pg, components, ...)` | `section.force`, `section.deformation` (along beams) | `elements/line_stations/` |
| `recorders.gauss(pg, components, ...)` | `stress`, `strain`, `material.stress`, `material.strain` (continuum) | `elements/gauss_points/` |
| `recorders.fibers(pg, components, ...)` | `section.fiber.stress` / `material.fiber.stress` (auto shell-keyword swap) | `elements/fibers/` |
| `recorders.layers(pg, components, ...)` | layered shell stress/strain per layer + sub-GP | `elements/layers/` |
| `recorders.modal(n_modes, ...)` | eigenvalues + mode shapes | one stage per mode (`kind="mode"`) |

#### API conventions

- **Components accept shorthand or explicit, never crossing
  categories.** A shorthand expands to all components of one named
  family, clipped to the active `ndm`/`ndf`. A user who wants two
  families lists both:

  | Shorthand | Expands to |
  |---|---|
  | `"displacement"` | translational only — `displacement_x/y/z` (clipped to ndm) |
  | `"rotation"` | rotational only — `rotation_x/y/z` (only if ndf ≥ 6) |
  | `"velocity"` | `velocity_x/y/z` |
  | `"angular_velocity"` | `angular_velocity_x/y/z` |
  | `"acceleration"` | `acceleration_x/y/z` |
  | `"angular_acceleration"` | `angular_acceleration_x/y/z` |
  | `"force"` | `force_x/y/z` |
  | `"moment"` | `moment_x/y/z` |
  | `"reaction"` | `reaction_force_x/y/z` AND `reaction_moment_x/y/z` (one shorthand for the whole reaction record, since `ops.recorder ... reaction` returns both) |
  | `"stress"` | all 6 stress components |
  | `"strain"` | all 6 strain components |

  Mixed kinematics is two entries: `["displacement", "rotation"]`.
  Explicit per-component (`["displacement_x"]`) is always valid.
- **Default recorder set** can be enabled with
  `g.opensees.recorders.enable_defaults()`: displacements on all nodes,
  reactions on fixed nodes, stress on all GPs of continuum elements,
  axial forces on all line elements. Off by default.
- **Output cadence** via `dt=` or `n_steps=`. Defaults to every
  analysis step.
- **Validation at resolve time.** `g.opensees.recorders.gauss("Beams",
  components=["stress"])` raises if those elements have no GPs in
  apeGmsh's element-spec capability table — fail fast, not after a
  4-hour run.

#### Resolution

```python
fem = g.mesh.queries.get_fem_data(dim=3)
spec = g.opensees.recorders.resolve(fem)   # ResolvedRecorderSpec

# spec carries per-recorder:
#   - canonical component names (shorthand expanded)
#   - concrete node IDs / element IDs (PGs flattened)
#   - group keys (class_tag, int_rule) for element-level results
#   - cadence (dt or n_steps)
#   - intended output channel
#
# spec also carries a top-level fem snapshot_id (content hash of the
# FEMData it was resolved against). Re-meshing produces a new
# snapshot_id; old specs refuse to emit/capture against the new fem.
```

The resolved spec is the **single source of truth** for what gets
recorded. It also serves as the manifest for parsing the recorder
output files.

### Layer 2 — Execution strategies

Five strategies, one spec. The user picks based on workflow. For a
focused walkthrough see
[architecture/apeGmsh_results_obtaining.md](../architecture/apeGmsh_results_obtaining.md);
this section is the schema-grounded reference.

The strategies group into three families that share a producer:

- **A. Classic recorders** — `recorder Node …` / `recorder Element …`
  text files. Three sub-strategies differ only in *where the recorder
  command runs* (Tcl script, Python script, in-process). All three
  produce `.out` / `.xml` and read back through `from_recorders`.
- **B. Domain capture** — apeGmsh queries the live `ops` domain itself
  and writes native HDF5. One strategy.
- **C. MPCO** — STKO's `recorder mpco` writes one HDF5. Two
  sub-strategies (Tcl-export, in-process).

#### Strategy A₁ — Tcl/Python file recorders

```python
g.opensees.export.tcl("model.tcl", recorders=spec)
# OpenSees writes .out / .xml files when the user runs the script
results = Results.from_recorders(spec, output_dir="out/", fem=fem)
```

`g.opensees.export.{tcl,py}(recorders=spec)` adds the matching
`recorder Node …` / `recorder Element …` commands to the script. The
spec is also serialized as a sidecar HDF5 manifest so
`Results.from_recorders` can decode the column layout of each file.

`Results.from_recorders` runs `RecorderTranscoder`, which:
1. Reads the manifest + each recorder file (txt or xml).
2. Translates raw OpenSees tokens to canonical apeGmsh names.
3. Writes a single `run.h5` in native schema.
4. Caches the result. Re-running with unchanged inputs is a no-op
   (mtime + size + fem `snapshot_id` check on each source).

#### Strategy A₃ — Live recorders (in-process)

```python
with spec.emit_recorders("out/") as live:
    live.begin_stage("gravity", kind="static")
    for _ in range(n_grav):
        ops.analyze(1, 1.0)
    live.end_stage()

    live.begin_stage("dynamic", kind="transient")
    for _ in range(n_dyn):
        ops.analyze(1, dt)
    live.end_stage()

grav = Results.from_recorders(spec, "out/", fem=fem, stage_id="gravity")
dyn  = Results.from_recorders(spec, "out/", fem=fem, stage_id="dynamic")
```

Same recorder commands as A₁/A₂, but pushed into the live openseespy
domain via `ops.recorder(*args)` — no script written, no subprocess.
Per-stage filename prefix `<stage>__<record>_<token>.{out,xml}`
keeps multi-stage output unambiguous; `Results.from_recorders` learns
the prefix via the `stage_id=` parameter (which threads through the
emitter, the cache key, and the transcoder's path discovery).

Single-source-of-truth note: the `LogicalRecorder` dataclass returned
by `emit_logical()` is shared between A₁ (`format_tcl`), A₂
(`format_python`), and A₃ (`to_ops_args`). The same MPCO
`mpco_ops_args` mirrors `emit_mpco_python` for C₁/C₂.

Coverage: nodes, elements, gauss, line_stations. Fibers / layers
warn-and-skip (use B or C). Modal raises at `__enter__`.

#### Strategy B — In-process domain capture

```python
with spec.capture(path="run.h5", fem=fem) as cap:
    cap.begin_stage("gravity")
    for _ in range(n_grav_steps):
        ops.analyze(1, 1.0)
        cap.step(t=ops.getTime())
    cap.end_stage()
    cap.begin_stage("dynamic")
    for _ in range(n_dyn_steps):
        ops.analyze(1, dt)
        cap.step(t=ops.getTime())
    cap.end_stage()

    # Modal stages — one per mode, kind="mode"
    cap.capture_modes(n_modes=10)

results = Results.from_native("run.h5", fem=fem)
```

`spec.capture()` returns a `DomainCapture` writer. On each `step()`:
1. Calls `ops.nodeDisp()`, `ops.eleResponse()`, etc. for everything
   the resolved spec asks for.
2. Buffers per-step arrays in RAM up to a configurable threshold
   (default 10 MB).
3. Flushes buffered chunks to the open `run.h5` handle.

`cap.capture_modes(n_modes)` runs `ops.eigen(n_modes)`, extracts the
mode shapes via `ops.nodeEigenvector(...)`, and writes one stage per
mode with `kind="mode"` and the eigenvalue/frequency/period in the
stage attrs. Bounded RAM, no per-step file I/O storms.

#### Strategy C₁ — STKO/MPCO bridge (Tcl export)

```python
g.opensees.export.tcl("model.tcl", recorders=spec, mpco=True)
# Writes `recorder mpco …` commands instead of (or alongside) text
results = Results.from_mpco("out/run.mpco", fem=fem)
```

For users who want STKO compatibility or already have MPCO files
from upstream. The spec maps to `recorder mpco -N <tokens> -E <tokens>`;
the MPCO recorder produces its own HDF5, which `MPCOReader` reads
directly without transcoding.

#### Strategy C₂ — Live MPCO (in-process)

```python
with spec.emit_mpco("run.mpco"):
    for _ in range(n_steps):
        ops.analyze(1, dt)

results = Results.from_mpco("run.mpco")
```

Same MPCO file as C₁, but emitted in-process via
`ops.recorder("mpco", ...)`. No `begin_stage`/`end_stage` ceremony —
MPCO writes one file containing all stages with `pseudoTime` encoding
boundaries internally. `LiveMPCO.__enter__` runs a build-gate probe;
if the active openseespy build doesn't include MPCO, it raises with
a remediation pointer (use STKO's bundled Python, fall back to A₃,
or use C₁).

Coverage: all categories — including fibers, layers, and modal —
because the MPCO recorder handles them natively
(`section.fiber.stress`, layered-section tokens, `modesOfVibration`).

### Layer 3 — Native HDF5 schema

We control this format. Every non-MPCO path writes here; the reader
above is source-agnostic.

#### Top-level layout

```
run.h5
├── attrs:
│   ├── schema_version        "1.0"
│   ├── source_type           "tcl_recorders" | "domain_capture"
│   ├── source_path           original file or "<domain>"
│   ├── created_at            ISO timestamp
│   ├── apegmsh_version
│   └── analysis_label        optional user tag
│
├── model/                    # embedded FEMData snapshot
│   ├── attrs:
│   │   ├── snapshot_id       deterministic content hash
│   │   ├── ndm, ndf
│   │   ├── model_name
│   │   └── units
│   ├── nodes/
│   │   ├── ids               (N,)
│   │   ├── coords            (N, 3)
│   │   ├── physical_groups/  # subgroups: name → ids
│   │   ├── labels/           # apeGmsh-specific
│   │   ├── loads/            # resolved nodal loads
│   │   └── masses/
│   ├── elements/
│   │   ├── per-type subgroups (ids, connectivity)
│   │   ├── physical_groups/
│   │   ├── labels/
│   │   └── loads/
│   └── constraints/          # node-pair, surface, etc.
│
└── stages/
    └── stage_0/
        ├── attrs:
        │   ├── name="gravity", label="…"
        │   ├── kind="transient"|"static"|"mode"
        │   └── mode-only: eigenvalue, frequency_hz, period_s, mode_index
        ├── time              (T,)         # for kind="mode": shape (1,) holding 0.0
        └── partitions/
            └── partition_0/
                ├── nodes/
                │   ├── _ids                 (N,)
                │   ├── displacement_x       (T, N)
                │   ├── displacement_y       (T, N)
                │   ├── reaction_force_x     (T, N)
                │   └── …
                │
                └── elements/
                    ├── _ids                 (E,)
                    │
                    ├── nodal_forces/        # globalForce / localForce
                    │   └── group_<n>/       # one per (class_tag, npe)
                    │       ├── attrs: class_tag, frame="global"|"local"
                    │       ├── _element_index            (E_g,)
                    │       ├── nodal_resisting_force_x   (T, E_g, npe_g)
                    │       └── …
                    │
                    ├── line_stations/       # beam section forces
                    │   └── group_<n>/       # one per (class_tag, n_IP)
                    │       ├── attrs: class_tag, int_rule
                    │       ├── _element_index            (E_g,)
                    │       ├── _station_natural_coord    (n_IP_g,)
                    │       ├── axial_force               (T, E_g, n_IP_g)
                    │       └── …
                    │
                    ├── gauss_points/        # continuum stress/strain
                    │   └── group_<n>/       # one per (class_tag, int_rule)
                    │       ├── attrs: class_tag, int_rule, custom_rule_idx
                    │       ├── _element_index            (E_g,)
                    │       ├── _natural_coords           (n_GP_g, dim)
                    │       ├── _local_axes_quaternion    (E_g, 4)   # shells
                    │       ├── stress_xx                 (T, E_g, n_GP_g)
                    │       └── …
                    │
                    ├── fibers/              # within fiber-section GPs
                    │   └── group_<n>/       # one per fiber section
                    │       ├── attrs: section_tag, section_class
                    │       ├── _element_index            (sum_F,)
                    │       ├── _gp_index                 (sum_F,)
                    │       ├── _y, _z, _area             (sum_F,)
                    │       ├── _material_tag             (sum_F,)
                    │       ├── fiber_stress              (T, sum_F)
                    │       └── fiber_strain              (T, sum_F)
                    │
                    └── layers/              # layered shells
                        └── group_<n>/
                            ├── _element_index            (sum_L,)
                            ├── _gp_index                 (sum_L,)
                            ├── _layer_index              (sum_L,)
                            ├── _sub_gp_index             (sum_L,)
                            ├── _thickness                (sum_L,)
                            ├── _local_axes_quaternion    (sum_L, 4)
                            ├── stress_xx                 (T, sum_L)
                            └── …
```

#### Schema conventions

- **Underscore-prefixed datasets** are index/metadata (location,
  IDs, parent links). **No-prefix** datasets are result components.
- **Per-group rectangular arrays.** Within a group, all elements
  share `(class_tag, int_rule)` so per-element shapes are uniform —
  we use `(T, E_g, n_GP_g)` rectangular tensors, not CSR. Across
  groups, shapes differ; the reader stitches.
- **Gauss coordinates are natural** in `[-1, +1]`, matching MPCO.
  Global coordinates are computed on demand by the reader using
  the bound FEMData and shape functions.
- **Fibers and layers use CSR-style flat arrays** (variable count
  per element/GP), since their per-section/per-layer structure can
  legitimately differ across elements of the same class.
- **Local axes** for shells stored as quaternions on the gauss /
  layer groups. Stress is in local axes; viewer transforms to
  global on demand.
- **Time vector is per-stage** — analysis stages can have different
  step counts and dt. Different *kinds* of stages can also coexist:
  one transient + one static + ten mode stages, all in the same
  `run.h5`.
- **Modes are stages with `kind="mode"`.** No separate `modes/` group,
  no separate `Modes` class. A mode stage carries a single "step"
  (T=1) holding the mode shape — stored in the same `nodes/`
  datasets (`displacement_x/y/z`, `rotation_x/y/z`) that transient
  stages use. The mode's eigenvalue / frequency / period live in
  the stage attrs. Sugar accessor: `results.modes` filters stages
  by `kind=="mode"`.
- **Single-partition** files just have `partition_0`. Multi-partition
  files have `partition_0`, `partition_1`, … each with the same
  result categories. Reader stitches transparently.

#### Cache layout (recorder transcoder only)

The cache lives at `<project_root>/results/` — visible at the
project root, not hidden:

```
<project_root>/results/
├── <hash>.h5                 # transcoded native HDF5
└── <hash>.manifest.h5        # small HDF5 sidecar (source path,
                              # mtime, size, parser_version,
                              # fem_snapshot_id, recorder spec)
```

Manifest is **HDF5** (not JSON) so it can carry numpy ID arrays
from the resolved spec without lossy serialization tricks.

Cache key: `(source_path, mtime, size, parser_version,
fem_snapshot_id)`. On mtime change → re-transcode. On parser version
bump → invalidate all. On fem mismatch → refuse with a clear error
(re-meshing without re-recording is a real bug, not a soft warning).

### Layer 4 — Reader protocol

Two readers (`NativeReader`, `MPCOReader`), one protocol. The
composite layer above never branches.

```python
class ResultsReader(Protocol):
    # Stage discovery
    def stages(self) -> list[StageInfo]: ...
    def time_vector(self, stage_id: str) -> np.ndarray: ...
    def partitions(self, stage_id: str) -> list[str]: ...

    # FEM access (native: from embedded; MPCO: synthesized from MODEL)
    def fem(self) -> FEMData | None: ...

    # Component discovery
    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]: ...
    # ResultLevel ∈ {NODES, ELEMENTS, LINE_STATIONS, GAUSS, FIBERS, LAYERS}

    # Slab reads — return value-plus-metadata containers
    def read_nodes(
        self, stage_id, component, *,
        node_ids=None, time_slice=None,
    ) -> NodeSlab: ...

    def read_elements(
        self, stage_id, component, *,
        element_ids=None, time_slice=None,
    ) -> ElementSlab: ...

    def read_line_stations(
        self, stage_id, component, *,
        element_ids=None, time_slice=None,
    ) -> LineStationSlab: ...

    def read_gauss(
        self, stage_id, component, *,
        element_ids=None, time_slice=None,
    ) -> GaussSlab: ...

    def read_fibers(
        self, stage_id, component, *,
        element_ids=None, gp_indices=None, time_slice=None,
    ) -> FiberSlab: ...

    def read_layers(
        self, stage_id, component, *,
        element_ids=None, gp_indices=None, layer_indices=None,
        time_slice=None,
    ) -> LayerSlab: ...
```

`StageInfo` carries `id`, `name`, `kind` (`"transient"|"static"|"mode"`),
`n_steps`, and mode-only fields (`eigenvalue`, `frequency_hz`,
`period_s`, `mode_index`).

Slab dataclasses (`NodeSlab`, `GaussSlab`, etc.) carry both the
values and the location index (which element, which GP, etc.) so
the caller never has to re-derive it. They are numpy-native; the
viewer wraps them in `xarray` if it wants labeled axes.

The reader hides the **(class_tag, int_rule) grouping** internally:
the composite asks "stress_xx for elements [5, 7, 11]", the reader
figures out which groups they live in, reads the relevant slabs,
concatenates. Caller never sees groups.

### Layer 5 — Composite API (user-facing)

```python
results = Results.from_native("run.h5", fem=fem)        # native path
results = Results.from_mpco("run.mpco", fem=fem)        # MPCO path
results = Results.from_recorders(spec, output_dir="out/", fem=fem)
                                                        # parses .out/.xml,
                                                        # builds native HDF5,
                                                        # opens for read

# fem= is optional — fallbacks:
# - native: FEMData reconstructed from /model/ group
# - MPCO:   partial FEMData synthesized from /MODEL group
results = Results.from_native("run.h5")     # auto-bind from embedded fem
results = results.bind(g_session_fem)       # explicit override

# Stages
results.stages                              # list[StageInfo] — all stages
gravity = results.stage("gravity")          # stage-scoped Results
dynamic = results.stage("dynamic")

# Modes — sugar accessor that filters stages by kind="mode"
results.modes                               # list of mode-kind stages
mode_3 = results.modes[2]                   # 0-indexed (mode 3)
mode_3.frequency_hz                         # 5.41
mode_3.period_s                             # 0.185
mode_3.eigenvalue                           # 1158.7
shape_x = mode_3.nodes.get(component="displacement_x")
shape_x.values                              # ndarray (1, N) — single "step"

# Nodal — selection by pg/label/ids matches FEMData
disp_z = gravity.nodes.get(pg="Top", component="displacement_z")
disp_z.values                               # ndarray (T, N)
disp_z.node_ids                             # ndarray (N,)
disp_z.time                                 # ndarray (T,)

# Single time step
peak = gravity.nodes.get(
    pg="Top", component="displacement_z", time=10.0,   # nearest step
)
peak.values                                 # ndarray (N,)

# Time slicing — numpy half-open semantics: [start, stop)
window = gravity.nodes.get(
    pg="Top", component="displacement_z",
    time=slice(0.0, 5.0),                   # 0.0 ≤ t < 5.0
)
picks = gravity.nodes.get(
    pg="Top", component="displacement_z",
    time=[0, 50, 100],                      # explicit step indices
)

# Continuum stress at GPs
sigma = gravity.elements.gauss.get(
    pg="Body", component="stress_xx",
)
sigma.values                                # ndarray (T, sum_GP)
sigma.element_index                         # ndarray (sum_GP,)
sigma.natural_coords                        # ndarray (sum_GP, dim)
sigma.global_coords()                       # method — computed via fem

# Fibers
sf = gravity.elements.fibers.get(
    pg="Columns", component="fiber_stress",
)
sf.values                                   # ndarray (T, sum_F)
sf.element_index, sf.gp_index               # parent (elem, gp) per fiber
sf.y, sf.z, sf.area                         # section-local geometry

# Layered shells
sl = gravity.elements.layers.get(
    pg="Slabs", component="stress_xx",
)
sl.element_index, sl.gp_index, sl.layer_index, sl.sub_gp_index

# Inspection
print(results.inspect.summary())            # stages, partitions, components
results.inspect.components()                # what's available where
```

Selection rules match FEMData exactly: provide one of `pg=`,
`label=`, `ids=` (or none → all). PG/label resolution requires
bound FEMData; `ids=` works without.

### FEMData embedding & binding

#### Snapshot identity

Every FEMData carries a `snapshot_id` — a deterministic content
hash computed at construction time over the canonicalized node IDs,
node coords, element IDs, connectivity, and physical-group
membership. This hash is the contract that ties artifacts together:

- A `run.h5` file embeds the FEMData snapshot in `/model/`, including
  its `snapshot_id`.
- A resolved recorder spec carries the `snapshot_id` of the FEMData
  it was resolved against (and it participates in the recorder
  cache key).
- `bind()` does **not** validate `snapshot_id`. The hash is computed
  and stored as metadata, but bind never enforces equality —
  pairing a FEMData with a results file from the same run is the
  user's responsibility (see `_bind.py:8-10`).

Re-meshing produces a new hash automatically, which keeps caches
honest, but old artifacts will still bind silently against a new
FEMData. The `BindError` symbol is retained for back-compat but is
no longer raised.

#### Native HDF5 — full snapshot

`NativeWriter` snapshots the entire FEMData into `/model/` at
write time, including `snapshot_id` as a root attribute. Storage
cost is small relative to results (few MB for a 100k-node model;
results are GBs). On read, `FEMData.from_native_h5(h5_handle)`
reconstructs the snapshot with the same hash.

This means a `run.h5` shipped to a colleague is **fully
self-contained** — they can open the viewer without any
apeGmsh session that produced it.

#### MPCO — partial reconstruction

MPCO's `MODEL/` group contains:
- `MODEL/NODES/COORDINATES`, `MODEL/NODES/ID`
- `MODEL/ELEMENTS/<classTag>-<ClassName>[…]/CONNECTIVITY`
- `MODEL/SETS/SELECTION_SET_<n>` → mapped to `physical_groups/`
- `MODEL/SECTION_ASSIGNMENTS/…` → fiber section data
- `MODEL/LOCAL_AXES/…` → shell quaternions

`FEMData.from_mpco_model(h5_handle)` synthesizes a partial
FEMData. **Missing** vs native: apeGmsh-specific `labels`, Part
provenance, pre-mesh load/mass/constraint declarations.
Selection by name still works (selection sets become PGs).
Queries that touch missing fields (`fem.labels.entities("foo")`)
raise with a clear message.

#### Bind semantics

```python
# Bare construction — works for direct ID queries
raw = Results.from_mpco("run.mpco")      # auto-binds from MPCO MODEL
raw.nodes.get(ids=[1,2,3], component="displacement_x")     # OK
raw.nodes.get(pg="Top", component="displacement_x")        # OK (PGs from MPCO)

# Explicit re-bind — for the case where you re-built an identical
# fem in a fresh session and want to use its labels / loads / Parts
results = raw.bind(g_session_fem)

# No drift detection — bind accepts any FEMData
results = raw.bind(other_fem)            # accepted; no hash check
```

The `snapshot_id` hash is computed and stored as metadata, but bind
never enforces equality — pairing a FEMData with a results file from
the same run is the user's responsibility (see `_bind.py:8-10`). The
`BindError` symbol is retained for back-compat but is no longer
raised.

For MPCO files (no native snapshot_id), the partial FEMData
synthesized from `MODEL/` gets a hash computed on the fly for
metadata. Bind accepts any candidate regardless of how that hash
compares.

### Canonical naming vocabulary

Verbose, lowercase, no abbreviations except mechanics tensor
indices.

#### Nodal kinematics

| Name | Notes |
|---|---|
| `displacement_x`, `displacement_y`, `displacement_z` | translational |
| `rotation_x`, `rotation_y`, `rotation_z` | rotational DOFs (ndf >= 6) |
| `velocity_x/y/z`, `angular_velocity_x/y/z` | first time derivative |
| `acceleration_x/y/z`, `angular_acceleration_x/y/z` | second time derivative |
| `displacement_increment_x/y/z` | OpenSees `incrDisp` |

#### Nodal forces

| Name | Notes |
|---|---|
| `force_x/y/z`, `moment_x/y/z` | applied loads |
| `reaction_force_x/y/z`, `reaction_moment_x/y/z` | OpenSees `reaction` |
| `pore_pressure`, `pore_pressure_rate` | u-p formulations |

#### Per-element nodal resisting forces

| Name | Notes |
|---|---|
| `nodal_resisting_force_x/y/z` | OpenSees `globalForce` |
| `nodal_resisting_force_local_x/y/z` | OpenSees `localForce` |
| `nodal_resisting_moment_x/y/z`, `nodal_resisting_moment_local_x/y/z` | rotational components |

#### Beam line diagrams (per station)

| Name | Notes |
|---|---|
| `axial_force` | along beam axis |
| `shear_y`, `shear_z` | transverse shears in section frame |
| `torsion` | about beam axis |
| `bending_moment_y`, `bending_moment_z` | bending moments in section frame (distinct from `moment_x/y/z` at nodes — different frame and topology) |

#### Continuum stress / strain

| Name | Notes |
|---|---|
| `stress_xx`, `stress_yy`, `stress_zz` | normal Cauchy stresses |
| `stress_xy`, `stress_yz`, `stress_xz` | shears |
| `strain_xx`, …, `strain_xz` | infinitesimal strain components |
| `von_mises_stress`, `pressure_hydrostatic` | scalar derived |
| `principal_stress_1/2/3` | eigenvalues of stress |
| `equivalent_plastic_strain` | scalar plasticity state |

#### Fiber

| Name | Notes |
|---|---|
| `fiber_stress`, `fiber_strain` | uniaxial along fiber axis |

#### Material state

| Name | Notes |
|---|---|
| `damage` | scalar damage variable |
| `state_variable_<n>` | generic, material-defined |

#### Mode shapes — no separate names

Mode shapes are stored as `displacement_x/y/z` (and `rotation_x/y/z`
when ndf ≥ 6) inside a stage with `kind="mode"`. Per-mode scalars
(`eigenvalue`, `frequency_hz`, `period_s`, `mode_index`) live in
the stage's `attrs`, not as components.

### Validation against source-of-truth schemas

Cross-checked the design against three sources:

#### MPCO (`mpco-recorder` skill)

- ✅ HDF5 + lazy reads + multi-partition: matches
- ✅ Element grouping by `(class_tag, int_rule, custom_rule_idx)`: adopted
- ✅ Natural coordinates for Gauss points: adopted (with
  global coords on-demand via shape functions)
- ✅ Local axes via quaternion: adopted for shells / layers
- ✅ Fiber section structure: matches `fibers/` schema
- ✅ Multiple model stages: adopted as `stages/` top-level
- 🟡 META unflattening complexity: **avoided** — our schema
  pre-splits by topology (gauss/fibers/layers as separate groups),
  so no per-step unflattening step is needed. Pay parsing cost
  once in the transcoder.

#### OpenSees recorders (`opensees-expert` skill)

- ✅ All standard `-N` and `-E` tokens accounted for in canonical
  vocabulary
- ✅ Shell keyword swap (`section.fiber.stress` ↔ `material.fiber.stress`)
  hidden behind canonical `fiber_stress` name
- ✅ XML self-describing: parser uses XML headers when available
- ✅ TXT requires manifest: provided by `ResolvedRecorderSpec`
- ✅ Domain introspection via `ops.nodeDisp`, `ops.eleResponse`:
  used by Strategy B

#### STKO_to_python (`stko-to-python` skill)

- ✅ `MPCODataSet.nodes/elements` composite pattern: adopted
- ✅ Selection-set-based queries: mapped to `pg=`/`label=`
- ✅ Multi-partition transparency: matched
- 🟡 Component naming: STKO uses `'DISPLACEMENT'` (caps) and
  1-based components; we use lowercase explicit names
  (`displacement_x`). Translation happens in `MPCOReader`.

### Replacing the existing `Results` class

`src/apeGmsh/results/Results.py` is being rebuilt from scratch.
It is a VTK in-memory carrier with a fundamentally different shape;
no migration shims, no compatibility layer. The new module ships
under the same path with the full API described above.

VTU / PVD export remains available — it becomes a method on the
new class (`results.export.vtu(...)`), built on top of the same
backend reads everything else uses.

---

## Part II — Implementation Plan

### Build order rationale

The user explicitly wants to **rebuild the viewer from this**.
That makes the critical path:

```
schema → native writer → native reader → composite API → MPCO reader
```

Once that loop closes, they can read existing STKO files and the
viewer rebuild becomes its own project working against a stable
Results API. Recording capability layers on after — useful but not
blocking the rebuild.

The existing `src/apeGmsh/results/Results.py` is being replaced
wholesale. No shims, no compatibility layer. New module ships under
the same path.

### Top-level package layout (target)

```
src/apeGmsh/results/
├── __init__.py              # public API: Results, StageInfo, slab types
├── Results.py               # composite — user-facing class
├── _composites.py           # NodeResultsComposite, ElementResultsComposite, etc.
├── _slabs.py                # NodeSlab, GaussSlab, FiberSlab, LayerSlab dataclasses
├── _vocabulary.py           # canonical component names + shorthand expansion
├── _bind.py                 # bind() resolution (lenient — no hash check)
│
├── schema/
│   ├── __init__.py
│   ├── _native.py           # native HDF5 schema constants + path builders
│   └── _versions.py         # schema_version, parser_version
│
├── readers/
│   ├── __init__.py
│   ├── _protocol.py         # ResultsReader Protocol, ResultLevel enum
│   ├── _native.py           # NativeReader
│   └── _mpco.py             # MPCOReader (+ STKO name → canonical translation)
│
├── writers/
│   ├── __init__.py
│   ├── _native.py           # NativeWriter (file-based, append + bulk)
│   └── _cache.py            # cache key + manifest for transcoders
│
├── transcoders/
│   ├── __init__.py
│   ├── _txt.py              # OpenSees text recorder parser
│   ├── _xml.py              # OpenSees XML recorder parser
│   └── _recorder.py         # RecorderTranscoder (orchestrates)
│
└── capture/
    ├── __init__.py
    └── _domain.py           # DomainCapture context manager (Strategy B)
```

> **Drift from the original plan.** As shipped, `StageInfo` lives
> in `readers/_protocol.py` (no separate `_stage.py`), and the
> `export/` subpackage (`results.export.vtu/pvd`) was deferred —
> Phase 10 was not delivered.

Recorder spec lives next to the existing OpenSees bridge:

```
src/apeGmsh/solvers/
├── Recorders.py             # NEW — g.opensees.recorders composite
├── _recorder_specs.py       # NEW — declarative records + ResolvedRecorderSpec
└── _opensees_export.py      # MODIFIED — accept recorders=spec, emit commands
```

FEMData hashing lives next to `FEMData.py`:

```
src/apeGmsh/mesh/
└── _femdata_hash.py         # NEW — compute_snapshot_id(fem)
```

### Phase 0 — Schema lock + scaffolding

**Goal:** lock the native HDF5 schema, the canonical naming
vocabulary, the shorthand expansion table, and the FEMData
snapshot hash. Get the package skeleton in place so later phases
just fill modules.

**New files:**
- `src/apeGmsh/results/schema/_native.py` — path builders, attr keys
- `src/apeGmsh/results/schema/_versions.py` — `schema_version`,
  `parser_version` constants
- `src/apeGmsh/results/_vocabulary.py` — canonical component
  registry, dimension info (scalar/vector/tensor), category mapping,
  **shorthand expansion table** (one shorthand → list of canonical
  names, ndm/ndf clipping rule)
- `src/apeGmsh/results/_slabs.py` — `NodeSlab`, `ElementSlab`,
  `LineStationSlab`, `GaussSlab`, `FiberSlab`, `LayerSlab`
  dataclasses (numpy values + location index fields)
- `src/apeGmsh/results/readers/_protocol.py` — `ResultsReader`
  Protocol, `ResultLevel` enum, `StageInfo` dataclass (incl.
  `kind` field: "transient" | "static" | "mode")
- `src/apeGmsh/mesh/_femdata_hash.py` — `compute_snapshot_id(fem)`
  deterministic hash over canonical node IDs / coords /
  connectivity / PG membership

**Modified files:**
- `src/apeGmsh/mesh/FEMData.py` — `snapshot_id` lazy property
  (computed once, cached) backed by `_femdata_hash`

**Tests:**
- `tests/results/test_vocabulary.py` — every recorder token in
  `opensees-expert` skill maps to a canonical name; round-trip
  parses
- `tests/results/test_shorthand.py` — `"displacement"` →
  `["displacement_x", "displacement_y", "displacement_z"]` for
  ndm=3; clipped to xy for ndm=2; rotation excluded from
  `"displacement"` regardless of ndf; `"reaction"` expands to all
  6 components
- `tests/results/test_schema_paths.py` — path builder consistency
- `tests/results/test_femdata_hash.py` — same fem → same hash;
  permuted-but-equal connectivity → same hash; one-coord change →
  different hash; remesh → different hash

**Verification:** old `Results.py` still imports and tests pass —
this phase is purely additive (does not yet replace it).

### Phase 1 — Native writer + reader

**Goal:** synthetic round-trip works. Write a small dataset to a
`run.h5` file, read it back through `NativeReader`, get the same
arrays out.

**New files:**
- `src/apeGmsh/results/writers/_native.py` — `NativeWriter`:
  `open(path, fem)`, `begin_stage`, `write_nodes/elements/gauss/...`
  bulk writes, `append_step` for incremental, `end_stage`, `close`
- `src/apeGmsh/results/readers/_native.py` — `NativeReader`:
  implements `ResultsReader` protocol, lazy h5py reads, multi-partition
  stitching

**Modified files:**
- `src/apeGmsh/mesh/FEMData.py` — add `to_native_h5(group)` and
  `from_native_h5(group)` for the embedded snapshot

**Tests:**
- `tests/results/test_native_roundtrip.py` — write nodes/gauss/fibers
  to a temp `.h5`, read back, assert equality
- `tests/results/test_native_partitions.py` — write 3 partitions,
  read stitched
- `tests/results/test_native_stages.py` — write gravity + dynamic
  stages, read each independently
- `tests/results/test_native_modes.py` — write a mode stage
  (kind="mode", T=1, eigenvalue/frequency/period attrs), read back
- `tests/results/test_femdata_native_roundtrip.py` — FEMData →
  embedded → reconstructed FEMData equality (including snapshot_id)

**Verification:** schema spec from Phase 0 is consistent with what
the writer emits — schema version checks pass on read.

### Phase 2 — Composite API + bind

**Goal:** the user-facing `Results` class works end-to-end on
native files. PG/label selection resolves through bound FEMData.

**New files:**
- `src/apeGmsh/results/Results.py` — `Results` class, `from_native()`
  classmethod, `stage` / `modes` accessors, `bind()` / `inspect`
  methods
- `src/apeGmsh/results/_composites.py` —
  `NodeResultsComposite`, `ElementResultsComposite` (with nested
  `gauss`, `fibers`, `layers` sub-composites), `LineStationsComposite`,
  `NodalForcesComposite`
- `src/apeGmsh/results/_bind.py` — `validate_bind(fem, reader)`,
  hash-based comparison
- `src/apeGmsh/results/_stage.py` — stage-scoped Results returned
  by `results.stage(name)` / `results.modes[i]`

**Modified files:**
- `src/apeGmsh/results/__init__.py` — export `Results`,
  `StageInfo`, slab types (replace prior contents)

**Replace:** the existing `src/apeGmsh/results/Results.py` is
overwritten in this phase. No shims.

**Tests:**
- `tests/results/test_results_composite.py` — selection by `pg=`,
  `label=`, `ids=`; component/time slicing; stage scoping
- `tests/results/test_results_modes.py` — `results.modes` filters
  correctly; per-mode attrs accessible; mode shape is shape (1, N)
- `tests/test_results_bind.py` — auto-bind from embedded snapshot,
  explicit bind with matching hash, and `test_bind_accepts_mismatched_fem`
  (lenient contract — bind no longer raises on snapshot_id mismatch)
- `tests/results/test_results_self_contained.py` — open `run.h5`
  without external FEM, FEMData reconstructed from `/model/`,
  PG queries work

**Verification:** `Results.from_native("run.h5")` opens a
self-contained file with no apeGmsh session. The composite API
is stable enough that the viewer rebuild can proceed against it.

### Phase 3 — MPCO reader

**Goal:** read existing STKO `.mpco` files through the same
composite API. Validates that the protocol is genuinely
backend-agnostic.

**New files:**
- `src/apeGmsh/results/readers/_mpco.py` — `MPCOReader`,
  STKO-name → canonical translation table, partial FEMData
  synthesis from MPCO `MODEL/` group
- `src/apeGmsh/mesh/FEMData.py` — add `from_mpco_model(h5_group)`
  classmethod (partial reconstruction)

**Modified files:**
- `src/apeGmsh/results/Results.py` — add `from_mpco(path, fem=None)`
  classmethod

**Tests:**
- `tests/results/test_mpco_translation.py` — STKO `'DISPLACEMENT'`,
  components 1/2/3 → `displacement_x/y/z`; shell keyword swap;
  fiber section detection
- `tests/results/test_mpco_partial_fem.py` — partial FEMData has
  nodes / elements / PGs from selection sets; `labels` access
  raises with clear message
- `tests/results/test_mpco_real_file.py` — open a fixture `.mpco`
  (small one in `tests/fixtures/results/`), read displacement and
  stress, sanity-check shapes

**Fixture:** create one small `.mpco` from a 4-element cantilever
analysis in `tests/fixtures/results/cantilever_small.mpco`. Generated
once via openseespy; committed to repo.

**Verification:** the same composite test suite from Phase 2 passes
when the source is MPCO instead of native HDF5 — backend swap is
transparent.

### Phase 4 — Recorder spec composite

**Goal:** `g.opensees.recorders.*` declarative API works. Spec
resolves through FEMData and locks to its snapshot_id. No emission
yet.

**New files:**
- `src/apeGmsh/solvers/Recorders.py` — `Recorders` composite with
  `.nodes`, `.elements`, `.line_stations`, `.gauss`, `.fibers`,
  `.layers`, `.modal` methods
- `src/apeGmsh/solvers/_recorder_specs.py` —
  `RecorderRecord` (declarative, by PG/label),
  `ResolvedRecorderRecord` (concrete IDs after resolve),
  `ResolvedRecorderSpec` (collection, carries fem `snapshot_id`)

**Modified files:**
- `src/apeGmsh/solvers/OpenSees.py` — add `recorders` attribute,
  expose `g.opensees.recorders.resolve(fem)`
- `src/apeGmsh/solvers/_element_specs.py` — extend `_ElemSpec` with
  capability flags: `has_gauss: bool`, `has_fibers: bool`,
  `has_layers: bool`, `has_line_stations: bool`,
  `n_gauss_points: int | None`. Cross-reference the
  `mpco-recorder` skill's element compatibility catalog and the
  `opensees-expert` skill's element families to fill the table
  correctly. `mat_family="section"` is a strong hint for
  `has_fibers=True`; force/disp beam-column families have
  `has_line_stations=True`; tet/brick/quad families have
  `has_gauss=True`.

**Tests:**
- `tests/results/test_recorders_declare.py` — declarations stored
  per category, shorthand expansion (per the Phase 0 rule)
- `tests/results/test_recorders_resolve.py` — resolve flattens PGs
  to IDs; locks fem `snapshot_id`; raises if `gauss(...)` requested
  on elements without GPs; default-set toggle works
- `tests/results/test_recorders_drift.py` — re-mesh fem (new
  `snapshot_id`) → resolved spec refuses to re-emit / re-capture
  with a clear error
- `tests/results/test_recorders_validation.py` — invalid component
  names, missing PGs, conflicting cadences all raise
- `tests/results/test_element_capabilities.py` — every element in
  `_ELEM_REGISTRY` has a sensible capability annotation; spot-check
  ~10 representatives against the skill catalogs

**Verification:** `print(spec)` produces a readable summary of what
will be recorded; spec serializes to a small HDF5 manifest with
node/element ID arrays preserved as datasets.

### Phase 5 — Tcl / Python recorder emission

**Goal:** Strategy A part 1 — apeGmsh writes recorder commands into
the exported script.

**Modified files:**
- `src/apeGmsh/solvers/_opensees_export.py` — accept
  `recorders=spec` kwarg on `tcl()` / `py()`; emit one
  `recorder Node …` / `recorder Element …` per resolved record
- `src/apeGmsh/solvers/_recorder_specs.py` — add
  `to_tcl_commands(self) -> list[str]` and `to_python_commands(self)
  -> list[str]` methods on the resolved spec
- `src/apeGmsh/solvers/Recorders.py` — `spec.to_manifest_h5(path)`
  for sidecar serialization

**Tests:**
- `tests/results/test_recorder_emit_tcl.py` — snapshot the emitted
  commands for a representative spec (nodes + gauss + fibers)
- `tests/results/test_recorder_emit_py.py` — same for `ops.recorder(...)`
- `tests/results/test_recorder_manifest.py` — manifest round-trip
  (write → read → equal spec)

**Verification:** an emitted Tcl script runs in OpenSees and
produces expected `.out` / `.xml` files for a small fixture model
(test runs `OpenSees model.tcl` if available, skips if not — opt-in
via env var).

### Phase 6 — Recorder transcoder + cache

**Goal:** Strategy A part 2 — `Results.from_recorders(spec, "out/",
fem=fem)` parses the recorder output files into native HDF5 and
opens for read. Cache lives at `<project_root>/results/`.

**New files:**
- `src/apeGmsh/results/transcoders/_txt.py` — text recorder parser
  (column-major numpy load, manifest-driven decode)
- `src/apeGmsh/results/transcoders/_xml.py` — XML parser using the
  XML header for self-describing column metadata
- `src/apeGmsh/results/transcoders/_recorder.py` —
  `RecorderTranscoder` orchestrator: parse all recorder files,
  merge by category, write native HDF5
- `src/apeGmsh/results/writers/_cache.py` — cache key
  `(source_path, mtime, size, parser_version, fem_snapshot_id)`,
  manifest sidecar, invalidation logic. Cache root resolves to
  `<project_root>/results/` (override via env var
  `APEGMSH_RESULTS_DIR` or kwarg). Manifest format is **HDF5**, not
  JSON — carries numpy ID arrays from the resolved spec.

**Modified files:**
- `src/apeGmsh/results/Results.py` — `from_recorders(spec,
  output_dir, fem=None)` classmethod

**Tests:**
- `tests/results/test_txt_parser.py` — synthetic `.out` file →
  decoded by manifest → expected component arrays
- `tests/results/test_xml_parser.py` — synthetic `.xml` with header
  → decoded without manifest
- `tests/results/test_transcoder_roundtrip.py` — declare spec →
  emit Tcl → run OpenSees (skipped if unavailable) → transcode →
  read back via composite API
- `tests/results/test_cache_invalidation.py` — touch source file
  mtime → cache rebuilds; bump parser_version → cache rebuilds;
  fem_snapshot_id mismatch → refuses with clear error
- `tests/results/test_cache_location.py` — cache root resolution
  (cwd, env var override, explicit kwarg)
- `tests/results/test_manifest_hdf5.py` — manifest carries numpy
  ID arrays; round-trips losslessly

**Verification:** second `from_recorders` call on unchanged inputs
takes <100ms (cache hit, no re-parsing).

### Phase 7 — Domain capture (Strategy B)

**Goal:** in-process recording during an openseespy analysis,
including modal capture.

**New files:**
- `src/apeGmsh/results/capture/_domain.py` — `DomainCapture`
  context manager:
  - `__enter__` opens a `NativeWriter`, snapshots FEMData
  - `begin_stage(name)`, `end_stage()`
  - `step(t)` calls `ops.nodeDisp(...)`, `ops.eleResponse(...)` for
    each spec entry, buffers, flushes on threshold
  - `capture_modes(n_modes)` runs `ops.eigen(n_modes)`, extracts
    mode shapes via `ops.nodeEigenvector(...)`, writes one stage
    per mode with `kind="mode"` + eigenvalue/frequency/period attrs
  - `__exit__` closes writer

**Modified files:**
- `src/apeGmsh/solvers/_recorder_specs.py` — add
  `iter_domain_calls(self) -> Iterator[DomainCallSpec]` describing
  what `ops.*` to call for each resolved record

**Tests:**
- `tests/results/test_domain_capture.py` — mock `ops` module,
  capture nodes + gauss + fibers across two stages, verify written
  HDF5 matches expectations
- `tests/results/test_domain_capture_modes.py` — capture 5 modes,
  verify 5 stages written with correct kind/attrs/shapes
- `tests/results/test_domain_capture_real.py` — real openseespy
  analysis (cantilever), capture, read back, compare to
  recorder-based output (deltas should be < numerical tolerance)
- `tests/results/test_domain_capture_buffer.py` — flush threshold
  triggers correctly; long analysis stays bounded in RAM

**Verification:** capturing 10k steps on a 1k-node mesh stays
under 50 MB RAM throughout.

### Phase 8 — MPCO bridge in spec emission

**Goal:** Strategy C — same spec drives `recorder mpco` commands
for users who want STKO compatibility.

**Modified files:**
- `src/apeGmsh/solvers/_opensees_export.py` — accept `mpco=True`
  on `tcl()` / `py()`; emit `recorder mpco -file run.mpco -N <…> -E <…>`
  with translated tokens
- `src/apeGmsh/solvers/_recorder_specs.py` — add
  `to_mpco_command(self) -> str` (single recorder mpco line aggregating
  all records, since MPCO is one-recorder-per-file)

**Tests:**
- `tests/results/test_mpco_emit.py` — snapshot the `recorder mpco`
  command for a representative spec; tokens correctly mapped (e.g.
  `displacement` → `-N displacement`, `stress` → `-E stress` for
  solids, `-E material.stress` for shells if needed by class tag
  rules)

**Verification:** an emitted MPCO command runs in OpenSees
(skipped if unavailable) and produces a `.mpco` readable through
`Results.from_mpco(...)`.

### Phase 9 — Viewer rebuild (deferred — separate plan)

The viewer rebuild is the main consumer of this system but it is
its own project. After Phases 0-3 land and the composite API
stabilises, a dedicated `plan_viewer_rebuild.md` will plan the
viewer changes against it.

**This plan does not size or schedule the viewer work.** The
critical path for *this* plan stops at Phase 3 (MPCO reading
through the composite API). Everything else is downstream.

### Phase 10 — VTU / PVD export

**Goal:** parity with the old `Results` class for VTU output, on
top of the new backend.

**New files:**
- `src/apeGmsh/results/export/_vtu.py` — `results.export.vtu(path,
  *, components=None, time=None)` writes a single step
- `src/apeGmsh/results/export/_pvd.py` — `results.export.pvd(base,
  *, components=None)` writes a full time series

**Modified files:**
- `src/apeGmsh/results/Results.py` — `export` attribute exposing
  the two methods

**Tests:**
- `tests/results/test_export_vtu.py` — VTU file opens in pyvista,
  has expected point/cell data
- `tests/results/test_export_pvd.py` — PVD time series loads in
  ParaView (verified via pyvista round-trip)

### Phase 11 — Docs + examples

**Goal:** a user can discover and use the system from docs alone.

**Modified files:**
- `docs/api/results.md` — full API reference (replace existing
  contents)
- `docs/index.md` — add results section to TOC
- `docs/changelog.md` — entry for the rebuild

**New files:**
- `docs/examples/results_native.md` — domain capture walkthrough
- `docs/examples/results_recorders.md` — Tcl recorder workflow
- `docs/examples/results_mpco.md` — opening an MPCO file

**Verification:** `mkdocs serve` builds without warnings; an
external user could follow each example end-to-end without consulting
the source.

### Cross-phase concerns

#### Test fixtures

Generate once, commit to `tests/fixtures/results/`:
- `cantilever_small.mpco` — minimal MPCO file (4-element beam)
- `cantilever_small.h5` — native HDF5 equivalent (same data)
- `cantilever_small.out`, `.xml` — text/xml recorder output for
  the same model
- `frame_partitioned/` — 4-partition HDF5 dataset for partition
  stitching tests

A single `tests/fixtures/results/_generate_fixtures.py` script
recreates all of them; checked into the repo, run only when the
schema or model definitions change.

#### Performance benchmarks

Add `tests/results/bench_results.py` (skipped in normal CI, runs
under `pytest -m bench`):
- 1M-node displacement read time
- 100k-element stress (T=1000 steps) read time
- Memory ceiling during capture of a long dynamic analysis

Numbers go in this doc as targets after the first run.

#### Schema versioning

Phase 0 sets `schema_version = "1.0"`. Any change after Phase 1
goes through:
1. Bump `schema_version` to `"1.x"` (additive) or `"2.0"` (breaking)
2. `NativeReader` reads `1.x` for `x ≤ current_x` natively
3. Breaking changes go through a migration tool, not silent reads

#### Dependencies

This module adds no new third-party dependencies beyond what
apeGmsh already uses (`numpy`, `h5py`, `pandas`). `xarray` stays
optional — used only when the viewer asks for labeled axes.

### Decision log (re-affirmed in this plan)

- **Greenfield rebuild.** No migration shims for the old
  `Results.py`. Replaced wholesale in Phase 2.
- **Stages first-class.** Mirrors MPCO's `MODEL_STAGE[…]`. Time
  vector is per-stage.
- **Modes are stages with `kind="mode"`.** No separate `Modes`
  class. Mode shapes stored as displacements with T=1; per-mode
  scalars in stage attrs. Sugar accessor: `results.modes`.
- **Natural Gauss coordinates.** Global computed on demand from
  bound FEMData. Matches MPCO; correct under large deformation.
- **Group-by-(class_tag, int_rule).** Rectangular per-group arrays;
  reader stitches across groups.
- **Soft FEM coupling, hash-tagged.** Bare construction works;
  PG queries need bound FEM. The `snapshot_id` hash is computed and
  stored as metadata, but `bind()` never enforces equality — pairing
  is the user's responsibility (see `_bind.py:8-10`).
- **All recorders OpenSees gives us.** `nodes`, `elements`,
  `line_stations`, `gauss`, `fibers`, `layers`, `modal`.
- **Both component shorthand and explicit.** `"displacement"`
  expands to translations only (per the table in Layer 1);
  explicit per-component always valid.
- **Default recorder set on opt-in.**
- **MPCO bridge.** Spec drives MPCO recorder commands as Strategy C.
- **Cache at `<project_root>/results/`.** Visible at the project
  root, not hidden.
- **Manifest format is HDF5.** Carries numpy ID arrays losslessly.
- **No pickle support in v1.**
- **Viewer rebuild is a separate plan.** Critical path for this
  plan ends at Phase 3.

### Estimated phase footprint

These are rough estimates to give a sense of scale, not commitments:

| Phase | LOC (src) | LOC (tests) | New files |
|---|---:|---:|---:|
| 0  Schema + scaffolding + hash | ~500 | ~300 | 6 |
| 1  Native writer/reader | ~1200 | ~600 | 2 |
| 2  Composite API | ~1500 | ~800 | 4 |
| 3  MPCO reader | ~800 | ~500 | 1 |
| 4  Recorder spec | ~700 | ~400 | 2 |
| 5  Tcl/Py emission | ~300 | ~200 | 0 (mods only) |
| 6  Transcoder + cache | ~1100 | ~600 | 4 |
| 7  Domain capture (incl. modal) | ~700 | ~500 | 1 |
| 8  MPCO bridge emit | ~150 | ~100 | 0 (mods only) |
| 9  Viewer rebuild | (deferred — separate plan) |||
| 10 VTU/PVD export | ~300 | ~200 | 2 |
| 11 Docs | — | — | 3 docs |
| **Total (excl. viewer)** | **~7250** | **~4200** | **25** |

### Phase ordering check

Dependencies:
- 1 needs 0
- 2 needs 1
- 3 needs 0, 2 (composite API)
- 4 needs 0
- 5 needs 4
- 6 needs 1, 4, 5
- 7 needs 1, 4
- 8 needs 4, 5
- 10 needs 2
- 11 needs everything

**Critical path: 0 → 1 → 2 → 3.** That delivers the rebuild
target — viewer can read MPCO through the composite API. Phases
4-8 layer in recording capability after; Phase 10 + 11 close out;
Phase 9 (viewer) is its own plan.

---

## Part III — Open questions / future work

Items deferred from the design conversation, kept here so they
don't get lost:

1. **In-memory HDF5 escape hatch.** For ephemeral scripts where
   "data on disk" is overkill, allow `path=":memory:"` or
   `driver="core"` h5py mode. Not part of v1.
2. **Derived components.** Things like `von_mises_stress` can
   either be stored (transcoder computes once) or computed lazily
   (reader computes from `stress_xx` etc. on read). Initial plan:
   store. Lazy computation can be added later if storage gets
   tight.
3. **Multi-analysis support.** A single FEMData often runs many
   analyses (parametric studies). We assume one `Results` per
   analysis. A higher-level container (`ResultsCollection`) modeled
   after STKO's `MPCOResults` may be useful — deferred.
4. **Database backends.** SQLite or DuckDB for indexing across many
   `run.h5` files in a parametric study. Out of scope for v1.
5. **Streaming/append for live analysis.** Confirmed out of scope
   (post-only). If reinstated later, the writer protocol already
   supports `append_step` — readers would need a "follow tail" mode.
6. **Pickle support.** Out of scope for v1. If reinstated, would
   live as `Results.to_pickle()` / `from_pickle()` for STKO_to_python
   parity.
7. **Viewer rebuild.** The viewer is the main consumer of this
   system but its rebuild is a separate project. Once the Results
   composite API stabilises (Phases 0-3), a dedicated
   `plan_viewer_rebuild.md` will plan the viewer changes against
   it.

---

## See also

- [[apeGmsh_architecture]] — overall architecture (FEMData broker, Parts, etc.)
- [[apeGmsh_principles]] — invariants this design must respect
- `src/apeGmsh/mesh/FEMData.py` — composite pattern this design mirrors
- `src/apeGmsh/solvers/OpenSees.py` — bridge composite to extend with `recorders`
