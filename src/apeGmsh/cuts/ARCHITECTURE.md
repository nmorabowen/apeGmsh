# `apeGmsh.cuts` — section-cut spec producer

Architecture note for the `apeGmsh.cuts` subpackage. Lives next to the
code, like `opensees/architecture/`.

## Charter

apeGmsh produces, [STKO_to_python](https://github.com/nmorabowen/STKO_to_python)
consumes.

STKO_to_python (the MPCO post-processor) already ships a battle-tested
section-cut kernel: beam / shell / solid integration, side-aware
shared-edge resolution, Cyrus-Beck + Sutherland-Hodgman polygon
clipping, layered-shell per-fiber views, and three universal validators
(`consistency_check`, `compare_to`, `moment_about`). Reimplementing any
of that here is months of work for zero engineering payoff.

What apeGmsh *uniquely* has is CAD-level geometry and named physical
groups — exactly what's painful to derive from raw MPCO output. So:

**apeGmsh's job is to produce `SectionCutSpec` objects from physical
groups; STKO_to_python's job is to consume them.**

The seam is `STKO_to_python.cuts.SectionCutSpec`, which is already
designed as a portable carrier: dataset-agnostic, picklable, hashable,
validated at construction. We just feed it.

## Data flow

```
Gmsh model + physical groups + model.h5 (Phase 8.6 fem_eids)
        │
        │  apeGmsh: derive plane (from PG surface), resolve
        │  element_ids (from PG → FEM eid → ops_tag mapping)
        ▼
SectionCutDef                              ← apeGmsh.cuts
        │
        │  .to_spec()  (lazy STKO import)
        ▼
STKO_to_python.cuts.SectionCutSpec         ← seam
        │
        │  pickle, persist, ship to batch worker
        ▼
ds.section_cut(spec=spec, model_stage=...)
        │
        ▼
SectionCut (F, M, time, …)                  ← STKO_to_python
```

## Strategic decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Where this lives | `apeGmsh.cuts` (new top-level subpackage) | Cuts are neither pure-mesh nor pure-results; they bridge. Top-level keeps the import path clean. |
| 2 | STKO_to_python dependency | **Optional, lazy import** at `.to_spec()` time | STKO pulls h5py / pandas / matplotlib transitively. Modeling-only users shouldn't pay for that. |
| 3 | `Plane` representation | Store as `(point_tuple, normal_tuple)` in `SectionCutDef`; construct STKO `Plane` only in `.to_spec()` | Don't require STKO to construct a Def or to pickle it. |
| 4 | v1 scope | Spec producer only | No result consumption, no viewer overlay, no recorder emission. |
| 5 | FEM eid → ops_tag bridge | Read from `model.h5` (`/opensees/element_meta/{type}/fem_eids` parallel to `ids`) | Phase 8.6's chosen source-of-truth. Survives session boundaries; matches Phase 8.7's viewer direction. |

## Package layout

```
src/apeGmsh/cuts/
├── ARCHITECTURE.md          ← this file
├── __init__.py              ← re-exports SectionCutDef (and SectionSweepDef once it exists)
├── _defs.py                 ← SectionCutDef frozen dataclass + to_spec()
├── _optional_stko.py        ← lazy STKO_to_python import with clean error
├── _planes.py               ← plane builders (horizontal / vertical / 3-point / SVD-fit / from-PG)
├── _polygons.py             ← convex hull + bounding_polygon_from_physical_surface
├── _tag_map.py              ← FemToOpsTagMap reading model.h5
└── _sweeps.py               ← SectionSweepDef (sequence of SectionCutDef, one filter, many planes)
```

Tests mirror under `tests/cuts/test_*.py`.

## Public API (v1 — spec producer)

```python
from apeGmsh.cuts import SectionCutDef

# Manual construction — caller already has plane and ops_tags
cut_def = SectionCutDef(
    plane_point=(0.0, 0.0, 2500.0),
    plane_normal=(0.0, 0.0, 1.0),
    element_ids=(101, 102, 103),       # OpenSees tags
    side="positive",
    label="Story 3 base shear",
)

# Round-trip to STKO (lazy import — requires STKO_to_python installed)
spec = cut_def.to_spec()
assert spec.label == "Story 3 base shear"

# Persistence (uses apeGmsh-side pickle; no STKO needed to save/load)
cut_def.save_pickle("story3.pkl")
restored = SectionCutDef.load_pickle("story3.pkl")
```

The ergonomic constructor (Phase 4):

```python
# Physical-group driven — the real API
cut_def = SectionCutDef.from_planar_pg(
    plane_pg="diaphragm-3",       # PG defining the cut plane
    elements_pg="tower-cols",     # PG defining the element filter
    fem=fem,                       # FEMData for FEM-eid lookup
    model_h5="path/to/model.h5",   # for FEM↔ops_tag bridge
)
# .label auto-set to "plane=diaphragm-3, elements=tower-cols"

# Or, plane from elsewhere:
cut_def = SectionCutDef.from_plane_and_pg(
    plane=plane_horizontal(z=2500.0),
    elements_pg="tower-cols",
    fem=fem,
    model_h5="path/to/model.h5",
    label="Story 3 base shear",
)
```

## Phase roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Package scaffold + `SectionCutDef` + `.to_spec()` + pickle | **done** |
| 2 | `plane_from_physical_surface()` + Plane convenience wrappers | **done** |
| 3 | `FemToOpsTagMap` from `model.h5` | **done** |
| 4 | `SectionCutDef.from_plane_and_pg` / `.from_planar_pg` builders | **done** |
| 5 | `SectionSweepDef` + `from_pg_pattern` | **done** |
| v2.1 | `bounding_polygon_from_physical_surface` + `with_bounding=True` flag | **done** |
| v3.1 | `SectionSweepDef.from_pg_glob(pattern=...)` + `with_bounding` propagation | **done** |
| v2.2 | Viewer overlay — `SectionCutDiagram` Layer kind + filter highlight | **done** |
| v2.3 | `SectionCutDef.preflight(fem)` validator — drift checks | **done** |
| Phase 4 | Add-Diagram dialog file picker for pickled cuts / sweeps | **in progress** |
| v5 | `DriftDef` / `DriftSweepDef` — node-pair drift carrier | **in progress** |
| v4 | `model.h5` persistence — `/opensees/cuts/` writer/reader + viewer auto-load + dialog h5-source mode | **in progress** |

v4 picks up `model.h5` persistence — see the
`## v4 — Cuts persisted in `model.h5`` section below. Live edit of
persisted cuts, drift persistence under `/opensees/drifts/`, and the
`apeGmsh.outputs` reorganization remain out of scope until concrete
call sites appear.

## v2.2 — Viewer overlay

Renders planned cuts in `results.viewer()` so a user can see the
plane, identify the kept side, and inspect which elements the filter
resolves to. Out of scope: F/M resultants (post-analysis), per-step
animation (cuts are static), STKO kernel changes.

### Integration choice — Diagram Layer, not Overlay

A cut becomes a `Diagram` of `kind="section_cut"` slotted into the
existing Geometries → Diagrams → Layers outline. One `SectionCutDef`
maps to one Layer; a `SectionSweepDef` fans out into N Layers at add
time. Rationale:

- Reuses the four-method Diagram lifecycle (attach / update / detach
  / settings_widget), the registry's add/remove/replace plumbing, the
  outline tree's selection routing, and the dispatcher's GATE pump
  for per-Geometry visibility — no new infrastructure.
- The "non-Results-data diagram kind" precedent already exists:
  `LoadsDiagram` reads `fem.nodes.loads`, not a Results composite.
  `SectionCutDiagram` is the same shape — it reads a `SectionCutDef`
  carried on its own style record.
- One Layer per cut means individual toggle / select / highlight
  comes for free. Fanning a sweep into N Layers preserves that
  granularity at the cost of a longer outline list.

### Locked design decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Primary viewer | `results.viewer()` first; `g.mesh.viewer(fem=)` reuse deferred to Phase 1c | The outline hierarchy the brief calls out lives in `results.viewer()` post-PR-#54 |
| D2 | Quad extent | `bounding_polygon` when set; else clip plane to the filter elements' AABB (not the model AABB) | Bounded case matches kernel exactly. Unbounded case shows roughly where the cut actually integrates — model-AABB would be a visual fiction on large structures |
| D3 | Kept-side visual | Two-tone quad (front=kept color, back=discarded color) + small fixed-size normal arrow at quad centroid | Half-space translucent fill dominates with multiple cuts; two-tone uses VTK's native front/back face coloring with the arrow as the orbit-edge-on fallback |
| D4 | Filter highlight | Per-card "Show filter elements" checkbox on the layer's `settings_widget()` — explicit toggle. The v2 outline pivot dropped per-Layer selection events, so auto-highlight-on-select is deferred to Phase 2 when a per-Layer selection model lands. | Always-on crowds the scene and fights contour/vector layers' coloring |
| D5 | Filter highlight rendering | Separate `scene.grid.extract_cells(...)` actor drawn on top of substrate, uniform color | Avoids stomping substrate scalars used by other layers in the same composition |
| D6 | Deformation | Cuts are reference-config only; ignore the Geometry's deform field | The cut plane was defined against the input model — making it follow deformation is conceptually muddled |
| D7 | OpenSees → FEM eid bridge | `ResultsDirector` gains a `model_h5` setter + lazy `tag_map: Optional[FemToOpsTagMap]` property; Layers consume it | Tag map is per-viewer-session state, not per-layer. Avoids re-reading model.h5 once per cut |
| D8 | Spec carrier | `SectionCutStyle(cut: SectionCutDef, ...)` — frozen-in-frozen, used as `DiagramSpec.style` | Keeps the existing `DiagramSpec` record untouched; cut data rides where styles already ride |
| D9 | Phase-1 ingress | `director.add_section_cut(def, ...)`, `director.add_section_cut_sweep(sweep, ...)`, and `results.viewer(cuts=[...])` kwarg | Covers in-process and `blocking=False` subprocess launches. File-picker dialog deferred to Phase 4 |
| D10 | Live edit | Deferred to Phase 2 — `settings_widget()` rebuilds a fresh spec, `registry.replace(old, new)` swaps it | Frozen spec preserves the persistence story; replace is already in the registry |

### Data flow at attach

```
SectionCutDef (style.cut)
  │
  │  attach(plotter, fem, scene)
  ▼
1. Resolve OpenSees tags → FEM eids via director.tag_map
2. Compute quad vertices:
     if cut.bounding_polygon:   use it directly
     else:                       clip cut.plane to filter-AABB
3. Build polydata (quad, front-face=kept, back-face=discarded)
4. Add normal-arrow actor at quad centroid
5. Cache resolved FEM eids for filter-highlight on selection
```

### Package layout additions

```
src/apeGmsh/viewers/diagrams/
├── _section_cut.py          ← SectionCutDiagram (new)
├── _styles.py               ← SectionCutStyle (added)
├── _kind_catalog.py         ← "section_cut" entry (added)
└── __init__.py              ← re-export (added)

src/apeGmsh/viewers/diagrams/_director.py
└── model_h5 setter, tag_map property, add_section_cut*, add_section_cut_sweep*  (added)

src/apeGmsh/viewers/results_viewer.py
└── cuts= kwarg threaded into director at boot  (added)
```

Tests mirror under `tests/cuts/test_viewer_*.py` and
`tests/viewers/test_section_cut_diagram.py`.

### Phase roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| v2.2-0 | This architecture section | **in progress** |
| v2.2-1 | Static quad rendering + programmatic ingress + tests | pending |
| v2.2-1b | Filter highlight on outline selection + tests | pending |
| v2.2-1c | `g.mesh.viewer(fem=)` reuse (optional) | deferred |
| v2.2-2 | Live edit via settings_widget + replace | deferred (v3 of viewer plan) |
| v2.2-3 | Session JSON persistence for cut layers | deferred |
| v2.2-4 | Add-layer dialog with `.pkl` file picker | deferred |

## Acceptance test (north star)

A single integration test, skipped if STKO_to_python is not installed:

```python
def test_section_cut_end_to_end(tmp_path):
    # 1. Build a fixture tower with named "diaphragm-3" and "tower-cols" PGs
    g = build_fixture_tower(tmp_path)
    fem = g.mesh.queries.get_fem_data(dim=3)
    ape = apeSees(fem)
    # ... wire recorders, run analysis to tmp_path/"Results.mpco"
    ape.h5(tmp_path / "model.h5")

    # 2. Build a SectionCutDef from physical groups
    cut_def = SectionCutDef.from_planar_pg(
        plane_pg="diaphragm-3",
        elements_pg="tower-cols",
        model_h5=tmp_path / "model.h5",
        label="Story 3 base shear",
    )
    spec = cut_def.to_spec()

    # 3. Consume with STKO_to_python — kernel-side validator is the oracle
    from STKO_to_python import MPCODataSet
    ds = MPCODataSet(str(tmp_path), "Results", verbose=False)
    cut = ds.section_cut(spec=spec, model_stage="MODEL_STAGE[1]")

    assert cut.F.shape[1] == 3
    ok, _ = cut.consistency_check(ds)      # Newton's 3rd law
    assert ok
```

If this passes, the seam holds.

## Out of scope for v1

- Reimplementing any kernel math. Use STKO's `Plane.intersect_*`, the
  beam/shell/solid kernels, the polygon clipping. No exceptions.
- Reading MPCO output. apeGmsh users open the dataset themselves.
- Recorder emission (computing the cut during analysis). Probably never.
- Non-convex bounding polygons. STKO doesn't support them; we won't either.

## Dependencies

- **Required at construction:** numpy. That's it.
- **Required for `to_spec()`:** `STKO_to_python` (optional, lazy-imported).
- **Required for `from_planar_pg(...)` (Phase 4):** `h5py` (already a hard apeGmsh dep via `model.h5`).

## v2.3 — Preflight validator

A `SectionCutDef` can drift away from the FEM that produced it: the
user re-meshes, renames a physical group, drops a column, edits the
diaphragm geometry. The pickled cut still loads — it doesn't know any
of this happened — and `.to_spec()` happily round-trips it. Errors
only surface much later, at consumption time inside STKO, with a stack
trace that doesn't point back to the stale cut.

`preflight()` is a structured validator that checks a cut against a
live `FEMData` (optionally plus a `model.h5` for the OpenSees-tag
bridge) and returns a report of errors and warnings. Doesn't auto-fix.
Doesn't mutate the cut. Pure inspection.

### Locked design decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| P1 | Inputs | `preflight(fem, *, model_h5=None, tag_map=None)` — either `model_h5` or a cached `tag_map` enables ops-tag checks; both omitted skips E1/E2; both supplied raises `ValueError` | Sweep callers cache one `FemToOpsTagMap` and reuse across N cuts. `fem`-only mode supports "I haven't emitted model.h5 yet, just want geometry sanity." |
| P2 | Severity model | Two buckets — errors (cut will produce wrong/empty results) vs warnings (suspicious but possibly legitimate) | Matches the brief. Avoids over-engineered severity ladders. |
| P3 | Error list | **E1** ops tag not in tag map; **E2** ops tag → FEM eid that's no longer in `fem.elements`; **E3** bounding polygon vertex off cut plane beyond `tol`; **E4** filter resolves to zero existing elements | All four are conditions where the cut, as written, cannot produce a correct result. |
| P4 | Warning list | **W1** bulk AABB of filter elements doesn't straddle the cut plane (all nodes one side, by `tol`) | The "cut would integrate zero area" case — usually a mistake, but legitimate for sweeps near a structure edge. Warning, not error. |
| P5 | Tolerance | One `tol: float = 1e-6` kwarg, used for both E3 (polygon-on-plane) and W1 (AABB-straddle) | Matches `plane_from_physical_surface`. Mismatched tolerances confuse users. |
| P6 | Module layout | New `_preflight.py` holds `PreflightIssue` + `PreflightReport`; `SectionCutDef.preflight` / `SectionSweepDef.preflight` are thin methods that import and dispatch | Keeps `_defs.py` focused on the dataclass; preflight grows independently. |
| P7 | Sweep return type | `SectionSweepDef.preflight(...) -> tuple[PreflightReport, ...]` — one report per cut in cut order | Caller composes aggregation; no wrapper class until a real call site demands one. |
| P8 | Dependencies | numpy only — no scipy, no STKO, no h5py beyond what `FemToOpsTagMap.from_h5` already pulls | Honors the "cuts importable without scipy" constraint. |

### Issue codes

| Code | Severity | Trigger | What the user usually did |
|------|----------|---------|----------------------------|
| E1 | error | `ops_tag in cut.element_ids` not present in tag map | Re-emitted model.h5 with new tag namespace; old cut is stale |
| E2 | error | `ops_tag` resolves to `fem_eid` not in `fem.elements.ids` | FEM eid was deleted from the mesh after the cut was built |
| E3 | error | `bounding_polygon` vertex distance from plane > `tol` | Edited plane or polygon independently; they no longer agree |
| E4 | error | No filter elements exist in the current FEM | All filter elements were removed; cut is empty |
| W1 | warning | Filter node AABB lies entirely one side of plane | Either a deliberate edge-of-structure sweep, or a forgotten plane-elevation update |

### Report shape

```python
@dataclass(frozen=True)
class PreflightIssue:
    code: str                          # "E1", "E2", "E3", "E4", "W1"
    severity: Literal["error", "warning"]
    message: str                       # human-readable, one line
    detail: Mapping[str, object] | None = None  # structured payload

@dataclass(frozen=True)
class PreflightReport:
    cut_label: str | None
    errors: tuple[PreflightIssue, ...]
    warnings: tuple[PreflightIssue, ...]

    @property
    def ok(self) -> bool: ...          # no errors (warnings don't block)

    def raise_for_errors(self) -> None: ...  # PreflightError if any

    def __str__(self) -> str: ...      # multi-line summary
```

### Data flow

```
SectionCutDef (carrying OpenSees tags)
    │
    │  preflight(fem, model_h5=... or tag_map=...)
    ▼
1. E3 — polygon-on-plane: vertex distance from plane vs tol
2. If tag_map present:
    a. E1 — every ops_tag in cut.element_ids must be in tag_map
    b. Resolve survivors to fem_eids
    c. E2 — every resolved fem_eid must be in fem.elements.ids
    d. E4 — at least one fem_eid must remain
   Else:
    a. (skip E1/E2/E4; document this in the report?)
3. If we resolved fem_eids successfully:
    a. Collect unique node ids from filter elements' connectivity
    b. Look up coords in fem.nodes
    c. W1 — signed distance min/max must straddle 0 (within tol)
4. Build PreflightReport(errors, warnings)
```

### Package-layout addition

```
src/apeGmsh/cuts/
└── _preflight.py            ← new: PreflightIssue, PreflightReport, _run_checks
```

`_defs.py` gains `SectionCutDef.preflight(...)`. `_sweeps.py` gains
`SectionSweepDef.preflight(...)`. `__init__.py` re-exports
`PreflightIssue` and `PreflightReport` (the report types — `_run_checks`
stays internal).

### Phase roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| v2.3-0 | This architecture section | **done** |
| v2.3-1 | `PreflightIssue` + `PreflightReport` + `SectionCutDef.preflight` + `SectionSweepDef.preflight` + tests | **done** |

### Out of scope for v2.3

- Auto-fixing drifted cuts (e.g. re-resolving filter elements after a
  re-mesh). The Def is frozen; "fix" means construct a new one.
- Validating against `model.h5` content beyond the tag map (e.g. is
  the cut plane near a recorder?). The recorder layer is a different
  concern.
- Strict-mode preflight that promotes warnings to errors. Caller can
  do this on the report.

## Phase 4 — Add-Diagram dialog file picker

A user with a pickled `SectionCutDef` (or `SectionSweepDef`) — saved
from a previous session, shared between users, generated by a script
— should be able to load it into the results viewer through the same
Add-Diagram dialog that drives every other Layer kind. Programmatic
ingress (D9 in v2.2) is already covered by `results.viewer(cuts=...)`
and `director.add_section_cut(...)`; Phase 4 adds the GUI route.

The dialog must surface the v2.3 preflight report so the user catches
drift *before* clicking OK. If the cut won't work against the current
FEM, OK is gated and the report explains why.

### Locked design decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| F1 | Where the file picker lives | Extend the existing `AddDiagramDialog` — new `_KindEntry(kind_id="section_cut", ...)` | Brief said so. One entry point keeps UX consistent with every other diagram kind. |
| F2 | Layout switching | When `kind = section_cut`, hide Stage / Topology / Averaging / Component / Preset / Selector / Selector-name rows; show File-picker + Model.h5 + Preflight-summary rows; keep Label edit | The Results-data rows are meaningless for a static cut. |
| F3 | Cut vs Sweep | One kind entry, one file picker. After `load_pickle`, dispatch on `isinstance(obj, SectionSweepDef)` → `director.add_section_cut_sweep`, else `add_section_cut` | `SectionSweepDef.load_pickle` already exists; no extra UI surface needed. |
| F4 | model_h5 sourcing | Auto-fill from `director.model_h5` if already set (from `results.viewer(model_h5=...)`); otherwise show editable path field with a Browse button | Don't make the user re-enter what the viewer already knows. |
| F5 | Preflight strategy | Auto-preflight on every file pick or model_h5 change. Run once more at OK as belt-and-braces (catches any user edit between pick and click). Errors block OK; warnings don't. | v2.3 unlock — pay off the work. No "force add" escape hatch (F7). |
| F6 | Preflight summary widget | Read-only `QTextEdit` (monospace, ~6 visible lines) showing `str(report)`. Surrounding status label shows OK / WARNINGS / ERRORS with colored swatch. | Inline summary beats a popup; user can copy-paste failure detail. |
| F7 | "Force add" escape | None. If preflight errors, OK stays disabled and the user must re-pickle from a fresh `from_planar_pg`. | Allowing it defeats the v2.3 promise. |
| F8 | Sweep summary | A single aggregate line: `"3 cuts: 2 ok, 1 with W1 warning, 0 errors"` plus the concatenated `str(report)` for each underlying cut. OK still gated on **any** error across the sweep. | One readable widget; per-cut detail still visible. |
| F9 | What fires at OK | `director.add_section_cut(cut, model_h5=...)` (or `add_section_cut_sweep`) — exactly the same code path as the programmatic ingress. No new director method. | Reuses tested infra. |
| F10 | File pattern | QFileDialog filter ``"Pickled cut (*.pkl *.pkl.gz)"``. `load_pickle` already auto-detects gzip by suffix. | Both formats already supported by `_defs.py` / `_sweeps.py`. |

### Layout when `section_cut` is the chosen kind

```
┌─────────────────────────────────────────────────────────┐
│ Kind:           [ Section cut          ▼]               │
│ File:           [ /path/to/story3.pkl  ] [Browse…]      │
│ Model.h5:       [ /path/to/model.h5    ] [Browse…]      │
│ Preflight:      [● OK]                                  │
│                ┌────────────────────────────────────┐   │
│                │ PreflightReport — story 3 base ... │   │
│                │   (no issues)                      │   │
│                └────────────────────────────────────┘   │
│ Display label:  [(optional)            ]                │
│                                                         │
│                                       [ OK ] [Cancel]   │
└─────────────────────────────────────────────────────────┘
```

Status swatch colors:
- green dot, "OK" — no errors, no warnings
- amber dot, "WARNINGS (N)" — warnings only (OK enabled)
- red dot, "ERRORS (N)" — at least one error (OK disabled)

### Data flow at OK

```
1. Load pickle           → SectionCutDef OR SectionSweepDef
2. Resolve model_h5      → director.model_h5 OR file picker text
3. Preflight (rerun)     → PreflightReport(s); errors → reject with message
4. Branch on type:
     SectionCutDef    → director.add_section_cut(cut, model_h5=...)
     SectionSweepDef  → director.add_section_cut_sweep(sweep, model_h5=...)
5. Dialog accepts; Diagrams tab refreshes via existing registry signals
```

### Test plan

- Construction: section_cut kind appears in kind combo (11 entries)
- Row-switching: picking section_cut hides Stage/Component/etc.; picking back restores them
- File-picker: setting a valid `.pkl` path loads the cut into internal state
- Preflight (mocked):
  - clean → OK enabled, "OK" status
  - W1-only → OK enabled, "WARNINGS (1)" status
  - E-anything → OK disabled, "ERRORS (N)" status
- model_h5 autofill: when `director.model_h5` already set, the dialog pre-fills the path
- Sweep fan-out: picking a sweep `.pkl` triggers `director.add_section_cut_sweep` (via patching)
- OK dispatch: single cut → `director.add_section_cut` called once with the loaded def

Tests patch `QFileDialog.getOpenFileName`, `SectionCutDef.load_pickle`, and `director.add_section_cut*` rather than driving full file I/O — UI logic is what we're validating.

### Files touched

```
src/apeGmsh/viewers/ui/_add_diagram_dialog.py    ← kind entry + layout switch + load + preflight + OK gate
src/apeGmsh/cuts/ARCHITECTURE.md                  ← this section
tests/viewers/test_add_diagram_section_cut.py     ← new file (3-5 focused tests)
```

### Phase 4 roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| P4-0  | This architecture section | **in progress** |
| P4-1  | Dialog wiring + tests | pending |

### Out of scope for Phase 4

- A force-add escape hatch (F7)
- File-picker for cuts in `g.mesh.viewer()` — that's Phase 1c and lives elsewhere
- Drag-and-drop file ingestion onto the viewer window
- Browsing for cuts inside `model.h5` — that's v4 of the roadmap

## v5 — `DriftDef` / `DriftSweepDef`

A sibling type to `SectionCutDef`: instead of integrating force over a
planar slice through elements, drift extracts the relative
displacement between two reference nodes — optionally projected onto
an axis, optionally normalized by a story height. The natural carrier
for inter-story drift in building models, but useful anywhere you
need `node_A.u - node_B.u` extracted from MPCO results.

`DriftDef` is the unit; `DriftSweepDef` is a frozen sequence of them
(typically one per story) for building-level profiles. Both are pure
data — no `.to_spec()`, no `.extract()` in v5. Same philosophy as
`SectionCutDef` v1: build the carrier; defer consumption.

### Why this lives in `apeGmsh.cuts`

The subpackage name is a slight misnomer once drift lands — drift is
not a "cut". But:

- The plumbing is identical: frozen dataclass + builders + pickle +
  preflight validator against a live FEM.
- The user-facing import (`from apeGmsh.cuts import DriftDef`) is
  still natural — "cuts" reads as "post-process specs" in spirit.
- A reorg to `apeGmsh.outputs` (or similar) becomes worth doing only
  when a third post-process type lands; deferring keeps v5 focused.

Re-evaluate when v6 arrives.

### Locked design decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Names | `DriftDef`, `DriftSweepDef` | Mirror `SectionCutDef` / `SectionSweepDef`. |
| D2 | Stored fields | `top_node: int`, `bottom_node: int`, `direction: Vec3 \| None`, `story_height: float \| None`, `label: str \| None` | Minimal carrier; everything else is computed downstream. |
| D3 | `direction` semantics | `None` → caller wants the raw Δu vector; `Vec3` → caller wants Δu projected onto the (auto-unit-normalized) axis | Common case is horizontal-only ((1,0,0) or (0,1,0)); raw vector is for 3-D analytics. Mirrors `SectionCutDef`'s auto-normalized normal. |
| D4 | Construction-time validation | top_node ≠ bottom_node; `direction` (if set) is nonzero; `story_height` (if set) is positive; node IDs coerced to `int` | Things that are always wrong regardless of FEM — fail at construction. |
| D5 | Builders | `from_node_pair(top, bottom, …)` (already covered by `__init__`, but kept as a named classmethod for symmetry with cuts); `from_pgs(top_pg, bottom_pg, fem, …)` | Two-PG builder matches how users tag floor reference points (CM, diaphragm node, etc.). |
| D6 | Multi-node PG | `from_pgs` raises `ValueError` if either PG resolves to >1 node | Silent averaging would hide a tagging bug; explicit error tells the user to tag a single representative node. |
| D7 | Story-height auto-derivation | Never auto-derived. `.drift_ratio()` is out of scope for v5; story_height is purely metadata until a consumer needs it | Pure carrier in v5. |
| D8 | `.to_spec()` / `.extract()` | **Out of scope for v5.** STKO has no `DriftSpec`; no demanded consumer | Matches `SectionCutDef` v1. Add when a real call site appears. |
| D9 | Preflight | Shared `PreflightReport` (no new report class); codes prefixed `D-` to avoid clashing with cut codes | Option α from the v5 design pass. One report type for both subjects keeps `apeGmsh.cuts` cohesive. |
| D10 | Preflight issue catalog | **D-E1** top_node not in `fem.nodes`; **D-E2** bottom_node not in `fem.nodes`; **D-W1** top and bottom coordinates coincident within `tol` | Minimal set. Construction-time validation already covers the "always wrong" cases. |
| D11 | Sweep shape | `DriftSweepDef` = frozen `tuple[DriftDef, ...]`; container protocol; `from_pg_pairs(pg_pairs=[(top, bot), …], fem=…)` builder; `elevations(axis="z", *, fem)` returns one float per drift (top-node coord along axis) | Mirrors `SectionSweepDef` exactly; same pickle support. |
| D12 | Sweep preflight | Returns `tuple[PreflightReport, ...]` — one per drift, in sweep order | Same contract as `SectionSweepDef.preflight`. |
| D13 | Module layout | New `_drift.py` next to `_defs.py`. Preflight checks live alongside `DriftDef.preflight`; reuses `PreflightIssue` and `PreflightReport` from `_preflight.py` | Keeps the cuts subpackage organized by topic. |

### Public API

```python
from apeGmsh.cuts import DriftDef, DriftSweepDef

# Direct construction
d = DriftDef(
    top_node=1234,
    bottom_node=5678,
    direction=(1.0, 0.0, 0.0),    # project onto x; None → raw Δu vector
    story_height=3000.0,           # optional metadata
    label="story 3 drift X",
)

# Physical-group driven (each PG must resolve to exactly one node)
d = DriftDef.from_pgs(
    top_pg="floor-3-CM",
    bottom_pg="floor-2-CM",
    fem=fem,
    direction=(1.0, 0.0, 0.0),
    story_height=3000.0,
)

# Building-level profile
sweep = DriftSweepDef.from_pg_pairs(
    pg_pairs=[("floor-1-CM", "floor-0-CM"),
              ("floor-2-CM", "floor-1-CM"),
              ("floor-3-CM", "floor-2-CM")],
    fem=fem,
    direction=(1.0, 0.0, 0.0),
)

# Drift against the current FEM
report = d.preflight(fem)
assert report.ok
reports = sweep.preflight(fem)  # tuple[PreflightReport, ...]

# Plot top-node elevations for a drift profile
ys = sweep.elevations(axis="z", fem=fem)

# Pickle round-trip
d.save_pickle("story3.pkl")
restored = DriftDef.load_pickle("story3.pkl")
```

### Phase roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| v5-0 | This architecture section | **done** |
| v5-1 | `DriftDef` + `DriftSweepDef` + preflight + tests | **done** |

### Out of scope for v5

- `.to_spec()` — STKO_to_python has no `DriftSpec` counterpart; build when there's a real call site.
- `.extract()` against an MPCODataSet — same reason. Currently the user computes drift themselves from nodal displacement histories.
- `.drift_ratio()` — needs a story_height consumer; trivial to add when one exists.
- Viewer overlay — drift has no spatial extent; a "drift between these two nodes" decoration is a different design pass.
- Renaming `apeGmsh.cuts` → `apeGmsh.outputs`. Revisit when v6 arrives.

## v4 — Cuts persisted in `model.h5`

Cuts shipped through v5 are picklable, share-by-file artifacts. They
live on disk only when the user (or a script) explicitly calls
`save_pickle(...)`, and they reach the viewer through either
`results.viewer(cuts=[...])` (programmatic) or the Phase-4 file
picker (GUI). Neither route ties the cuts to the model definition
they were built against — they survive in their own `.pkl` files,
which is great for sharing and bad for keeping the cut + model in
lockstep.

v4 makes `model.h5` the canonical store. A user who runs
`ape.h5(path, cuts=[...], sweeps=[...])` writes cuts under
`/opensees/cuts/` alongside the model definition; the next viewer
session against that same file auto-loads them. Cuts and the model
they were preflighted against now travel together; sharing the
`.h5` shares the cuts.

### Why `/opensees/cuts/`, not `/cuts/`

`SectionCutDef.element_ids` are OpenSees tags. The whole preflight /
`to_spec()` pipeline depends on `/opensees/element_meta/{type}/fem_eids`
to map them back to FEM eids. Stashing cuts at the file root would
suggest they're solver-neutral; in practice they are not. A future
solver zone (`/aster/...`, `/abaqus/...`) carrying its own post-process
specs would land under its own namespace and not collide.

### Locked design decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| H1 | Where it lives | `/opensees/cuts/` and `/opensees/sweeps/`, nested in the OpenSees zone | `element_ids` are OpenSees tags; schema location matches the data dependency |
| H2 | Sweep cuts layout | Nested — each sweep group owns its own `cuts/` subgroup | Self-contained; cleaner reader; the rare standalone-vs-sweep dedup isn't worth a cross-reference layer |
| H3 | Cut and sweep naming | Positional: `cut_0`, `cut_1`, …; `sweep_0`, `sweep_1`, …, with an explicit `order` attr on each sweep | HDF5 group iteration is alphabetic (`cut_10` sorts before `cut_2`); the `order` attr survives padding choices and renames |
| H4 | Cut group shape | Fixed-shape state in attrs; vlen fields (`element_ids`, `bounding_polygon`) as datasets | Mirrors `/opensees/sections/{name}/` — attrs + sub-datasets |
| H5 | `None` vs `""` vs empty | Companion flags `has_label`, `has_bounding` (int 0/1) | HDF5 has no `None`; flags make the read deterministic so a missing label round-trips as `None`, not `""` |
| H6 | Schema bump | `2.4.0 → 2.5.0` — additive, major unchanged (2.3.0 = Phase 9 commit 6 / unified recorders; 2.4.0 = Phase 8.7 commit 2 / `/mesh_selections/`) | New groups; existing readers ignore. `EXPECTED_SCHEMA_MAJOR` stays at 2 |
| H7 | Writer primitive | `apeGmsh.cuts._h5_io.write_cuts_into(f, *, cuts, sweeps)` over an open `h5py.File` | One primitive, two ergonomic callers — mirrors `H5Emitter.write_opensees_into` |
| H8 | Primary write path | `ape.h5(path, *, cuts=(), sweeps=())` — model + cuts in one shot | Symmetric with how the rest of the model is written; the user already has both at this point |
| H9 | Append path | `apeGmsh.cuts.persist_to_h5(path, *, cuts=(), sweeps=())` — opens existing `model.h5` in `r+`, calls the primitive, bumps schema | Covers "wrote the model yesterday, building cuts today" without re-running `ape.h5(...)` |
| H10 | Reader | `apeGmsh.cuts._h5_io.read_cuts_and_sweeps(path)` returning `(tuple[SectionCutDef], tuple[SectionSweepDef])` | Reconstructs through the public `__init__` so `__post_init__` validation runs on every read |
| H11 | Missing groups | Reader returns `((), ())` when `/opensees/cuts/` is absent | Pre-v4 files (2.4.0 and earlier) keep working without a special case |
| H12 | Director ingress | `ResultsDirector.load_cuts_from_h5() -> list[Diagram]` | Wraps the reader, dispatches to the existing `add_section_cut*` — no new wiring |
| H13 | Viewer auto-load | `results.viewer(model_h5=p)` with no `cuts=` kwarg triggers `load_cuts_from_h5()` at boot | Persistence is invisible to the user unless they want to override |
| H14 | Kwarg precedence | `cuts=[...]` kwarg overrides the h5 — when both are set, h5 is ignored | Caller's explicit intent is stronger than persisted state |
| H15 | Editability | Read-only in v4; viewer never writes back. Live edit deferred to v4.1 | Keeps v4 to one design pass; the frozen-spec story still holds |
| H16 | Dialog ingress | Add a "Source" mode toggle to `AddDiagramDialog`'s section-cut rows: `.pkl` file (Phase 4) OR pick-from-h5 (v4) | One dialog, two pick modes — no parallel UI |
| H17 | Drift parallel | Out of scope for v4. `/opensees/drifts/` + `/opensees/drift_sweeps/` is v4.1 | One subject per design pass; v5 just landed and bundling them conflates two designs |

### On-disk shape

```
/opensees/cuts/                              standalone cuts
├── attrs: count=N
└── /cut_0/, /cut_1/, ...
    ├── attrs:
    │   plane_point        (3,) f64
    │   plane_normal       (3,) f64        — unit-normalized; reader does not re-normalize
    │   side               utf-8           — "positive" | "negative"
    │   label              utf-8           — "" when has_label=0
    │   has_label          i8              — 0/1 (distinguishes None from "")
    │   has_bounding       i8              — 0/1
    ├── element_ids        (Ne,) i64       — OpenSees element tags
    └── bounding_polygon   (Mb, 3) f64     — present iff has_bounding=1

/opensees/sweeps/                            sweeps, each carrying its own cuts
├── attrs: count=M
└── /sweep_0/, /sweep_1/, ...
    ├── attrs:
    │   count=K
    │   order              vlen utf-8      — ["cut_0", "cut_1", ...] in sweep order
    └── /cuts/
        └── /cut_0/, /cut_1/, ...           — same shape as standalone cut groups
```

The reader walks `order` to reconstruct the sweep's tuple in the right
sequence rather than relying on alphabetic group iteration.

### Public API

```python
from apeGmsh.cuts import (
    SectionCutDef, SectionSweepDef, persist_to_h5,
)
from apeGmsh.cuts._h5_io import read_cuts_and_sweeps
from apeGmsh.opensees import apeSees

# Primary path — model + cuts in one shot
ape = apeSees(fem)
cut = SectionCutDef.from_planar_pg(
    plane_pg="diaphragm-3", elements_pg="tower-cols",
    fem=fem, model_h5="model.h5",
)
ape.h5("model.h5", cuts=[cut])

# Append path — model.h5 already exists
persist_to_h5("model.h5", cuts=[cut])              # appends /opensees/cuts/cut_0
persist_to_h5("model.h5", sweeps=[sweep])          # appends /opensees/sweeps/sweep_0
persist_to_h5("model.h5", cuts=[c1, c2], sweeps=[s1])  # both at once

# Reading
cuts, sweeps = read_cuts_and_sweeps("model.h5")    # ((), ()) on a pre-v4 file

# Viewer auto-load
results.viewer(model_h5="model.h5").show()         # /opensees/cuts/* attach as Layers

# Kwarg-wins override
results.viewer(model_h5="model.h5", cuts=[c_override]).show()  # h5 ignored
```

### Data flow

**Writer (primary path):**

```
SectionCutDef / SectionSweepDef
    │
    │  ape.h5(path, cuts=..., sweeps=...)
    ▼
1. Broker writes /meta + neutral zone
2. H5Emitter writes /opensees/... (materials, sections, ...)
3. write_cuts_into(f, cuts=..., sweeps=...) appends:
     - /opensees/cuts/cut_{i}/ for each standalone cut
     - /opensees/sweeps/sweep_{i}/cuts/cut_{j}/ for sweep members
4. /meta/schema_version set to "2.5.0"
```

**Writer (append path):**

```
SectionCutDef / SectionSweepDef
    │
    │  persist_to_h5(path, cuts=..., sweeps=...)
    ▼
1. Open path in r+ mode
2. Schema-major check (must be 2.x.y)
3. write_cuts_into(f, cuts=..., sweeps=...)
4. Bump /meta/schema_version to "2.5.0" if lower
```

**Reader / viewer auto-load:**

```
ResultsViewer(model_h5=path)
    │
    │  show()
    ▼
1. Director created; set_model_h5(path) called
2. If pending_cuts is empty:
     director.load_cuts_from_h5()
       └─ read_cuts_and_sweeps(path) → (cuts, sweeps)
       └─ for cut in cuts:   self.add_section_cut(cut)
       └─ for sweep in sweeps: self.add_section_cut_sweep(sweep)
3. Else (kwarg-wins): apply pending_cuts; ignore /opensees/cuts/
```

### Dialog integration

The Phase-4 section-cut rows get one new control — a **Source** combo
above the file row that gates which load path is active:

```
┌─────────────────────────────────────────────────────────┐
│ Kind:           [ Section cut          ▼]               │
│ Source:         [ File (.pkl)          ▼]               │  ← NEW
│ File:           [ /path/to/story3.pkl  ] [Browse…]      │  ← visible when Source=File
│   — OR —                                                │
│ Source:         [ In model.h5          ▼]               │  ← NEW (alternate state)
│ Cut:            [ Story 3 (cut_2)      ▼]               │  ← visible when Source=h5
│                                                         │
│ Model.h5:       [ /path/to/model.h5    ] [Browse…]      │
│ Preflight:      [● OK]                                  │
│                ┌────────────────────────────────────┐   │
│                │ PreflightReport — Story 3 ...      │   │
│                └────────────────────────────────────┘   │
│ Display label:  [(optional)            ]                │
│                                       [ OK ] [Cancel]   │
└─────────────────────────────────────────────────────────┘
```

| # | Dialog decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Mode toggle | A "Source" combo with two entries: `"File (.pkl)"` and `"In model.h5"` | One control, two states; cleaner than radio buttons or a parallel tab |
| D2 | h5 dropdown contents | `Cut: [combo]` listing `f"{label or cut_name} ({cut_name})"` for cuts and `f"{label or sweep_name} (sweep, {N} cuts)"` for sweeps | Display label first so users see what they tagged; group name as disambiguator |
| D3 | Repopulation trigger | Re-enumerate the dropdown whenever model.h5 changes (`textChanged`) | Same pattern as the existing preflight rerun |
| D4 | Preflight in h5 mode | Run on every dropdown selection change AND every model.h5 change | Mirrors the file-mode contract — preflight gates OK |
| D5 | Empty-h5 fallback | Source=h5 with no `/opensees/cuts/` → dropdown placeholder "(no cuts persisted in this model.h5)"; OK disabled | Same UX shape as the empty-component-list path |
| D6 | OK dispatch | Same as Phase 4 — branch on `isinstance(loaded, SectionSweepDef)`, call `director.add_section_cut*` | One OK handler; two pick paths feed the same loaded state |

### Package layout additions

```
src/apeGmsh/cuts/
├── _h5_io.py                            ← new: write_cuts_into, read_cuts_and_sweeps, persist_to_h5
└── __init__.py                          ← re-export persist_to_h5 + read_cuts_and_sweeps

src/apeGmsh/opensees/emitter/h5.py
└── SCHEMA_VERSION 2.4.0 → 2.5.0; history note

src/apeGmsh/opensees/architecture/h5-schema.md
└── /opensees/cuts/ and /opensees/sweeps/ sections; 2.5.0 entry

src/apeGmsh/opensees/apesees.py
└── apeSees.h5(path, *, cuts=(), sweeps=())

src/apeGmsh/viewers/diagrams/_director.py
└── ResultsDirector.load_cuts_from_h5() -> list[Diagram]

src/apeGmsh/viewers/results_viewer.py
└── _apply_pending_cuts: auto-load branch when pending_cuts is empty and model_h5 is set

src/apeGmsh/viewers/ui/_add_diagram_dialog.py
└── Source combo + h5-cut dropdown + repopulate-on-h5-change wiring

tests/cuts/test_h5_io.py                            ← new
tests/viewers/test_results_viewer_h5_cuts.py        ← new
tests/viewers/test_add_diagram_section_cut.py       ← extended with h5-source tests
```

### Test plan

The byte-for-byte round-trip is the real check. One test file per layer:

| Test | Validates |
|------|-----------|
| `test_roundtrip_full_shape_cut` | Write SectionCutDef with every field populated, read back, assert dataclass equality |
| `test_roundtrip_minimal_cut` | `label=None`, `bounding_polygon=None` → reads as `None` (not `""` or empty array) |
| `test_roundtrip_sweep_3` | Sweep of 3 cuts; `order` attr drives reader; sequence preserved |
| `test_standalone_and_sweep_coexist` | 2 cuts + 1 sweep with 2 cuts; reader partitions correctly |
| `test_schema_bump_on_append` | `persist_to_h5` against a pre-v4 file → `/meta/schema_version` becomes `"2.5.0"` |
| `test_pre_v4_forward_compat` | Open existing pre-v4 model.h5 (no `/opensees/cuts/`); reader returns `((), ())` |
| `test_append_after_ape_h5` | Full pipeline: `ape.h5(path)` then `persist_to_h5(path, cuts=...)`. File parses through `h5_reader.open` AND `FemToOpsTagMap.from_h5`; cuts read back |
| `test_ape_h5_with_cuts_kwarg` | `ape.h5(path, cuts=...)` produces the same on-disk shape as `ape.h5(path)` + `persist_to_h5(path, cuts=...)` |
| `test_director_load_cuts_from_h5` | After `set_model_h5` + `load_cuts_from_h5()`, registry has one Diagram per cut + one per sweep cut |
| `test_viewer_autoload` | `results.viewer(model_h5=p)` with no `cuts=` kwarg auto-loads |
| `test_viewer_kwarg_wins` | `results.viewer(model_h5=p, cuts=[c_override])` ignores `/opensees/cuts/`; only `c_override` attaches |
| `test_dialog_source_h5_mode` | Source=h5, valid model.h5 with one cut → dropdown lists it, OK enabled |
| `test_dialog_source_h5_empty` | Source=h5, model.h5 has no `/opensees/cuts/` → empty dropdown, OK disabled |

Writer / reader tests run without Qt — `_h5_io` is pure h5py + the
existing `apeGmsh.cuts` types. Viewer tests follow the existing
`patch director.add_section_cut*` pattern rather than driving full
attach.

### Phase 4 roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| v4-0  | This architecture section | **done** |
| v4-1  | `_h5_io.py` writer primitive + reader + round-trip tests | **done** |
| v4-2  | `ape.h5(path, cuts=, sweeps=)` integration + `SCHEMA_VERSION` bump + tests | **done** |
| v4-3  | `persist_to_h5` append helper + in-place schema bump + tests | **done** |
| v4-4  | `ResultsDirector.load_cuts_from_h5` + viewer auto-load + kwarg-wins tests | **done** |
| v4-5  | Dialog Source toggle + h5-cut dropdown + tests | **done** |
| v4-6  | `h5-schema.md` update — `/opensees/cuts/` + `/opensees/sweeps/` body + 2.5.0 entry | **done** |

The `SCHEMA_VERSION` constant bump moved from v4-3 to v4-2 during
implementation: v4-2 is the first producer of v4-shape content
(via `ape.h5(path, cuts=...)`), so it owns the constant change.
v4-3 keeps the in-place version-bump logic for append-mode files
that started life at 2.4.0 or earlier.

### Out of scope for v4

- Drift persistence (`/opensees/drifts/`, `/opensees/drift_sweeps/`) — v4.1
- Live edit of persisted cuts (rewriting `/opensees/cuts/` mid-session) — v4.1
- A "force re-write" GUI button on the dialog — defer until a real call site appears
- Cross-file cut import (read cuts from one model.h5, attach to a different model) — speculative; the FEM-eid bridge is per-file
- Reorganizing `apeGmsh.cuts` → `apeGmsh.outputs` (still bound to the v6-trigger rule)

## Versioning

apeGmsh follows pyproject `version` bumps. New optional subpackage =
minor bump. Schema bump only at v4 of the roadmap (cuts persisted in
`model.h5`, schema 2.4.0 → 2.5.0). v2.3, Phase 4, and v5 are pure
additive surface — no schema impact.
