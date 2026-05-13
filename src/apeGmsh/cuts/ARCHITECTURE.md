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

v4 and beyond (`model.h5` persistence of cuts, live editing, drift
specs, sweep templates) are described in the session that drafted this
plan — out of scope for this directory until v2.3 is complete.

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

## Versioning

apeGmsh follows pyproject `version` bumps. New optional subpackage =
minor bump. Schema bump only at v4 of the roadmap (cuts persisted in
`model.h5`, schema 2.2.0 → 2.3.0). v2.3 and Phase 4 are pure additive
surface — no schema impact.
