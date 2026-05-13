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
| v2.2 | Viewer overlay — `SectionCutDiagram` Layer kind + filter highlight | **in progress** |

v4 and beyond (`model.h5` persistence of cuts, live editing, drift
specs, sweep templates) are described in the session that drafted this
plan — out of scope for this directory until v2.2 is complete.

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

## Versioning

apeGmsh follows pyproject `version` bumps. New optional subpackage =
minor bump. Schema bump only at Phase 4 if we decide to persist cuts
in `model.h5` (currently planned for v4 of the roadmap, not v1).
