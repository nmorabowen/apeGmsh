# Phase 8.7 — Viewer migrates off `FEMData` / `solvers`

**Status:** Scoping (May 2026).

Phase 8.7 finishes the Phase 8 untangle on the **consumer side**: the
viewer stops reaching into `apeGmsh.mesh.FEMData` (and into the
already-deleted `apeGmsh.solvers`) and consumes only

- `apeGmsh.results` (for result data — slabs, components, time
  slices), and
- `apeGmsh.opensees.emitter.h5_reader` (for the *model* — the neutral
  + `/opensees/` zones of `model.h5`).

After this phase, the viewer can render any test fixture `model.h5`
file end-to-end without ever instantiating a `FEMData`.

This is the consumer-side bookend to Phase 8.5 (broker writes the
neutral zone) and Phase 8.6 (bridge writes the enrichment zone).
Together, the chain reaches the layering laid out in the Phase 8
master plan ([phase-8-untangle.md §2](phase-8-untangle.md)):

```
Mesh → Broker → model.h5 → Results / Viewer
                  ↑
        Bridge ───┘   (side-feeder)
```

## 1. The change

### 1a. New layer: `apeGmsh.viewers.data.ViewerData`

A read-only structural snapshot consumed by every viewer entry point
(diagrams, scene builder, overlays, UI tabs). One adapter,
two builders, no leakage back to `mesh/` or `opensees/<internals>`.

```python
from apeGmsh.viewers.data import ViewerData

view = ViewerData.from_fem(fem)               # live g.mesh path
view = ViewerData.from_h5("model.h5")         # post-solve / fixture path
view = ViewerData.from_h5_model(h5model)      # already-open H5Model
```

ViewerData exposes only the surface the current viewer code actually
exercises (audited in §2 below). It is **not** a 1:1 mirror of
`FEMData` — it is a viewer-facing slice. Anything outside that slice
stays on `FEMData` and is not migrated.

### 1b. Schema bump 2.3.0 → 2.4.0 (additive): `/mesh_selections/`

Post-mesh selection sets (`g.mesh_selection` → `fem.mesh_selection`)
become part of the neutral zone. Selection sets are user-authored
groupings captured at `get_fem_data()` time; without them the viewer's
`selection=` selector silently returns empty IDs after a round-trip
through `model.h5`.

The on-disk layout mirrors `/physical_groups/` exactly so the same
`H5Reader._read_named_index` helper handles both:

```
/mesh_selections/
└── {name}/                       (one group per selection)
    ├── attrs: dim (int), tag (int), name (utf-8)
    ├── node_ids       int64 (N,)
    ├── node_coords    float64 (N, 3)
    └── element_ids    int64 (E,)        # only for dim >= 1
```

Schema 2.4.0 is **additive**: pre-8.7 viewers ignore the new group
and lose only the `selection=` round-trip convenience (post-mesh
selectors still work in live mesh_viewer sessions, where the live
`fem.mesh_selection` is consulted directly).

### 1c. ADR 0014 — viewer is a pure model.h5 consumer

Lands in this commit. See
[decisions/0014-viewer-is-pure-h5-consumer.md](decisions/0014-viewer-is-pure-h5-consumer.md).
The architectural decision: the viewer package depends on
`apeGmsh.results` and `apeGmsh.opensees.emitter.h5_reader` only.
No `from apeGmsh.mesh ...` and no `from apeGmsh.opensees.<not
emitter.h5_reader> ...` survive in `src/apeGmsh/viewers/`.

## 2. The audit — what the viewer reads from FEMData today

Compiled from `grep` over `src/apeGmsh/viewers/`. Each row is an
accessor the migration must preserve on `ViewerData`. Live-Gmsh
viewers (`model_viewer.py`, the `mesh_viewer.py` pre-overlay path)
continue to consume their own gmsh / core surface — they are not
Phase 8.7's target.

### 2a. Structural substrate (load-bearing for `fem_scene.py`)

| FEMData accessor | Used by |
|---|---|
| `fem.nodes.ids` (1-D int64) | `fem_scene`, `_fiber_section`, `_line_force`, `_section_cut`, `_spring_force` |
| `fem.nodes.coords` (N×3 float64) | (same) + `constraint_overlay` |
| `fem.nodes.index(node_id) → row` | `mesh_viewer`, `constraint_overlay` |
| `fem.elements` iter of `ElementGroup` with `.element_type.{code,dim,npe}`, `.connectivity`, `.ids`, `len` | `fem_scene`, `_gauss_marker`, `_fiber_section`, `_layer_stack`, `_line_force`, `_section_cut`, `_spring_force` |
| `fem.elements.types` | result composites (geometric helpers); not directly used by the viewer |
| `fem.elements.resolve(element_type=name) → (ids, conn)` | result composites (centroid helpers) |

### 2b. Group / label / selection membership

| FEMData accessor | Used by |
|---|---|
| `fem.nodes.physical.node_ids(name)` | `_selectors`, result composites |
| `fem.nodes.labels.node_ids(name)` | `_selectors`, result composites |
| `fem.elements.physical.element_ids(name)` | `_selectors`, result composites |
| `fem.elements.labels.element_ids(name)` | `_selectors`, result composites |
| `fem.mesh_selection.node_ids(name)` / `.element_ids(name)` | `_selectors` (the live-session selection lookup) |

### 2c. Record sets (live FEMData only; not round-tripped today)

`FEMData.from_native_h5` and `from_mpco_model` explicitly do NOT
restore loads / masses / constraints (per
`FEMData.from_native_h5` docstring). So today, results-side overlays
that depend on these are silent no-ops. After Phase 8.7, model.h5
becomes their source — and the overlays light up whenever a
`model.h5` is available alongside the run.

| FEMData accessor | Used by |
|---|---|
| `fem.nodes.loads` iter | `_kind_catalog`, `mesh_viewer` |
| `fem.nodes.loads.patterns()` | `_kind_catalog`, `loads_tab` |
| `fem.nodes.loads.by_pattern(name) → iter` | `_loads`, `mesh_viewer` |
| `fem.nodes.masses` iter | `mesh_viewer`, `mass_tab` |
| `fem.nodes.constraints` iter | `mesh_viewer`, `constraints_tab` |
| `fem.nodes.constraints.pairs() → iter` | `constraint_overlay` |
| `fem.nodes.constraints.phantom_nodes()` | `mesh_viewer` |
| `fem.elements.constraints.interpolations() → iter` | `constraint_overlay` |
| `fem.elements.constraints.couplings() → iter` | `constraint_overlay` |

### 2d. Metadata

| FEMData accessor | Used by |
|---|---|
| `fem.snapshot_id` | `_session` (diagram session-state persistence — informational) |

## 3. ViewerData surface

A read-only struct. No mutation — every accessor is a query.

```python
@final
class ViewerData:
    """Read-only structural snapshot consumed by the viewer package.

    Built from a live FEMData (``from_fem``) or from a ``model.h5``
    file (``from_h5`` / ``from_h5_model``). The two builders produce
    interchangeable instances; downstream code does not need to know
    which source the snapshot came from.
    """

    # ── Source-of-truth metadata ───────────────────────────────────
    @property
    def snapshot_id(self) -> str: ...
    @property
    def source_kind(self) -> Literal["fem", "h5"]: ...

    # ── Nodes ──────────────────────────────────────────────────────
    nodes: ViewerNodes      # .ids, .coords, .index(nid) → row
                            # .physical.node_ids(name), .labels.node_ids(name)
                            # .selection.node_ids(name)  (returns [] when missing)
                            # .loads (RecordSet), .masses (RecordSet),
                            # .constraints (ConstraintRecordSet)

    # ── Elements ───────────────────────────────────────────────────
    elements: ViewerElements    # iter of ElementGroup view, .types
                                # .physical.element_ids(name), .labels.element_ids(name)
                                # .selection.element_ids(name)
                                # .constraints (ConstraintRecordSet)
```

The `ElementGroup` view exposes `element_type` (with `code`, `dim`,
`npe`), `connectivity`, `ids`, and `__len__`. From `from_h5`, the
type info comes from each `/elements/{gmsh_alias}` group's attrs
(`code`, `dim`, `npe` — written by Phase 8.5).

The `RecordSet` shapes are decoded read-side row dataclasses defined
in `viewers/data/_records.py` (see §3a). They are NOT
`apeGmsh.mesh.records._constraints.NodePairRecord` etc. — those are
the write-side authoritative types and importing them from the
viewer would re-establish the forbidden coupling.

### 3a. Record decode

`viewers/data/_records.py` defines lightweight read-side row tuples,
one per `payload_kind`:

```python
@dataclass(frozen=True)
class NodalLoadRow:    # payload_kind="point_load"
    node_id: int
    pattern: str
    values: tuple[float, ...]

@dataclass(frozen=True)
class NodePairRow:     # payload_kind="rigid_beam" | "node_pair" | ...
    payload_kind: str
    master_id: int
    slave_id: int
    dofs: tuple[int, ...]
    # subtype-specific extras (kept as a dict for forward-compat)

# ...one per Kind in mesh.records._kinds (RigidBeam, NodePair,
# NodeGroup, NodeToSurface, Interpolation, SurfaceCoupling, ...)
```

`from_fem` reads the authoritative records via existing FEMData
accessors and copies into these tuples. `from_h5` reads the compound
arrays via `H5Reader.constraints()` / `loads()` / `masses()` and
dispatches on `payload_kind` to populate the same tuples.

**Why duplicate the schema in viewers/data/** — the `payload_kind`
dispatch reads the compound row's typed fields by name; the field
names are part of the symmetric compound contract documented in
`mesh/_record_h5.py` and `architecture/h5-schema.md`. Duplicating
the names (not the logic) in `viewers/data/_records.py` decouples
the viewer from the write-side dataclass library while keeping the
contract enforced by the schema document.

If a new `payload_kind` is added to the writer side and the viewer
hasn't been updated, `from_h5` raises a `ViewerDataDecodeError`
naming the unknown kind. (One-shot — does not corrupt the rest of
the read.)

### 3b. Selection sets — the 2.4.0 round-trip

`view.nodes.selection.node_ids(name)` returns:

- From `from_fem`: proxy to `fem.mesh_selection.node_ids(name)`.
- From `from_h5`: read `/mesh_selections/{name}/node_ids` (the new
  2.4.0 group). Returns `np.array([], dtype=int64)` when the name is
  unknown — same shape as the live API's "no match" branch.

## 4. Source-side surface

### 4a. New files

| File | Purpose |
|---|---|
| `src/apeGmsh/viewers/data/__init__.py` | Re-exports `ViewerData`. |
| `src/apeGmsh/viewers/data/_viewer_data.py` | The adapter class + builders. |
| `src/apeGmsh/viewers/data/_records.py` | Read-side row dataclasses + compound-row decoder. |
| `src/apeGmsh/viewers/data/_nodes.py` | `ViewerNodes` + sub-composites (physical, labels, selection, loads, masses, constraints). |
| `src/apeGmsh/viewers/data/_elements.py` | `ViewerElements` + `ElementGroup` view. |
| `src/apeGmsh/opensees/architecture/decisions/0014-viewer-is-pure-h5-consumer.md` | ADR (lands in commit 1). |
| `tests/viewers/data/test_viewer_data.py` | Parity test: `ViewerData.from_fem(fem)` and `ViewerData.from_h5(path)` produce equivalent accessor results for the fixtures. |
| `tests/test_viewers_pure_h5_consumer.py` | AST acceptance test: walks `viewers/*.py` and rejects `from apeGmsh.mesh ...` and `from apeGmsh.opensees.<not emitter.h5_reader> ...`. |

### 4b. Files modified

| File | Change |
|---|---|
| `src/apeGmsh/mesh/_femdata_h5_io.py` | (Commit 2) Write `/mesh_selections/` when `fem.mesh_selection` is present. |
| `src/apeGmsh/opensees/emitter/h5_reader.py` | (Commit 2) Add `H5Reader.mesh_selections()` accessor. |
| `src/apeGmsh/opensees/architecture/h5-schema.md` | (Commit 2) Document the new `/mesh_selections/` group + bump 2.3.0 → 2.4.0. |
| `src/apeGmsh/viewers/scene/fem_scene.py` | (Commit 4) `build_fem_scene(fem)` → `build_fem_scene(view: ViewerData)`. |
| `src/apeGmsh/viewers/mesh_viewer.py`, `results_viewer.py` | (Commit 4) Drop direct FEMData imports; build a `ViewerData` and pass it down. The `mesh_viewer:929 from apeGmsh.mesh._record_set import ConstraintKind` line goes away — `ConstraintKind` becomes a string token on `ConstraintRecordRow`. |
| `src/apeGmsh/viewers/diagrams/*.py` (~15 files) | (Commit 5) `def attach(self, plotter, fem, scene)` → `def attach(self, plotter, view, scene)` across the diagram hierarchy. |
| `src/apeGmsh/viewers/ui/{constraints_tab,loads_tab,mass_tab}.py` | (Commit 6) Switch from `FEMData` to `ViewerData`; drop the `from apeGmsh.core.{loads,masses,constraints}.defs` imports (the tabs only used the resolved records, not the pre-mesh defs). |
| `src/apeGmsh/viewers/overlays/constraint_overlay.py` | (Commit 6) Switch from `mesh._record_set.ConstraintKind` + `mesh.records._constraints.*` to `viewers/data/_records.py` row types. |
| `src/apeGmsh/opensees/architecture/viewer-integration.md` | (Commit 7) Replace `FEMData` references with `ViewerData`; document the discovery / construction contract. |
| `src/apeGmsh/opensees/architecture/phase-8-untangle.md` | (Commit 7) Mark §5 8.7 as landed; cross-ref this scope doc. |
| `src/apeGmsh/opensees/architecture/parallel-execution.md` | (Commit 7) Mark Phase 8 row fully landed. |

### 4c. Files NOT touched

- `viewers/model_viewer.py` — the live pre-mesh g.viewer. Imports
  `core.Model` + `core.Labels`. The acceptance criterion forbids
  `mesh.*` and `opensees.*` (not `core.*`); model_viewer is the live
  twin of mesh_viewer / results_viewer, not a fixture consumer.
- `viewers/mesh_viewer.py` substrate path — still reads from a live
  gmsh session via `MeshSceneData` (separate from `FEMSceneData`).
  Phase 8.7 only migrates the overlay-data side (loads, masses,
  constraints) so that `from apeGmsh.mesh ...` imports vanish.
- `apeGmsh.mesh.records.*` — the write-side authoritative records
  stay where Phase 8.1 (PR #119) landed them.
- Results' FEMData synthesis pipeline (`results/readers/_mpco*.py`,
  `Results.from_native`, `Results.bind`) — Results legitimately
  needs a FEMData for its own result composites' geometric helpers
  (`nearest_to`, `in_box`, …). Phase 8.7 cuts the viewer's
  dependency on it, not Results'.

## 5. Commit decomposition

Seven commits, one PR each. Branch fresh from `origin/main` for each
commit; wait for the prior PR to merge before starting the next.

| # | Title | Concern | Risk |
|---|---|---|---|
| 1 | `viewers: phase-8.7 commit 1 — scope doc + ADR 0014` | This doc + ADR. No code. | none |
| 2 | `mesh: phase-8.7 commit 2 — mesh_selection round-trip in model.h5 (schema 2.4.0)` | `/mesh_selections/` writer in `_femdata_h5_io.py`, `H5Reader.mesh_selections()`, h5-schema.md, round-trip test. | low (additive) |
| 3 | `viewers: phase-8.7 commit 3 — ViewerData adapter` | Implements `viewers/data/{_viewer_data,_nodes,_elements,_records}.py`. Unit tests against fixtures. | low (new code only; nothing depends on it yet) |
| 4 | `viewers: phase-8.7 commit 4 — fem_scene migrates to ViewerData` | `fem_scene.py` + entry points in `mesh_viewer.py` / `results_viewer.py`. | medium (load-bearing UI substrate path) |
| 5 | `viewers: phase-8.7 commit 5 — diagrams attach(view, scene)` | Batch rename across ~15 diagrams. | medium (broad but mechanical) |
| 6 | `viewers: phase-8.7 commit 6 — UI tabs + overlays` | `constraints_tab`, `loads_tab`, `mass_tab`, `constraint_overlay`. | medium |
| 7 | `viewers: phase-8.7 commit 7 — acceptance test + docs sweep` | AST test, viewer-integration.md, phase-8-untangle.md, parallel-execution.md. | low |

## 6. Acceptance criteria

For Phase 8.7 as a whole:

- [ ] `git grep "from apeGmsh.mesh" src/apeGmsh/viewers/` returns zero
      matches.
- [ ] `git grep "from apeGmsh.opensees" src/apeGmsh/viewers/` returns
      matches only for `from apeGmsh.opensees.emitter.h5_reader`.
- [ ] `viewers/data/_viewer_data.py::ViewerData.from_h5("frame_3d.h5")`
      produces a renderable viewer without any FEMData synthesis.
- [ ] All existing viewer tests pass against `ViewerData.from_fem(fem)`
      inputs.
- [ ] `tests/test_viewers_pure_h5_consumer.py` enforces the import
      ban via AST walk (run on every commit).
- [ ] `architecture/viewer-integration.md` documents `ViewerData` as
      the contract surface, with no remaining FEMData references in
      the v2.4.0+ flow.

## 7. Open questions

1. **Diagram param name**: `fem` → `view`, `vd`, or `model`? `view`
   collides with VTK "view" terminology; `model` collides with
   `model.h5` / `Model` class; `vd` is opaque. **Tentative: `view`**
   (the diagrams are downstream of `ViewerData`; the type annotation
   disambiguates). Resolved in commit 5.

2. **`ConstraintKind` import in mesh_viewer.py:929**: today the
   viewer reaches into `apeGmsh.mesh._record_set.ConstraintKind` to
   branch on constraint type. After 8.7, the row dataclasses in
   `viewers/data/_records.py` carry `payload_kind` as a plain
   string, and the branch becomes a string comparison. No public
   `ConstraintKind` enum re-export from viewers/ — keep the seam
   one-directional.

3. **`Results.from_h5_only(path)` shortcut**: not landed in Phase
   8.7. The viewer can already open a `model.h5` directly via
   `ViewerData.from_h5(path)`; Results is needed only when there are
   *result data* to plot, in which case the existing `from_native` /
   `from_mpco` paths synthesize a FEMData internally and the viewer
   constructs `ViewerData.from_fem(results.fem)` over it. A future
   refactor could let Results delegate its own structural queries to
   `ViewerData` instead of synthesizing a FEMData, but that's a
   Results-internal change and out of scope here.

## 8. Out of scope

- Editing `model.h5` from the viewer. The contract remains read-only.
- Streaming updates (live link to a running bridge). The H5 is a
  snapshot.
- Result enrichment overlays driven by `model.h5` alone (no MPCO /
  run.h5). That's a future direction; Phase 8.7 only retargets the
  *model* side of the viewer.
- A second solver's enrichment zone in `model.h5`. The namespace is
  in place; the actual plug-in is a future project.

## References

- [phase-8-untangle.md](phase-8-untangle.md) — the master plan;
  §5 (sub-phase 8.7), §6 (acceptance), §7 (open question 4 already
  resolved).
- [phase-8.5-scope.md](phase-8.5-scope.md) — the broker-side neutral
  zone writers this phase consumes.
- [phase-8.6-scope.md](phase-8.6-scope.md) — the bridge-side
  enrichment writers this phase consumes.
- [viewer-integration.md](viewer-integration.md) — the existing
  viewer contract; rewritten in commit 7.
- [h5-schema.md](h5-schema.md) — schema 2.3.0 today; bumps to 2.4.0
  in commit 2.
- [decisions/0014-viewer-is-pure-h5-consumer.md](decisions/0014-viewer-is-pure-h5-consumer.md)
  — the architectural decision (lands in this commit).
- [decisions/0013-records-in-mesh-not-solvers.md](decisions/0013-records-in-mesh-not-solvers.md)
  — Phase 8.1's symmetric decision on the producer side.
