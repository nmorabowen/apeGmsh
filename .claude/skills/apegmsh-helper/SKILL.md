---
name: apegmsh-helper
description: Use whenever the user is working with apeGmsh — the structural-FEM wrapper around Gmsh with OpenSees integration. Triggers on building FEM models from CAD/STEP imports, Part-based assembly workflows, composite-based geometry/mesh/constraint APIs (g.model, g.mesh, g.physical, g.constraints, etc.), the apeSees(fem) OpenSees bridge with typed primitives, loads/masses/constraints resolution into the FEMData broker, post-processing OpenSees output via Results (from_native/from_mpco/from_recorders) and the ResultsViewer, native model.h5 persistence (FEMData.to_h5/from_h5, save_to=/g.save()), and exporting models to OpenSees Tcl or openseespy scripts. Covers apeGmsh's own abstractions on top of Gmsh and OpenSees. For raw gmsh API questions see the gmsh-structural skill; for raw OpenSees analysis commands see opensees-expert; for FEM theory first principles see fem-mechanics-expert.
---

# apeGmsh helper

apeGmsh is a structural-FEM wrapper around [Gmsh](https://gmsh.info) with a
composition-based API and an OpenSees integration.  The library describes a
structural model **once** — geometry, physical groups, loads, masses,
constraints — and feeds it to any solver through a snapshot broker (`FEMData`).

This skill teaches **apeGmsh's vocabulary and idioms**.  It does not re-teach
Gmsh or OpenSees — for those, see the cross-reference section at the bottom.

> Every code snippet below carries an inline `# verified: tests/<file>::<test>`
> citation. Those tests pass green under the pinned harness (full suite
> **5864 passed, 64 skipped** against the worktree `src/`).

---

## 1. Mental model

### 1.1  The session owns composites

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="my_model", verbose=True) as g:   # verified: tests/test_session_save.py::test_autosave_writes_on_exit
    ...
```

`g` is an **apeGmsh session** — one `gmsh.initialize()`, one model, and a
set of composite objects as attributes. Each composite owns a focused slice
of the API. You never call `gmsh.*` directly. OpenSees is **not** a session
composite — it is a separate post-session bridge, `apeSees(fem)` (§5).

| Access | Purpose |
|---|---|
| `g.model` | OCC geometry — points, curves, surfaces, solids, booleans, transforms, I/O |
| `g.parts` | Part registry — import, fragment, fuse, node/face maps |
| `g.physical` | Named physical groups (solver-facing, Tier 2) |
| `g.labels` | Internal labels (Tier 1) — survive boolean ops |
| `g.sections` | Parametric section builders (W_solid, rect_solid, W_shell) |
| `g.mesh` | Meshing — generation, sizing, fields, structured, editing, queries, partitioning |
| `g.mesh_selection` | Post-mesh node/element selection sets |
| `g.constraints` | Pre-mesh constraint definitions (14 declaration verbs — §3.1) |
| `g.loads` | Pre-mesh load definitions (pattern grouping) |
| `g.masses` | Pre-mesh mass definitions |
| `g.node_ndf` | Explicit per-node `ndf` (DOF count) declarations — required for mixed-ndf models (shell-on-solid). Fail-loud `fem.nodes.ndf_for(nid)` (ADR 0032). Override channel: `apeSees(fem)` emit passes `-ndf K` per-node where override exists; envelope (`apeSees(fem).model(ndm, ndf=K)`) wins on sentinel (ADR 0033) |
| `g.loader` | `.msh` file loader — build a `FEMData` snapshot from a saved mesh |
| `g.inspect` | Session diagnostics |
| `g.plot` | Matplotlib (optional dep) |
| `g.view` | Gmsh post-processing views |

The FEMData broker is organized around what the engineer needs — **nodes and
elements**, not raw Gmsh tags. Address mesh subsets by physical-group / label
name; query constraints with `Kind` constants, never magic strings (§1.3).
Key broker files: `mesh/FEMData.py`, `mesh/_group_set.py`, `mesh/_record_set.py`,
`mesh/_fem_factory.py`, `mesh/_fem_extract.py`.

### 1.2  Sub-composites are required — no shortcuts

The two biggest session composites have focused sub-namespaces:

**`g.model.*`** — geometry sub-composites:

| Sub-composite | Examples |
|---|---|
| `g.model.geometry` | `add_point`, `add_box`, `add_cylinder`, `add_cutting_plane`, `slice` |
| `g.model.boolean` | `fuse`, `cut`, `intersect`, `fragment` |
| `g.model.transforms` | `translate`, `rotate`, `scale`, `mirror`, `copy`, `extrude`, `revolve` |
| `g.model.io` | `load_step`, `save_step`, `heal_shapes` |
| `g.model.queries` | `bounding_box`, `center_of_mass`, `boundary`, `registry` |

**`g.mesh.*`** — meshing sub-composites:

| Sub-composite | Examples |
|---|---|
| `g.mesh.generation` | `generate`, `set_order`, `refine`, `optimize`, `set_algorithm` |
| `g.mesh.sizing` | `set_global_size`, `set_size`, `set_size_by_physical` |
| `g.mesh.field` | `distance`, `threshold`, `box`, `boundary_layer`, `minimum` |
| `g.mesh.structured` | `set_transfinite_curve/surface/volume`, `recombine` |
| `g.mesh.editing` | `embed`, `set_periodic`, `remove_duplicate_nodes`, `crack` |
| `g.mesh.queries` | `get_nodes`, `get_elements`, `get_fem_data`, `quality_report` |
| `g.mesh.partitioning` | `renumber`, `partition`, `unpartition`, `summary` |

OpenSees is **not** a `g.` sub-composite. It is the separate
post-session bridge `apeSees(fem)` (§5) with typed-primitive
namespaces (constructors return handles; no string types):

| Namespace | Examples |
|---|---|
| `ops.uniaxialMaterial` / `ops.nDMaterial` / `ops.section` | `Steel02(...)`, `ElasticIsotropic(...)`, `Fiber(...)` |
| `ops.geomTransf` / `ops.beamIntegration` | `Linear(...)`, `PDelta(...)`, `Lobatto(...)` |
| `ops.element` | `forceBeamColumn(pg=...)`, `FourNodeTetrahedron(pg=..., body_force=...)` |
| `ops.timeSeries` / `ops.pattern` / `ops.recorder` | `Linear()`, `Plain(series=...)`, `Node(...)` |
| flat verbs | `ops.model(...)`, `ops.fix(pg=, dofs=)`, `ops.mass(pg=, values=)`, `ops.build()`, `ops.tcl/py/h5(path)`, `ops.run()`, `ops.analyze(...)` |

**Critical:** the bridge does **not** ingest session-declared
`g.loads` / `g.masses` / `g.constraints`. Loads, masses and SPs
must be **re-declared explicitly** on `ops` (§4/§5). Multi-point
constraints (tie / rigid_link / equal_dof / node_to_surface) have
**no emission path at all** — deferred (§3.3).

### 1.3  FEMData is the broker — composite-based architecture

```python
fem = g.mesh.queries.get_fem_data(dim=3)   # verified: tests/test_fem_chain.py::test_select_hooks_return_point_family_meshselection
# or
fem = FEMData.from_gmsh(dim=3, session=g)
```

`fem` is a **FEMData snapshot** organized by what the engineer needs:

```
fem
  |-- .nodes              NodeComposite
  |     |-- .ids          ndarray(N,) — all node IDs (object dtype, yields Python int)
  |     |-- .coords       ndarray(N, 3) — all coordinates (float64)
  |     |-- .index(nid)   → O(1) array index lookup
  |     |-- .select(...)  → MeshSelection (point family — the ONLY accessor; §1.4)
  |     |-- .physical     PhysicalGroupSet (solver-facing PGs)
  |     |-- .labels       LabelSet (geometry-time labels)
  |     |-- .constraints  NodeConstraintSet — mixes 3 record types:
  |     |                    NodePairRecord (equal_dof, rigid_beam/rod, penalty)
  |     |                    NodeGroupRecord (rigid_diaphragm, rigid_body,
  |     |                                     kinematic_coupling)
  |     |                    NodeToSurfaceRecord (node_to_surface + phantom nodes)
  |     |                  Iterators:
  |     |                    .pairs()             → flat NodePairRecord expansion
  |     |                                           (compound records auto-expanded)
  |     |                    .rigid_link_groups() → (master, [slaves])
  |     |                    .rigid_diaphragms()  → (master, [slaves])
  |     |                    .equal_dofs()        → NodePairRecord
  |     |                    .node_to_surfaces()  → raw NodeToSurfaceRecord
  |     |-- .loads        NodalLoadSet (point forces)
  |     |-- .sp           SPSet (single-point constraints — from g.loads.face_sp)
  |     +-- .masses       MassSet (lumped nodal masses)
  |
  |-- .elements           ElementComposite
  |     |-- .ids          ndarray(E,) — all element IDs concatenated
  |     |-- .connectivity ndarray(E, npe) — ONLY if mesh is homogeneous
  |     |                                  (raises TypeError on mixed types)
  |     |-- .select(...)  → MeshSelection (point family — the ONLY accessor; §1.4)
  |     |-- .constraints  SurfaceConstraintSet (tie, mortar, tied_contact)
  |     +-- .loads        ElementLoadSet (pressure, body force)
  |
  |-- .info               MeshInfo (n_nodes, n_elems, bandwidth, elem_type_name)
  +-- .inspect            InspectComposite (summary, tables, source tracing)
```

**Selection API** — get subsets by physical group or label. The
canonical fluent `.select()` chain (§1.4) is the **only** accessor:
`fem.nodes.select(...)` / `fem.elements.select(...)` return a
`MeshSelection`. A node selection's terminals are `.ids` / `.coords` /
`.result()` (→ `NodeResult`); an element selection's are `.ids` /
`.connectivity` / `.groups()` / `.result()` (→ `GroupResult`).
`.resolve()` is **not** a `MeshSelection` terminal — it is a method of
the **`GroupResult`** you get from an element selection's `.result()`
(flattens single-type selections to `(ids, connectivity)`). The old
single-shot `fem.nodes.get/get_ids/get_coords`, `fem.elements.get/resolve`
were **removed** by selection-unification v2 — there is no shim.

```python
# .result() on a node selection -> NodeResult (yields (id, xyz) pairs;
# ids are Python ints)
for nid, xyz in fem.nodes.select(pg="Base").result():   # verified: tests/test_fem_chain.py::test_node_select_returns_nodechain_seeded_by_resolver
    ops.node(nid, *xyz)

# Or read the bulk arrays directly off the MeshSelection
sel = fem.nodes.select(pg="Base")                        # verified: tests/test_selection_dtype_contract.py::test_select_by_name_yields_python_int_tags
sel.ids                 # list[int]
sel.coords              # ndarray (N, 3) float64
sel.result().to_dataframe()   # pandas DataFrame (via NodeResult)
```

**Kind constants** — no magic strings (typed as `ClassVar[str]`):
```python
K = fem.nodes.constraints.Kind          # ConstraintKind   # verified: tests/test_constraint_resolver.py::test_matches_colocated_pairs
K.RIGID_BEAM, K.EQUAL_DOF, K.TIE, ...   # 15 values
K.NODE_PAIR_KINDS, K.SURFACE_KINDS      # frozensets for routing
```
The 15 values are EQUAL_DOF, RIGID_BEAM, RIGID_BEAM_STIFF, RIGID_ROD,
RIGID_DIAPHRAGM, RIGID_BODY, KINEMATIC_COUPLING, PENALTY,
NODE_TO_SURFACE, NODE_TO_SURFACE_SPRING, TIE, DISTRIBUTING, EMBEDDED,
TIED_CONTACT, MORTAR. Record dataclasses live under
`apeGmsh._kernel.records.*` (`_constraints` / `_loads` / `_masses` /
`_kinds`); `fem.nodes.constraints.Kind` re-exposes `ConstraintKind`.

**Grouped iteration** — the FEMData constraint-iteration API.
Note: `apeSees(fem)` does **not** emit these (MP constraints are
deferred — §3.3); iterate them yourself only for inspection or to
hand-drive a solver that supports multi-point constraints (the
`ops.*` below is raw openseespy you write yourself):
```python
# Rigid links — accumulated by master across rigid_beam/rod,
# rigid_diaphragm, rigid_body, kinematic_coupling, and
# node_to_surface phantom links
for master, slaves in fem.nodes.constraints.rigid_link_groups():   # verified: tests/test_femdata_from_h5.py::test_round_trip_node_to_surface_record
    for slave in slaves:
        ops.rigidLink("beam", master, slave)

# Equal DOFs — flat (direct pairs + expanded node_to_surface)
for pair in fem.nodes.constraints.equal_dofs():                    # verified: tests/test_femdata_from_h5.py::test_round_trip_node_to_surface_record
    ops.equalDOF(pair.master_node, pair.slave_node, *pair.dofs)
```

**Introspection** — what you have and why:
```python
print(fem.inspect.summary())              # verified: tests/test_femdata_to_h5.py::test_to_h5_writes_meta
```
`fem.inspect` (InspectComposite) also surfaces constraint counts +
source names and node / physical tables; lean on `fem.inspect.summary()`
first when a model "looks wrong".

### 1.4  `.select()` — the canonical fluent selection idiom

`.select()` is the **one canonical, daisy-chainable** selection idiom.
It exists at **all four levels** with the *same* verb names and set
algebra:

| Entry point | Returns | Family | Terminal |
|---|---|---|---|
| `g.model.select(target, *, dim=)` | `EntitySelection` | entity | `.to_label`/`.to_physical`/`.to_dataframe`; `.result()` → `Selection` payload |
| `fem.nodes.select(...)` | `MeshSelection` | point | `.ids`/`.coords`/`.result()` → `NodeResult` |
| `fem.elements.select(...)` | `MeshSelection` | point | `.ids`/`.groups()`/`.result()` → `GroupResult` (`.resolve()` lives here) |
| `results.nodes.select(...)` / `results.elements.select(...)` | `MeshSelection` | point | `.values(component=, time=, stage=)` → slab |
| `g.mesh_selection.select(*, level=, dim=, ids=, name=)` | `MeshSelection` | point | `.ids`/`.coords`/`.result()`; `.save_as(name)` (live engine) |

**Refining verbs** (identical on every chain), each returns a new
chain of the same type so they compose:

```python
sel = (fem.nodes.select(pg="Body")          # seed by pg/label/tag/dim/ids   # verified: tests/test_fem_chain.py::test_node_select_returns_nodechain_seeded_by_resolver
           .in_box((0, 0, 0), (1, 1, 1))     # half-open [lo, hi)
           .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
           .nearest_to((0.5, 0.5, 0.0), count=4)
           .where(lambda xyz: xyz[2] == 0.0))
nodes = sel.result()                          # -> NodeResult
```

- `.in_box(lo, hi, *, inclusive=False)` · `.in_sphere(center, radius)`
  · `.on_plane(point, normal, *, tol)` · `.nearest_to(point, *,
  count=1)` · `.where(predicate)`. **`tol=` on `.on_plane` is a
  REQUIRED kwarg** (no default — always pass it).
- **Set algebra**: `|` `&` `-` `^` and the aliases `.union` /
  `.intersect` / `.difference` (insertion-order preserving). Combining
  chains of different type or different engine raises `TypeError`
  (cross-level mixing is loud, never a silent empty set):
  ```python
  fem.nodes.select(ids=a) | fem.nodes.select(ids=b)   # verified: tests/test_fem_chain.py::test_select_hooks_return_point_family_meshselection
  ```
- **Seeding reuses the locked resolvers** — broker chains take
  `target`/`pg`/`label`/`tag`/`dim`/`partition` (`element_type=` for
  elements) plus `ids=`; no-arg seeds the whole domain. The seed
  delegates verbatim to the same contract-locked resolver
  (`_resolve_nodes` / `_resolve_elem_ids`) the removed `.get` used —
  the resolved selection is exactly what the locked resolution
  contract returns. Results chains take `pg=`/`label=`/`selection=`/
  `ids=`. `g.mesh_selection.select` takes `ids=` **or** `name=` (an
  **existing** `g.mesh_selection` set, seeded id-for-id — delegates
  verbatim to the existing `get_tag`/`get_nodes`/`get_elements`,
  no new resolver, read-only, fail-loud on an unknown name); no-arg
  seeds the full live-mesh universe. `g.model.select` resolves
  strings through the same label→PG→part tier as everything else.
  > Tripwire: the unknown-name error message intentionally names
  > **both** `from_physical` *and* `from_geometric` as the route hint.
  > `tests/test_mesh_selection_chain_name_seed.py::test_unknown_name_fails_loud_with_route_hint`
  > (line 227) asserts both substrings — that test pins
  > `src/apeGmsh/mesh/MeshSelectionSet.py` **byte-untouched** (its
  > stale `from_geometric` strings + a dead `:class:` xref to the
  > deleted `_mesh_selection_chain.py` are accepted P4 residue; do
  > **not** "fix" them — touching them reddens the suite).
- **Results terminal differs** — a results selection needs a
  component, so it ends in `.values(...)` (which forwards to the
  **retained** typed reader `results.<sub>.get(component=, ids=, pg=,
  label=, selection=)`), not `.result()`:
  ```python
  (results.nodes.select(pg="Base")          # verified: tests/test_results_selection_backfill.py::test_select_then_values_matches_get
       .in_box(lo, hi)
       .values(component="displacement_x"))
  ```

**Two spatial families — same names, NOT the same behavior:**

- **point family** (`fem.*`, `results.*`, `g.mesh_selection`) —
  `.in_box` is half-open `[lo, hi)` by default; `inclusive=True`
  gives the closed `[lo, hi]`. Element chains test the element
  **centroid**.
- **entity family** (`g.model.select`) — `.in_box` is Gmsh's
  `getEntitiesInBoundingBox`: BRep bbox **containment** (the whole
  entity bbox must fall inside the query box, ~1e-8 expanded). No
  half-open notion → passing `inclusive=` (or any kw) **raises
  `TypeError`**. For an exact geometric predicate use the
  `.on_plane(...)` / `.crossing_plane(spec, mode=...)` verbs (the old
  `g.model.queries.select(on=/crossing=)` selector was removed; its
  behaviour is folded into those verbs).

Never assume a cross-family `.in_box` gives the same set.
(`tests/test_selection_idiom.py::test_point_family_laws` /
`::test_entity_family_laws` lock both families.)

**The legacy selection surface was REMOVED (no shim)** —
selection-unification v2 hard-removed `fem.nodes/elements.get` /
`get_ids` / `get_coords` / `resolve`, the chain
`results.*.select(...).get(...)` terminal,
`g.mesh_selection.add_nodes` / `add_elements` / `from_geometric`,
`g.model.queries.select` / `select_all*` / `line`,
`g.model.selection` / `select_*` (`SelectionComposite`), and the
package exports of both `Selection` classes. Calling them raises
`AttributeError` / `ImportError`. Use `.select()` exclusively. The
`core._selection.Selection` and `viz.Selection` *classes* are
**retained by architecture** (ADR 0015 — the `.result()` payload and
the viewer pick-result type are structurally distinct internal types,
not user entry points); only the exports were dropped.
`EntitySelection.result()` returns the `core` `Selection` payload (so
`.to_label()` / `.to_physical()` / `.to_dataframe()` work, and they
also work directly on `EntitySelection`). "Remove both Selection
classes" was a category error. Retained on the mesh side:
`g.mesh_selection.add(dim, ids, name=)`, `from_physical(...)`,
`filter_set`, `sort_set`, `union`/`intersection`/`difference`.

**Persistence** — chains are query objects, but the live-mesh chain
has a `.save_as(name)` terminal:
`g.mesh_selection.select(...).<verbs>.save_as("my_set")` registers
the result into the live `g.mesh_selection` store (live engine only;
broker/results chains raise — they hold a read-only snapshot). It
then lands in the FEMData snapshot and round-trips as
`results(selection=...)`. For explicit ids use
`g.mesh_selection.add(dim, ids, name=)`.

**Mesh-selection name-seed**: `g.mesh_selection.select(name="my_set")`
seeds id-for-id from an **existing** `g.mesh_selection` set (node ids
for `level="node"`, element ids for `level="element"`), then narrow
with the usual spatial verbs — the fluent equivalent of `filter_set`
over that set. Seeding *directly* from a raw gmsh PG name / apeGmsh
label is **not** a `select()` parameter; use the two-step
`from_physical(...)` **then** `select(name=...)`.

### 1.5  Selection-v2 capability gaps — ADR-0017 (incomplete unification)

v2's mandate was **unification**, not capability reduction. Two
surfaces it removed are **incomplete unification**, *not* WONTFIX
(ADR 0017 supersedes ADR 0016 §4; CHANGELOG v1.6.0):

- **Gap 1 — geometry → named mesh-selection** (the removed one-call
  `g.mesh_selection.from_geometric` / `viz.Selection.to_mesh_*`):
  **capability is INTACT**. It survives via the retained 2-call
  route: pre-mesh `g.model.select(...).to_physical(name)` then
  post-mesh `g.mesh_selection.from_physical(dim, name, ms_name=)`
  (or `g.mesh_selection.add(dim, ids, name=)`). Only the one-call
  *ergonomic* was lost — a future one-liner is an ergonomics
  decision, not a functionality verdict.
- **Gap 2 — the `SelectionComposite` declarative filter grammar**
  (`select_*(labels=/kinds=/*_range=/predicate=/exclude_tags=)`): a
  **unique-capability loss** → a **v2-native `EntitySelection`
  successor is OWED / PLANNED** (tracked in
  `docs/plans/selection-gaps-v3.md`; the deleted
  `tests/test_selection_filters.py`, 33 tests, git-recoverable, is
  its behavioural floor). It is **not** a resurrected
  `SelectionComposite` / `g.model.selection` and must respect every
  v2 invariant (ADR 0015 core→viz boundary). Until it ships, reach
  the capability via the viewer-pick `viz.Selection.filter()` or a
  manual predicate over `g.model.select(...).result()`.

Source of truth: `docs/plans/selection-unification-v2.md` §6 ledger +
ADR 0016/0017. The v2 program itself is COMPLETE and merged; ADR 0017
opens a follow-on feature track, it does not reopen v2.

**Not yet available** (don't reach for this): results sub-composite
`.select()` (`gauss`/`fibers`/`layers`/`line_stations`/`springs` —
use the existing `results.elements.<sub>.in_box/...` helpers).

---

## 2. Parts and the session

### 2.1  A Part is an isolated geometry container

```python
from apeGmsh import Part

col = Part("column")
with col:
    col.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3, label="shaft")
# auto-persists to STEP tempfile + sidecar JSON on exit
```

Parts own their own Gmsh session. They know nothing about meshing,
constraints, loads, or masses. Labels created inside a Part travel
through the STEP round-trip via COM matching.

### 2.2  The session registry imports Parts into one assembly

```python
with apeGmsh(model_name="frame") as g:
    g.parts.add(col, label="col_A", translate=(0, 0, 0))
    g.parts.add(col, label="col_B", translate=(5, 0, 0))
    g.parts.fragment_all()   # conformal interfaces
```

After `fragment_all()`, volumes share faces at interfaces. Labels
survive: `g.labels.entities("col_A.shaft")` still resolves.

### 2.3  Sections builder — inline parametric members

```python
col = g.sections.W_solid(
    bf=150, tf=20, h=300, tw=10, length=2000,
    label="col", lc=50,
)
# Creates labeled sub-regions: col.top_flange, col.web, col.bottom_flange
# col.start_face, col.end_face
```

No Part intermediary — geometry built directly in the session.

---

## 3. Constraints — two-stage pipeline

### 3.1  Stage 1 — declare before meshing

```python
g.constraints.equal_dof("col", "beam", dofs=[1, 2, 3], tolerance=1e-3)
g.constraints.tie("shell", "beam", master_entities=[(2, face)],
                  slave_entities=[(1, edge)], dofs=[1,2,3,4,5,6], tolerance=5.0)
```

Full catalogue (14 declaration verbs on `g.constraints`: `bc`,
`equal_dof`, `rigid_link`, `penalty`, `rigid_diaphragm`, `rigid_body`,
`kinematic_coupling`, `node_to_surface`, `node_to_surface_spring`,
`tie`, `distributing_coupling`, `embedded`, `tied_contact`, `mortar`):

| Level | Method | Record type | Lives on |
|---|---|---|---|
| Node-to-node | `equal_dof`, `rigid_link`, `penalty` | `NodePairRecord` | `fem.nodes.constraints` |
| Node-to-group | `rigid_diaphragm`, `rigid_body`, `kinematic_coupling` | `NodeGroupRecord` | `fem.nodes.constraints` |
| Mixed-DOF | `node_to_surface`, `node_to_surface_spring` | `NodeToSurfaceRecord` (phantom nodes) | `fem.nodes.constraints` |
| Surface | `tie`, `distributing_coupling`, `embedded` | `InterpolationRecord` | `fem.elements.constraints` |
| Surface-to-surface | `tied_contact`, `mortar` | `SurfaceCouplingRecord` | `fem.elements.constraints` |

`node_to_surface_spring` is the **stiff-beam variant of
`node_to_surface`** — use it when the surface masters have free
rotational DOFs (resolves the rigidLink conditioning issue a plain
`node_to_surface` hits on rotation-free masters).

**`node_to_surface` dedup**: when the slave resolves to multiple surface
entities (e.g. a PG spanning several faces), one `NodeToSurfaceDef` is
created aggregating all of them.  Shared-edge mesh nodes are
deduplicated so each original slave node gets exactly one phantom — no
double constraints on shared boundaries.  Return type is always a
single `NodeToSurfaceDef`, never a list.

### 3.2  Stage 2 — resolution in get_fem_data

Records split into two composites on the broker:
- **`fem.nodes.constraints`** (`NodeConstraintSet`) — node-pair, node-group,
  and node_to_surface records
- **`fem.elements.constraints`** (`SurfaceConstraintSet`) — surface ties,
  distributing/embedded interpolations, mortar / tied_contact couplings

⚠️ `apeSees(fem)` does **not** emit any of these to OpenSees — MP
constraints are deferred (§3.3). The loops below are the raw
openseespy you would write yourself to drive a solver that
supports multi-point constraints, or for inspection:

```python
# 1. Phantom nodes first (node_to_surface generates them)
for nid, xyz in fem.nodes.constraints.phantom_nodes():            # verified: tests/test_femdata_from_h5.py::test_round_trip_node_to_surface_record
    ops.node(nid, *xyz)

# 2. Rigid links — grouped by master (all rigid kinds merged)
for master, slaves in fem.nodes.constraints.rigid_link_groups():  # verified: tests/test_femdata_from_h5.py::test_round_trip_node_to_surface_record
    for slave in slaves:
        ops.rigidLink("beam", master, slave)

# 3. Equal DOFs — flat pairs (direct + expanded node_to_surface)
for pair in fem.nodes.constraints.equal_dofs():                   # verified: tests/test_femdata_from_h5.py::test_round_trip_node_to_surface_record
    ops.equalDOF(pair.master_node, pair.slave_node, *pair.dofs)

# 4. OR: rigid diaphragms via the native multi-slave command
for master, slaves in fem.nodes.constraints.rigid_diaphragms():   # verified: tests/test_femdata_from_h5.py::test_round_trip_node_group_record
    ops.rigidDiaphragm(3, master, *slaves)

# 5. Surface constraints — interpolations (tie, distributing, embedded)
for interp in fem.elements.constraints.interpolations():          # verified: tests/test_femdata_from_h5.py::test_round_trip_interpolation_record
    # interp.slave_node, interp.master_nodes, interp.weights, interp.dofs
    ...
# Top-level couplings (mortar, tied_contact)
for coup in fem.elements.constraints.couplings():                 # verified: tests/test_femdata_from_h5.py::test_round_trip_surface_coupling_record
    ...
```

For solvers that don't have grouped commands, the flat fallback still
works — `fem.nodes.constraints.pairs()` expands every record
(including `NodeGroupRecord` and `NodeToSurfaceRecord`) into individual
`NodePairRecord` objects, and you can kind-dispatch with
`K = fem.nodes.constraints.Kind`:

```python
K = fem.nodes.constraints.Kind                                    # verified: tests/test_constraint_resolver.py::test_matches_colocated_pairs
for c in fem.nodes.constraints.pairs():
    if c.kind == K.RIGID_BEAM:
        ops.rigidLink("beam", c.master_node, c.slave_node)
    elif c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
```

When you need fields that the flattened pairs can't expose (e.g.
`phantom_coords` on a compound record), reach for the raw accessor:

```python
for nts in fem.nodes.constraints.node_to_surfaces(): # NodeToSurfaceRecord  # verified: tests/test_femdata_from_h5.py::test_round_trip_node_to_surface_record
    ...
```

### 3.3  ⚠️ Constraint emission is DEFERRED in `apeSees`

There is **no** OpenSees-emission path for multi-point constraints
in the new bridge — `tie`, `rigid_link`, `equal_dof`,
`rigid_diaphragm`, `node_to_surface`, `tied_contact`, `mortar`,
embedded rebar. The `apeSees` `Emitter` protocol **covers the
standard OpenSees commands** (`model`, `node`, `fix`, `mass`,
`uniaxialMaterial` / `nDMaterial` / `section`, `geomTransf`,
`patch` / `fiber` / `layer`, `element`, `timeSeries`,
`pattern_open` / `pattern_close`, `load`, `eleLoad`, `sp`,
`recorder`, `constraints` / `numberer` / `system` / `test` /
`algorithm` / `integrator` / `analysis`, `analyze`) — but it has
**no** `equalDOF`, `rigidLink`, `rigidDiaphragm`, or
`ASDEmbeddedNodeElement` method. So MP constraints simply cannot be
emitted. This is deferred by design — the canonical source is
**ADR 0016 + CHANGELOG v1.6.0** plus this no-emitter-method fact
(the older "FEM-direct, no ingest" phrasing is in the body of
`opensees/architecture/decisions/0009-no-backwards-compat-with-solvers.md`,
which is titled "No back-compat with apeGmsh.solvers" — not a
constraint-emission decision).

```python
# the Emitter protocol surface — note the absence of any MPC verb
# verified: tests/opensees/unit/test_emitter_protocol.py::test_recording_emitter_has_method
```

What this means in practice:

- **Declare constraints on the session as usual** (`g.constraints.tie`,
  `g.constraints.equal_dof`, …). They still resolve into the
  `FEMData` snapshot and are persisted into the **`model.h5`
  neutral zone** by `ops.h5(path)` — so the **viewer / `Results`**
  see them.
- They are **NOT** written into any runnable Tcl / Py / Live
  OpenSees deck. A model whose load path depends on a tie /
  rigid link will be wrong if you run the emitted deck as-is.
- To actually run such a model today you must hand-emit the
  constraint commands yourself by iterating
  `fem.nodes.constraints` / `fem.elements.constraints` (§1.3,
  §3.2) into raw openseespy, **or** wait for the deferred
  constraint-emission feature.

Homogeneous fixities and prescribed displacements **do** have a
path — see §4.3 (`ops.fix` / `p.sp`).

---

## 4. Loads and masses

### 4.1  Loads — pattern-grouped, resolved to nodes/elements

```python
with g.loads.pattern("Gravity"):
    g.loads.gravity("Body", density=2400)
with g.loads.pattern("Wind"):
    g.loads.surface("facade", magnitude=1.2e3, normal=True)
with g.loads.pattern("InternalPressure"):
    # 2-D curve pressure: normal=True picks side from Gmsh boundary
    # orientation; positive magnitude pushes into the structure.
    g.loads.line("InnerArc", magnitude=p, normal=True)
    # Override the auto-side via away_from= for ambiguous interfaces:
    # g.loads.line("Iface", magnitude=p, normal=True, away_from=(0,0,0))
```

After get_fem_data(), the session loads split into
`fem.nodes.loads` (`NodalLoadRecord`) and `fem.elements.loads`
(`ElementLoadRecord`). **`apeSees(fem)` does not consume these.**
They persist into `model.h5` (viewer / `Results`); the iteration
API below is inspection-only:

```python
for load in fem.nodes.loads:                                      # verified: tests/test_femdata_from_h5.py::test_round_trip_nodal_loads
    load.node_id, load.force_xyz, load.moment_xyz   # inspect only
```

To put the same loads into OpenSees you **re-declare them
explicitly** on the bridge, inside a pattern:

```python
ts = ops.timeSeries.Linear()                 # also Constant/Path/Trig/Pulse
with ops.pattern.Plain(series=ts) as p:      # also UniformExcitation
    p.load(pg="Tip", forces=(0.0, -5e4, 0.0, 0.0, 0.0, 0.0))   # verified: tests/opensees/unit/test_emitter_protocol.py::test_pattern_open_close_pair
    # p.load(node=<tag>, forces=(...))         # explicit node alternative
```

`p.load` fans a `pg=` across the group's nodes at build time.
Distributed/body loads have no pattern verb — gravity/body force
is an **element parameter** (`ops.element.<Solid>(pg=...,
body_force=(b1,b2,b3))`); 2-D solids take a `pressure=` arg.

**`face_load` / `face_sp`** (on `g.loads`) distribute a load / BC
directly onto solid faces, **bypassing reference-node coupling**.
`face_sp` resolves into `fem.nodes.sp` (an `SPSet`); both persist
into `model.h5` for the viewer / `Results`. There is **no**
`g.opensees.ingest` consumer (removed) — to drive these in OpenSees,
re-declare on the bridge (`ops.fix` / `p.sp` — §4.3).

### 4.2  Masses — accumulated per node

```python
g.masses.volume("Body", density=2400)
```

After resolution: `fem.nodes.masses` (persisted to `model.h5`;
inspection only — **not** auto-emitted by `apeSees`):

```python
for m in fem.nodes.masses:                                        # verified: tests/test_femdata_from_h5.py::test_round_trip_masses
    m.node_id, m.mass               # mass = (mx,my,mz,Ixx,Iyy,Izz)
print("Total:", fem.nodes.masses.total_mass())
```

Re-declare mass explicitly on the bridge:

```python
ops.mass(pg="Body", values=(mx, my, mz, 0.0, 0.0, 0.0))           # verified: tests/opensees/unit/test_emitter_protocol.py::test_recording_emitter_has_method
# ops.mass(nodes=[...], values=(...))   # explicit-node alternative
```

### 4.3  Boundary conditions / SP — explicit on the bridge

`g.constraints` fixities and `g.loads.face_sp` resolve into the
snapshot but are **not** auto-emitted. Re-declare on `ops`:

```python
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))      # homogeneous SP   # verified: tests/opensees/unit/test_emitter_protocol.py::test_fix_records_tag_and_dofs
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.sp(pg="LoadingPin", dof=3, value=0.01)     # prescribed (non-zero) SP
```

---

## 5. OpenSees bridge — `apeSees(fem)`

OpenSees is driven **after** the session, from the `FEMData`
snapshot, with typed primitives (no string types). `g.opensees` is
**gone** — the entry point is `from apeGmsh.opensees import apeSees`.

**TRAP — apeSees re-declares, it does not ingest.** It does **not**
read `g.loads` / `g.masses` / `g.constraints` / `fem.nodes.*`; there
is **no** replacement for the removed `g.opensees.ingest.*`. Loads /
masses / SPs are re-declared explicitly here (§4): `ops.fix(pg=,
dofs=)`, `ops.mass(pg=, values=)`, `with ops.pattern.Plain(series=ts)
as p: p.load(pg=, forces=)` / `p.sp(...)`; body force is an element
param (`ops.element.<Solid>(pg=, body_force=…)`) — there is no
`eleLoad` pattern verb. **MP constraints are DEFERRED** with no
emission path at all (§3.3). There is **no** `ops.recorder.resolve(fem)`
(agent-invented) — recorder resolution is implicit at
`ops.tcl/py/run`.

```python
from apeGmsh.opensees import apeSees                               # verified: tests/opensees/unit/test_apesees_class.py::test_apesees_constructs_with_fem

fem = g.mesh.queries.get_fem_data(dim=3)
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)                                            # verified: tests/opensees/unit/test_apesees_class.py::test_apesees_model_sets_ndm_ndf

conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)   # verified: tests/opensees/unit/test_apesees_class.py::test_apesees_namespace_is_present
# gravity = element body_force (no eleLoad pattern verb)
ops.element.FourNodeTetrahedron(
    pg="Body", material=conc, body_force=(0.0, 0.0, -9.81 * 2400),
)
ops.fix(pg="Base", dofs=(1, 1, 1))

with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))

ops.tcl("model.tcl")          # separate statements — not fluent
ops.py("model.py", run=False)
ops.h5("model.h5")            # native HDF5: bridge + broker neutral zone
# ops.run(); ops.analyze(steps=10, dt=0.01)   # in-process openseespy
```

Constructors return handles passed by reference
(`material=conc`, `transf=t`, `integration=integ`). `ops.build()`
returns an immutable `BuiltModel` and **requires `ops.model(...)`
first** (`tests/opensees/unit/test_apesees_class.py::test_apesees_build_requires_model_first`,
`::test_apesees_build_returns_built_model`); it is usually implicit
because `ops.tcl/py/h5/run` build internally. Post-build inspection
is broker-side (`fem.inspect.*`) or via
`apeGmsh.opensees.emitter.h5_reader` on the emitted `model.h5`.

### 5.1  Recorders — three declaration surfaces

Post-Phase-9 there are exactly three ways to declare recorders
(`tests/opensees/h5/test_h5_recorders_unified.py`):

1. **Typed primitives** — `ops.recorder.Node(...)` /
   `ops.recorder.Element(...)` / `ops.recorder.MPCO(...)`.
2. **`ops.recorder.declare(...)`** — the canonical fan-out, with
   per-category kwargs (`nodes=` / `elements=` / `line_stations=` /
   `gauss=`) and a `raw=` escape hatch. No `ndm` / `ndf` here.
3. **`DomainCaptureSpec`** (`apeGmsh.results.capture.spec`) for
   in-process capture: `ops.domain_capture(spec, path=)` (live), or
   `DomainCapture.from_h5(model_h5, spec=, fem=, output=)` (file).

The canonical component vocabulary is the neutral top-level
`apeGmsh._vocabulary` (`expand_shorthand` / `expand_many`);
`results._vocabulary` is a one-shot DeprecationWarning shim. The
old `Recorders` fluent helper was **deleted** (legacy paths raise
`ImportError`).

### 5.2  Native model.h5 — two zones + schema constants

`ops.h5(path)` writes a **two-zone** `model.h5`: a *neutral* zone
(broker-written — nodes/elements/PGs/labels/constraints/loads/masses,
the same content `g.save()` writes) and an `/opensees/` zone
(bridge-written — transforms, recorders, cuts/sweeps).

Two **independent** schema constants — check `origin/main` HEAD
before bumping either:

- `SCHEMA_VERSION` (bridge, `opensees/emitter/h5.py`) = **2.5.0** —
  stamps `/opensees/…`. (`tests/opensees/h5/test_h5_recorders_unified.py::test_schema_version_is_2_5_0`)
- `NEUTRAL_SCHEMA_VERSION` (broker, `mesh/_femdata_h5_io.py`) =
  **2.4.0** — stamps the root neutral zone.
  (`tests/test_femdata_to_h5.py::test_to_h5_writes_meta`)

Symmetry contract: every record-set group uses one compound dtype
(`target_kind, target, payload_kind, payload`); helper
`mesh/_record_h5.py`. Build-time fan-out lives in
`opensees/_internal/build.py` (topological order via Kahn; ADR P11
Option A — no auto-register, missing dep → `BridgeError`; transform
spec dedupe `VECXZ_TOL=1e-9`).

**Beam local axes (`vecxz`).** The bridge **already persists**
`vecxz`: `h5.py` writes `/opensees/transforms/{type}_{tag}/` with
`per_element_vecxz`(1,3) + `per_element_emitted_tag`(1,) — **one
group per `geomTransf` call** (a documented schema deviation, NOT
per-element `(n,3)`). The `element_id → vecxz` join is keyed by FEM
eid and **must live in `h5_reader`** (`H5Model.element_local_axes_vecxz()`)
— an AST test bars `viewers/` from importing anything but
`opensees.emitter.h5_reader`. `from_fem` → empty (no transforms
group). `_beam_geometry.py::LocalFrame` / `iter_local_frames` is the
reusable section-extrusion seam — reuse it, don't fork. DEFERRED:
persisting the orientation **rule** (Cartesian/Cylindrical origin/
axis/roll) — discarded in the Phase-4 fan-out; the h5 schema reserves
attrs but `_write_transforms` never writes them.

```python
# bridge persists per-call vecxz; h5_reader joins it back by FEM eid
# verified: tests/opensees/h5/test_h5_local_axes_vecxz.py::test_element_local_axes_vecxz_end_to_end
```

### 5.3  Section cuts / sweeps (`cuts` v4) — model.h5 persistence

`SectionCutDef` / `SectionSweepDef` persist under `/opensees/cuts/`
and `/opensees/sweeps/` (element_ids carry OpenSees tags, not FEM
eids — hence under `/opensees/`). Three writer paths:

- in-shot — `apeSees.h5(path, cuts=[…], sweeps=[…])`
  (`tests/opensees/h5/test_h5_apesees_cuts.py::test_apesees_h5_with_cuts_writes_groups`,
  `::test_apesees_h5_with_sweep_writes_groups`);
- append — `cuts.persist_to_h5(path, …)` (deletes only the supplied
  groups);
- primitive — `cuts._h5_io.write_cuts_into(f, …)` (raises if the
  groups already exist).

The viewer auto-loads persisted cuts; an explicit `viewer(cuts=[…])`
kwarg **wins** over persisted (no merge). Schema = bridge **2.5.0**
(`test_h5_apesees_cuts.py::test_apesees_h5_bumps_schema_version_to_2_5_0`).
DEFERRED v4.1: `/opensees/drifts/` persistence; live edit of
persisted cuts; the `apeGmsh.cuts` → `apeGmsh.outputs` rename.

---

## 6. Persistence — native `model.h5` round-trip

The session and the broker both write the **neutral-zone** HDF5
(`mesh/_femdata_h5_io.py`); `apeSees(fem).h5(p)` is a separate
solver-enrichment step (§5.2), not invoked by `g.save()`.

### 6.1  Session autosave + manual save

```python
# Autosave on context-manager exit:
with apeGmsh(model_name="s", save_to="model.h5") as g:   # verified: tests/test_session_save.py::test_autosave_writes_on_exit
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(3)
# model.h5 written before gmsh finalizes

# Manual save (same path, or an explicit one):
with apeGmsh(model_name="d", save_to="model.h5") as g:   # verified: tests/test_session_save.py::test_manual_save_uses_save_to_when_no_arg
    ...
    g.save()                      # uses save_to=
    g.save("checkpoint.h5")       # or an explicit path
```

- `save_to=None` (default) → **no** autosave. `g.save()` with neither
  a path nor `save_to=` raises `RuntimeError`
  (`tests/test_session_save.py::test_manual_save_without_path_or_save_to_raises`).
- `overwrite=False` (ctor) + an existing target → `g.save()` raises
  `FileExistsError`
  (`tests/test_session_save.py::test_overwrite_false_raises_on_existing`);
  `overwrite=True` (default) replaces it.
- If autosave fails (bad path), gmsh **still finalizes** — the
  failure is a `UserWarning`, not a hard error
  (`tests/test_session_save.py::test_autosave_failure_does_not_block_finalize`).

### 6.2  `FEMData.to_h5` / `FEMData.from_h5`

```python
fem.to_h5("model.h5", model_name="demo")        # verified: tests/test_femdata_to_h5.py::test_to_h5_writes_meta
rebuilt = FEMData.from_h5("model.h5")            # verified: tests/test_femdata_from_h5.py::test_round_trip_nodes_and_elements
```

The round-trip is **lossless** for every record type — nodes,
elements (per-type), PGs, labels, the full constraint zoo
(`NodePair` / `NodeGroup` / `NodeToSurface` with re-derived
sub-records / `Interpolation` / `SurfaceCoupling` with CSR
`slave_records`), loads (per-pattern), `SPRecord`s, masses, and
`/mesh_selections/` (`tests/test_femdata_from_h5.py::test_round_trip_node_to_surface_record`,
`::test_round_trip_surface_coupling_record`,
`::test_round_trip_mesh_selections`). The session-save path produces
a file `FEMData.from_h5` reads directly
(`tests/test_femdata_from_h5.py::test_session_save_then_from_h5`).

- `/meta/schema_version` = **2.4.0** (neutral); empty composites
  write no group at all
  (`tests/test_femdata_to_h5.py::test_to_h5_omits_empty_groups`).
- `/meta/snapshot_id` carries the broker's content hash and is
  **preserved** across the round-trip
  (`tests/test_femdata_to_h5.py::test_to_h5_snapshot_id_in_meta`,
  `tests/test_femdata_from_h5.py::test_snapshot_id_preserved`).
- Error paths fail loud:
  ```python
  from apeGmsh.opensees.emitter.h5_reader import (   # verified: tests/test_femdata_from_h5.py::test_missing_meta_raises
      MalformedH5Error, SchemaVersionError,
  )
  # missing /meta            -> MalformedH5Error
  # schema major mismatch    -> SchemaVersionError   # verified: tests/test_femdata_from_h5.py::test_wrong_schema_major_raises
  ```

---

## 7. Post-processing — `Results`

`Results` reads OpenSees output. Three constructors, all
context-managers (`with ... as r:` releases the HDF5 handle):

| Constructor | Source | Bind |
|---|---|---|
| `Results.from_native(path, *, fem=None)` | apeGmsh native `model.h5` results | embedded `/model/` snapshot, or your `fem=` |
| `Results.from_mpco(path, *, fem=None, merge_partitions=True)` | STKO `.mpco` HDF5 | synthesizes a **partial** FEMData from `MODEL/` if `fem=` omitted |
| `Results.from_recorders(spec, output_dir, *, fem)` | `.out`/`.xml` from a Tcl/Py recorder run | requires `fem=` (spec's `snapshot_id`); transcodes → native + caches |

```python
from apeGmsh.results import Results

with Results.from_native("results.h5") as r:                # verified: tests/test_results_modes.py::test_modes_accessor_returns_scoped_results
    r.stages                       # list[StageInfo]
    s = r.stage("dynamic")         # scope to one stage
    s.kind, s.n_steps              # stage-scoped metadata
    slab = r.nodes.get(pg="Body", component="displacement_x")
    slab.values, slab.time, slab.node_ids
```

- **Stages & modes.** `r.stages` lists all stages; `r.stage(name_or_id)`
  returns a stage-scoped `Results`. `r.modes` is the list of
  `kind="mode"` stages as mode-scoped `Results`, each exposing
  `.eigenvalue` / `.frequency_hz` / `.period_s` / `.mode_index`
  (`tests/test_results_modes.py::test_mode_indexing_and_attrs`).
  Sort modes for a stable index:
  `sorted(r.modes, key=lambda m: m.mode_index)`. Accessing a
  mode-only property on a non-mode / unscoped stage raises
  `AttributeError`
  (`tests/test_results_modes.py::test_mode_props_raise_on_non_mode_stage`).
- **Node / element reads.** `r.nodes.get(*, pg=/label=/selection=/ids=,
  component=, time=, stage=)` → a slab
  (`tests/test_results_modes.py::test_mode_shape_is_single_step`).
  `r.elements` additionally owns the sub-composites
  `r.elements.{gauss, fibers, layers, line_stations, springs}`. The
  fluent `.select()` chain (§1.4) composes spatial verbs then a
  terminal `.values(component=…)`
  (`tests/test_results_selection_backfill.py::test_select_then_values_matches_get`).
- **Component shorthand.** Component names expand DOF-aware:
  `"displacement"` → `displacement_x/y/z` clipped to `ndm`;
  `"stress"` → 6 in 3D / 3 in 2D; `"reaction"` combines forces +
  moments. Unknown names raise `ValueError`
  (`tests/test_results_shorthand.py::test_displacement_3d`,
  `::test_unknown_name_raises`).
- **MPCO multi-partition.** `Results.from_mpco("run.part-0.mpco")`
  auto-discovers `run.part-*.mpco` siblings and merges them (boundary
  nodes dedup by id, elements concatenate). `merge_partitions=False`
  reads just the one file.

### 7.1  Binding contract — pairing is YOUR responsibility

`from_native` / `from_mpco` / `r.bind(fem)` compute and store a
`snapshot_id`, but **nothing enforces it**. `BindError` is an
exported stub class that **nothing raises**
(`tests/test_results_bind.py::test_bind_accepts_mismatched_fem`
binds a *different-mesh* FEMData with no error). The hash was
deliberately de-enforced because legitimate workflows (re-meshing,
importing an `.mpco` against a fresh `fem`) tripped it. **Do not
expect a mismatch to error** — pairing a FEMData with the results
file from the *same run* is the user's job. If results look wrong /
selectors miss, suspect a **stale FEMData paired with a fresh
results file** (the removed check would have caught it).

```python
with Results.from_native(path) as r:                        # verified: tests/test_results_bind.py::test_bind_after_construction
    r2 = r.bind(fem)            # returns a NEW Results; no hash validation
```

MPCO synthesizes only a **partial** FEMData (negative synthetic
element codes, no PG regions, `snapshot_id` won't match a native
one) — PG/label queries against an unbound MPCO raise / return
empty; id queries still work
(`tests/test_results_mpco_partial_fem.py::test_pg_queries_without_regions_return_empty_or_raise`,
`::test_id_query_works_without_pg`). Pass `fem=` to recover names.

### 7.2  Static plots & the interactive viewer

```python
results.plot.contour("displacement_z", step=-1)     # static matplotlib
results.plot.deformed(step=-1, scale=50, component="stress_xx")
results.plot.history(node=412, component="displacement_x")

results.viewer(blocking=False)                       # interactive (subprocess)
```

- `results.plot` is the headless matplotlib renderer (mirrors the
  viewer's diagram catalog; needs the `[plot]` extra).
- `results.viewer()` opens the post-solve VTK viewer. **It already
  calls `.show()` — never chain `results.viewer().show()`** (opens
  two windows).
- The default `blocking=True` runs in-process and **native-crashes a
  Jupyter / VS Code kernel** (VTK+Qt event loop) even with a GPU. In
  a notebook use **`results.viewer(blocking=False)`** (spawns
  `python -m apeGmsh.viewers <path>` — needs a Results opened from
  disk; raises `RuntimeError` for in-memory), or run
  `python -m apeGmsh.viewers <file>` from a terminal, or stay
  headless with `results.plot.*` + `APEGMSH_SKIP_VIEWER=1` (the
  viewer call then returns `None`, so the cell survives
  `jupyter nbconvert --execute` / CI).
- **MPCO carries no beam `vecxz`.** This OpenSees ("Ladruno") build
  writes **no** `MODEL/LOCAL_AXES` to `.mpco` for beams
  (disp/force/elastic BeamColumn) even with `-E … localAxes`. Correct
  line/section diagram orientation needs the **native** path
  (`apeSees → model.h5 /opensees/transforms/ → Results.from_native`)
  or an explicit `{eid: vecxz}` injection seam — **don't** try to
  "read MPCO LOCAL_AXES", the data is not in the file. The MPCO
  injection seam itself is still DEFERRED (needs design).
- **`.mpco` + SeaDrive/OneDrive/Dropbox = kernel abort.** Writing an
  MPCO recorder onto a synced virtual FS native-crashes the kernel
  at `ops.wipe()` (deferred `H5Fclose` vs the sync client) and
  leaves the HDF5 open-for-write flag set. Point the recorder `file=`
  at **local disk**, `ops.wipe()` in the **same cell**, then
  `shutil.copy2`. `HDF5_USE_FILE_LOCKING=FALSE` does **not** fix it.

The viewer GUI is not testable headless here (no GPU/OpenGL); the
checks above are behavioural facts isolated empirically, not
asserted by a passing test in this environment.

### 7.3  Known result-catalog gap — Tri31 strain

`Tri31` (`SRC/element/triangle/Tri31.cpp`) has **no element-level
`"strain"` / `"strains"` `setResponse` branch** —
`ops.eleResponse(eid, "strains")` → `[]`. MPCO is unaffected (the
per-material path `eleResponse(eid,"material","<gp>","strain")`
works). The response catalog routes around it:
`PER_MATERIAL_STRAIN_CLASSES = frozenset({"Tri31"})` lives in
**`apeGmsh.opensees._response_catalog`** (NOT the old
`solvers/_element_response.py` — `solvers/` was deleted in Phase
8.8). To remove the workaround: patch `Tri31.cpp` (response ID 4 is
free; mirror `FourNodeQuad.cpp:1412+1520`, ~25 lines), then drop
`"Tri31"` from the set.

```python
# the routing constants + helpful KeyError live in _response_catalog
# verified: tests/test_results_element_response.py::test_lookup_miss_is_helpful
```

---

## 8. Viewer (pre-solve / mesh)

```python
# BRep geometry viewer — geometry only (labels, PGs, entities)
g.model.viewer()

# Mesh viewer — shows the mesh and (optionally) FEM overlays
g.mesh.viewer()
fem = g.mesh.queries.get_fem_data(dim=3)
g.mesh.viewer(fem=fem)        # FEM overlays live on the mesh viewer
```

`g.model.viewer()` is **geometry-only** — it does not accept `fem=`.
Loads, constraints, and masses are mesh-resolved concepts, so they
live on `g.mesh.viewer(fem=fem)` instead.

The mesh viewer with `fem=` shows:
- Load arrows (magnitude-scaled, textbook solid style)
- Moment curved arrows (270 arc with cone arrowhead)
- Mass spheres (scaled by mass^1/3, viridis colormap)
- Constraint lines (colored by kind, master sphere markers)
- Surface constraint highlights (semi-transparent face patches)
- Per-kind toggles via Constraints tab
- Overlay sizing sliders in Preferences (0.1x-10x multiplier)

All overlay sizes use `_characteristic_length()` (geometric mean of
significant bounding box spans) — works for any model scale.

### 8.1  Viewer architecture (Results & mesh viewers share this)

- **`ViewerData` is the single viewer↔model seam** (`apeGmsh.viewers.data`):
  `ViewerData.from_fem(fem)` (live) / `ViewerData.from_h5(path)`
  (post-solve) are interchangeable accessors. Read-side row
  dataclasses in `viewers/data/_records.py` are **deliberately
  duplicated** (not imported from `mesh.records`) — ADR 0014 keeps
  the seam one-directional; an AST test
  (`tests/test_viewers_pure_h5_consumer.py`) fails if any
  `viewers/*.py` imports `from apeGmsh.mesh` or any opensees module
  other than `emitter.h5_reader`. `director.fem` was removed →
  `director.view`.
- **Three-level outline: Geometry → Composition → Layer.** A
  *Geometry* (`viewers/diagrams/_geometries.py`) owns its name +
  `deform_enabled/field/scale` + its own CompositionManager; only
  the active geometry renders; the last one can't be deleted.
  *Composition* (UI label: *Diagram*) is a named layer bundle, per
  Geometry. *Layer* is one renderable (the internal class is still
  `Diagram`). Deformation is a **per-Geometry** property (warps that
  geometry's points; substrate + wireframe + OPID-bearing layers
  follow). `director.geometries` = GeometryManager (subscribe here
  for state); `director.compositions` = the active Geometry's
  manager (back-compat property). There is always a locked
  `GEOMETRY_ID` "Geometry" composition (renamable, not deletable,
  refuses layers); `Esc` returns to it.
- **VTK keyboard shortcuts**: in any pyvistaqt/VTK-hosting window set
  `sc.setContext(Qt.ShortcutContext.ApplicationShortcut)` — the
  default `WindowShortcut` does NOT fire while the VTK viewport has
  focus. Pair single-letter shortcuts with a focus-widget guard
  (skip while a `QLineEdit/QSpinBox/QComboBox/QPlainTextEdit/QTextEdit`
  is focused) and **hold a reference** to every `QShortcut` (else Qt
  GC reaps it).
- **Verification debt (highest-priority latent risk):** the
  topological hide-cascade + silhouette teardown + dim-0 point-glyph
  rebuild added to *shared* `viewers/core/visibility.py` (PR #199)
  silently ships into `g.mesh.viewer` + `results.viewer` (both build
  a `VisibilityManager`) but was **only verified in `g.model.viewer`**.
  mesh.viewer hides FE *elements* not BRep entities, yet
  `_expanded_hidden` uses `gmsh.model.getAdjacencies` (BRep) —
  run a real mesh.viewer + results.viewer hide/reveal regression
  before trusting it there.
- **Trap:** pre-mesh Loads/Masses live in `g.loads.load_defs` /
  `g.masses.mass_defs`, **not** `*_records` (those are the
  resolved post-mesh output, empty pre-mesh). No pre-mesh
  load/mass arrows by design.
- The big viewer modernization arcs (plans 01-08 + the ParaView
  pipeline-browser alignment) are **done** — don't re-propose them
  or plan 02/07/01-VTK-capture (deferred with reason). Remaining
  work is the Known-limitations list (e.g. FEM-input visibility
  drift between the mesh-outline eye state and the
  Loads/Mass/Constraints tab checkboxes) and is **gated on a user
  hands-on smoke verdict** — ask, don't assume.

Viewer rendering is not testable headless in this environment (no
GPU/OpenGL); the items above are design-of-record + empirical UI
laws, not statements backed by a passing test here.

---

## 9. Pitfalls & gotchas

### 9.1  `remove_duplicates` tolerance is unit-dependent
mm models: `tolerance=1e-3`. Metre models: `tolerance=1e-6`.

### 9.2  `set_size_sources(from_points=False)` after CAD import
CAD files bake tiny per-vertex `lc` values. Global sizing won't
override unless you disable point sources.

### 9.3  `renumber(dim=, base=1)` before `get_fem_data`
Gmsh tags are non-contiguous. OpenSees needs dense 1-based IDs.
Call `g.mesh.partitioning.renumber(dim=N, method="rcm", base=1)` —
`dim` is the element dimension you will later extract. **Frame this
as making Gmsh's non-contiguous tags dense + 1-based for a clean
export, NOT as a bandwidth optimisation**: OpenSees does its own DOF
renumbering at solve time via the `numberer` command on the equation
graph, so the mesh-side RCM order is overwritten before the sparse
matrix is built. Mention `method="rcm"` only if asked — it affects
saved-file ordering, not solve-time perf.

### 9.4  Don't modify FEMData after extraction
FEMData is an immutable snapshot. Re-declare on session and re-extract.

### 9.5  `generate(dim=2)` on a solid is valid
Imports solid for BRep robustness, meshes only surfaces. Viewer
auto-skips unmeshed dimensions.

### 9.6  Don't hold tags past `fragment_all()`
OCC renumbers entities. Re-query via `g.parts.instances` or
`g.model.select(...)` after fragmentation.

### 9.7  Tie penalty default is 1e18
Drop to 1e10-1e12 if Newton fails to converge. The element only
needs K >> parent element stiffness.

### 9.8  `in_box` is half-open `[lo, hi)` (point family)
Every point-family `.select().in_box(...)` (and the retained
`filter_set(in_box=)`) excludes a coordinate / centroid lying exactly
on the **upper** box face by default (matches the results side;
adjacent boxes don't double-count). Pass `inclusive=True` for the
closed box `[lo, hi]`. The entity family (`g.model.select().in_box`)
has **no** `inclusive=` knob — passing it raises `TypeError` (use
`.on_plane(...)` / `.crossing_plane(spec, mode=...)` for an exact
predicate).

### 9.9  `crack()` ordering + shared rim
`g.mesh.editing.crack(pg, *, dim=1, open_boundary=None, normal=None,
side_labels=True)` wraps Gmsh's Crack plugin. Call it **after**
`generate()` and **before** `get_fem_data` / `renumber`. Default
keeps the crack-curve boundary vertices **shared** (interior /
closed-rim crack); `open_boundary="<pg one dim lower>"` duplicates
them (a free-surface mouth). With `dim=2` `side_labels=True`
auto-creates `<pg>_normal` / `<pg>_inverted` PGs (classified by
signed distance from an adjacent tet centroid — the physical side is
**not** inferable from the triangles, so classify at runtime, never
assume `original == inverted`). Each call creates exactly one new
face entity. Don't call `gmsh.plugin.setNumber("Crack", …)` directly.
(`tests/test_mesh_editing_crack.py::test_crack_side_labels_default_creates_named_pgs`)

### 9.10  These selection paths fail loud (no silent wrong answer)
- `results(...)` with `selection=` on an import-origin FEM
  (`from_msh` / MPCO / native — no `mesh_selection`) raises
  `RuntimeError`. Build the selection on `g.mesh_selection` so it
  travels into the snapshot.
- A `g.loads` / `g.masses` whose `__ms__` named selection is gone
  raises `KeyError` instead of binding the load/mass to zero nodes.
- `results.elements.in_box/nearest_to/on_plane` (and centroid-based
  chains) raise `KeyError` if an element's connectivity references a
  node absent from the FEM — no more silently-corrupted centroid.

---

## 10. Anti-patterns

### 10.1  ❌ `equal_dof` for non-matching meshes → use `tie`
`equal_dof` needs co-located nodes. `tie` uses shape function
interpolation for non-matching interfaces.

### 10.2  ❌ `g.mesh.generate()` → use `g.mesh.generation.generate()`
No shortcut methods on parent composites. Sub-composite prefix required.

### 10.3  ❌ Skip `make_conformal` after STEP import
Touching bodies need `remove_duplicates` + `make_conformal` for
shared interfaces.

### 10.4  ❌ Store tags in local dicts → use `g.physical`
Physical groups are the only way to address mesh subsets by name across
the pipeline. Local dicts don't survive fragmentation.

### 10.5  ❌ `fem.node_ids` (old API) → use `fem.nodes.ids`
The FEMData API was rebuilt as composites. Old flat attributes no
longer exist. Use `fem.nodes.*` and `fem.elements.*`.

### 10.6  ❌ Hand-stitching spatial filters → use `.select()` chain
For a composed/spatial selection don't manually `np.isin` /
intersect results, and don't assume the entity-family
`g.model.select().in_box` matches a point-family box. Use the one
canonical chain (§1.4): `fem.nodes.select(pg=...).in_box(...)
.on_plane(...)`, set algebra with `| & - ^`. The old single-shot
`fem.*.get/.resolve` / `g.mesh_selection.add_nodes` accessors were
**removed** — `.select()` is the only accessor (its no-arg / `pg=`
seed covers the simple case too).

### 10.7  ❌ `results.elements.gauss.select(...)`
Not shipped yet. Results sub-composites (`gauss`/`fibers`/`layers`/
`line_stations`/`springs`) have **no** `.select()` — use their
existing `.in_box/.nearest_to/.on_plane` helpers.
(Mesh-selection name-seed, by contrast, **is** shipped:
`g.mesh_selection.select(name="my_set")` seeds id-for-id from an
existing set — see §1.4. Seeding directly from a raw gmsh PG name /
label is still not a `select()` parameter; use the two-step
`from_physical(...)` then `select(name=...)`. The one-step
`from_geometric` bridge was removed — Gap 1 in §1.5; the capability
is intact via that 2-call route.)

### 10.8  ❌ Reaching for `g.opensees` / `g.opensees.ingest`
Both are **gone** (Phase 8). Entry point is
`from apeGmsh.opensees import apeSees`. apeSees does not ingest
`g.*` — re-declare loads/masses/SPs explicitly (§5 TRAP). MP
constraints have no emission path (§3.3).

### 10.9  ❌ Expecting a `BindError` on a mismatched FEMData
`BindError` is an exported stub that nothing raises (§7.1). Pairing
the right FEMData with a results file is the user's responsibility;
there is no guard.

---

## 11. Quick reference

```python
from apeGmsh import apeGmsh, Part
from apeGmsh.opensees import apeSees
from apeGmsh.results import Results

with apeGmsh(model_name="model", verbose=True, save_to="model.h5") as g:
    # Geometry
    g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
    g.model.io.load_step("file.step")
    g.model.boolean.fragment(objects, tools)
    g.model.transforms.translate(tags, dx, dy, dz)

    # Physical groups
    g.physical.add_volume(tags, name="Body")
    g.physical.add_surface(tags, name="Base")

    # Pre-mesh definitions
    g.constraints.equal_dof("col", "beam", dofs=[1, 2, 3])
    with g.loads.pattern("dead"):
        g.loads.gravity("Body", density=2400)
    g.masses.volume("Body", density=2400)

    # Mesh
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)

    # FEM broker
    fem = g.mesh.queries.get_fem_data(dim=3)     # -> snapshot
    base = fem.nodes.select(pg="Base")           # -> MeshSelection
    ids, coords = base.ids, base.coords          # bulk arrays
    # canonical fluent selection (all 4 levels; same verbs everywhere):
    sel = (fem.nodes.select(pg="Body")
               .in_box((0, 0, 0), (1, 1, 1))   # half-open; inclusive=True → closed
               .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6))  # tol REQUIRED
    nodes = sel.result()                        # → NodeResult
    both = fem.nodes.select(ids=a) | fem.nodes.select(ids=b)   # set algebra
    faces = g.model.select("box", dim=2).in_box(lo, hi)        # entity family
    print(fem.inspect.summary())               # introspection

    # Viewer with FEM overlays (sees session loads/masses/constraints)
    g.mesh.viewer(fem=fem)
# model.h5 (neutral zone) autosaved on exit; FEMData.from_h5("model.h5") reloads it

# OpenSees — post-session bridge. Loads/masses/SPs are re-declared
# explicitly (not auto-pulled from g.*); MP constraints deferred (§3.3).
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
c = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(
    pg="Body", material=c, body_force=(0.0, 0.0, -9.81 * 2400),
)
ops.fix(pg="Base", dofs=(1, 1, 1))
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))
ops.h5("results.h5")           # native: bridge /opensees/ + broker neutral zone

# Post-processing — pair the SAME fem (no BindError guard, §7.1)
with Results.from_native("results.h5", fem=fem) as r:
    slab = r.nodes.get(pg="Tip", component="displacement_z")
    r.plot.history(node=base.ids[0], component="displacement_z")  # headless
```

---

## 12. Lower-level skills

| Skill | Use when |
|---|---|
| `gmsh-structural` | Raw Gmsh API, meshing algorithms, OCC kernel, transfinite internals |
| `opensees-expert` | Constraint handlers, analysis commands, element internals, convergence |
| `fem-mechanics-expert` | FEM theory, shape functions, constitutive models, penalty vs Lagrange |
| `stko-to-python` | Loading STKO/MPCO HDF5 recorder output for post-processing |
