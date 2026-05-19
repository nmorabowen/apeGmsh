---
name: apegmsh-helper
description: Use whenever the user is working with apeGmsh — the structural-FEM wrapper around Gmsh with OpenSees integration. Triggers on building FEM models from CAD/STEP imports, Part-based assembly workflows, composite-based geometry/mesh/constraint APIs (g.model, g.mesh, g.physical, g.constraints, etc.), the apeSees(fem) OpenSees bridge with typed primitives, loads/masses/constraints resolution into the FEMData broker, and exporting models to OpenSees Tcl or openseespy scripts. Covers apeGmsh's own abstractions on top of Gmsh and OpenSees. For raw gmsh API questions see the gmsh-structural skill; for raw OpenSees analysis commands see opensees-expert; for FEM theory first principles see fem-mechanics-expert.
---

# apeGmsh helper

apeGmsh is a structural-FEM wrapper around [Gmsh](https://gmsh.info) with a
composition-based API and an OpenSees integration.  The library describes a
structural model **once** — geometry, physical groups, loads, masses,
constraints — and feeds it to any solver through a snapshot broker (`FEMData`).

This skill teaches **apeGmsh's vocabulary and idioms**.  It does not re-teach
Gmsh or OpenSees — for those, see the cross-reference section at the bottom.

---

## 1. Mental model

### 1.1  The session owns composites

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="my_model", verbose=True) as g:
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
| `g.constraints` | Pre-mesh constraint definitions (12 types) |
| `g.loads` | Pre-mesh load definitions (5 types, pattern grouping) |
| `g.masses` | Pre-mesh mass definitions (4 types) |
| `g.loader` | `.msh` file loader — build a `FEMData` snapshot from a saved mesh |
| `g.inspect` | Session diagnostics |
| `g.plot` | Matplotlib (optional dep) |
| `g.view` | Gmsh post-processing views |

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
| `g.mesh.editing` | `embed`, `set_periodic`, `remove_duplicate_nodes` |
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
fem = g.mesh.queries.get_fem_data(dim=3)
# or
fem = FEMData.from_gmsh(dim=3, session=g)
```

`fem` is a **FEMData snapshot** organized by what the engineer needs:

```
fem
  |-- .nodes              NodeComposite
  |     |-- .ids          ndarray(N,) — all node IDs (object dtype, yields Python int)
  |     |-- .coords       ndarray(N, 3) — all coordinates (float64)
  |     |-- .get(pg=, label=)  → NodeResult (iterable of (id, xyz) pairs)
  |     |-- .get_ids(pg=, label=)  → ndarray
  |     |-- .index(nid)   → O(1) array index lookup
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
  |     +-- .masses       MassSet (lumped nodal masses)
  |
  |-- .elements           ElementComposite
  |     |-- .ids          ndarray(E,) — all element IDs concatenated
  |     |-- .connectivity ndarray(E, npe) — ONLY if mesh is homogeneous
  |     |                                  (raises TypeError on mixed types)
  |     |-- .get(pg=, label=)  → GroupResult (iterable of ElementGroup)
  |     |-- .resolve(pg=, label=, element_type=)
  |     |                      → flat (ids, connectivity) for single-type selections
  |     |-- .constraints  SurfaceConstraintSet (tie, mortar, tied_contact)
  |     +-- .loads        ElementLoadSet (pressure, body force)
  |
  |-- .info               MeshInfo (n_nodes, n_elems, bandwidth, elem_type_name)
  +-- .inspect            InspectComposite (summary, tables, source tracing)
```

**Selection API** — get subsets by physical group or label. The
canonical fluent `.select()` chain (§1.4) is the **only** accessor:
`fem.nodes.select(...)` / `fem.elements.select(...)` return a
`MeshSelection` with `.ids` / `.coords` / `.connectivity` /
`.groups()` / `.result()` / `.resolve()`. (The old single-shot
`fem.nodes.get/get_ids/get_coords`, `fem.elements.get/resolve` were
**removed** by selection-unification v2 — there is no shim.)
```python
# .result() on a node selection -> NodeResult (yields (id, xyz) pairs;
# ids are Python ints)
for nid, xyz in fem.nodes.select(pg="Base").result():
    ops.node(nid, *xyz)

# Or read the bulk arrays directly off the MeshSelection
sel = fem.nodes.select(pg="Base")
sel.ids                 # list[int]
sel.coords              # ndarray (N, 3) float64
sel.result().to_dataframe()   # pandas DataFrame (via NodeResult)
```

**Kind constants** — no magic strings (typed as `ClassVar[str]`):
```python
K = fem.nodes.constraints.Kind          # ConstraintKind
K.RIGID_BEAM, K.EQUAL_DOF, K.TIE, ...   # 13 values
K.NODE_PAIR_KINDS, K.SURFACE_KINDS      # frozensets for routing
```

**Grouped iteration** — the FEMData constraint-iteration API.
Note: `apeSees(fem)` does **not** emit these (MP constraints are
deferred — §3.3); iterate them yourself only for inspection or to
hand-drive a solver that supports multi-point constraints (the
`ops.*` below is raw openseespy you write yourself):
```python
# Rigid links — accumulated by master across rigid_beam/rod,
# rigid_diaphragm, rigid_body, kinematic_coupling, and
# node_to_surface phantom links
for master, slaves in fem.nodes.constraints.rigid_link_groups():
    for slave in slaves:
        ops.rigidLink("beam", master, slave)

# Equal DOFs — flat (direct pairs + expanded node_to_surface)
for pair in fem.nodes.constraints.equal_dofs():
    ops.equalDOF(pair.master_node, pair.slave_node, *pair.dofs)
```

**Introspection** — what you have and why:
```python
print(fem.inspect.summary())
print(fem.inspect.constraint_summary())   # kind counts + source names
fem.inspect.node_table()                  # DataFrame
fem.inspect.physical_table()              # DataFrame
```

### 1.4  `.select()` — the canonical fluent selection idiom

`.select()` is the **one canonical, daisy-chainable** selection idiom.
It exists at **all four levels** with the *same* verb names and set
algebra:

| Entry point | Returns | Family | Terminal |
|---|---|---|---|
| `g.model.select(target, *, dim=)` | `EntitySelection` | entity | `.to_label`/`.to_physical`/`.to_dataframe`; `.result()` → `Selection` payload |
| `fem.nodes.select(...)` | `MeshSelection` | point | `.ids`/`.coords`/`.result()` → `NodeResult` |
| `fem.elements.select(...)` | `MeshSelection` | point | `.ids`/`.groups()`/`.result()` → `GroupResult` |
| `results.nodes.select(...)` / `results.elements.select(...)` | `MeshSelection` | point | `.values(component=, time=, stage=)` → slab |
| `g.mesh_selection.select(*, level=, dim=, ids=, name=)` | `MeshSelection` | point | `.ids`/`.coords`/`.result()`; `.save_as(name)` (live engine) |

**Refining verbs** (identical on every chain), each returns a new
chain of the same type so they compose:

```python
sel = (fem.nodes.select(pg="Body")          # seed by pg/label/tag/dim/ids
           .in_box((0, 0, 0), (1, 1, 1))     # half-open [lo, hi)
           .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
           .nearest_to((0.5, 0.5, 0.0), count=4)
           .where(lambda xyz: xyz[2] == 0.0))
nodes = sel.result()                          # -> NodeResult
```

- `.in_box(lo, hi, *, inclusive=False)` · `.in_sphere(center, radius)`
  · `.on_plane(point, normal, *, tol=)` · `.nearest_to(point, *,
  count=1)` · `.where(predicate)`
- **Set algebra**: `|` `&` `-` `^` and the aliases `.union` /
  `.intersect` / `.difference` (insertion-order preserving). Combining
  chains of different type or different engine raises `TypeError`
  (cross-level mixing is loud, never a silent empty set):
  ```python
  fem.nodes.select(ids=a) | fem.nodes.select(ids=b)
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
- **Results terminal differs** — a results selection needs a
  component, so it ends in `.values(...)` (which forwards to the
  **retained** typed reader `results.<sub>.get(component=, ids=, pg=,
  label=, selection=)`), not `.result()`:
  ```python
  (results.nodes.select(pg="Base")
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
**retained by architecture** (the `.result()` payload and the viewer
pick-result type) — only the exports were dropped;
`EntitySelection.result()` returns the `Selection` payload (so
`.to_label()` / `.to_physical()` work, and they also work directly on
`EntitySelection`). Retained on the mesh side:
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
`from_physical(...)` **then** `select(name=...)`. (The one-step
`from_geometric` bridge was removed with no v2 successor — a
documented capability gap; use `to_physical` pre-mesh +
`from_physical` post-mesh instead.)

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

Full catalogue:

| Level | Method | Record type | Lives on |
|---|---|---|---|
| Node-to-node | `equal_dof`, `rigid_link`, `penalty` | `NodePairRecord` | `fem.nodes.constraints` |
| Node-to-group | `rigid_diaphragm`, `rigid_body`, `kinematic_coupling` | `NodeGroupRecord` | `fem.nodes.constraints` |
| Mixed-DOF | `node_to_surface` | `NodeToSurfaceRecord` (phantom nodes) | `fem.nodes.constraints` |
| Surface | `tie`, `distributing_coupling`, `embedded` | `InterpolationRecord` | `fem.elements.constraints` |
| Surface-to-surface | `tied_contact`, `mortar` | `SurfaceCouplingRecord` | `fem.elements.constraints` |

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
for nid, xyz in fem.nodes.constraints.phantom_nodes():
    ops.node(nid, *xyz)

# 2. Rigid links — grouped by master (all rigid kinds merged)
for master, slaves in fem.nodes.constraints.rigid_link_groups():
    for slave in slaves:
        ops.rigidLink("beam", master, slave)

# 3. Equal DOFs — flat pairs (direct + expanded node_to_surface)
for pair in fem.nodes.constraints.equal_dofs():
    ops.equalDOF(pair.master_node, pair.slave_node, *pair.dofs)

# 4. OR: rigid diaphragms via the native multi-slave command
for master, slaves in fem.nodes.constraints.rigid_diaphragms():
    ops.rigidDiaphragm(3, master, *slaves)

# 5. Surface constraints — interpolations (tie, distributing, embedded)
for interp in fem.elements.constraints.interpolations():
    # interp.slave_node, interp.master_nodes, interp.weights, interp.dofs
    ...
# Top-level couplings (mortar, tied_contact)
for coup in fem.elements.constraints.couplings():
    ...
```

For solvers that don't have grouped commands, the flat fallback still
works — `fem.nodes.constraints.pairs()` expands every record
(including `NodeGroupRecord` and `NodeToSurfaceRecord`) into individual
`NodePairRecord` objects, and you can kind-dispatch with
`K = fem.nodes.constraints.Kind`:

```python
K = fem.nodes.constraints.Kind
for c in fem.nodes.constraints.pairs():
    if c.kind == K.RIGID_BEAM:
        ops.rigidLink("beam", c.master_node, c.slave_node)
    elif c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
```

When you need fields that the flattened pairs can't expose (e.g.
`phantom_coords` on a compound record), reach for the raw accessor:

```python
for nts in fem.nodes.constraints.node_to_surfaces(): # NodeToSurfaceRecord
    ...
```

### 3.3  ⚠️ Constraint emission is DEFERRED in `apeSees`

There is **no** OpenSees-emission path for multi-point constraints
in the new bridge — `tie`, `rigid_link`, `equal_dof`,
`rigid_diaphragm`, `node_to_surface`, `tied_contact`, `mortar`,
embedded rebar. The `apeSees` `Emitter` protocol exposes only
`node / fix / mass / element / sp` — there is no `equalDOF`,
`rigidLink`, `rigidDiaphragm`, or `ASDEmbeddedNodeElement`. This is
deferred by design (ADR 0009 "FEM-direct, no ingest";
`src/apeGmsh/opensees/architecture/_DEFERRED.md`).

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
for load in fem.nodes.loads:
    load.node_id, load.force_xyz, load.moment_xyz   # inspect only
```

To put the same loads into OpenSees you **re-declare them
explicitly** on the bridge, inside a pattern:

```python
ts = ops.timeSeries.Linear()                 # also Constant/Path/Trig/Pulse
with ops.pattern.Plain(series=ts) as p:      # also UniformExcitation
    p.load(pg="Tip", forces=(0.0, -5e4, 0.0, 0.0, 0.0, 0.0))
    # p.load(node=<tag>, forces=(...))         # explicit node alternative
```

`p.load` fans a `pg=` across the group's nodes at build time.
Distributed/body loads have no pattern verb — gravity/body force
is an **element parameter** (`ops.element.<Solid>(pg=...,
body_force=(b1,b2,b3))`); 2-D solids take a `pressure=` arg.

### 4.2  Masses — accumulated per node

```python
g.masses.volume("Body", density=2400)
```

After resolution: `fem.nodes.masses` (persisted to `model.h5`;
inspection only — **not** auto-emitted by `apeSees`):

```python
for m in fem.nodes.masses:
    m.node_id, m.mass               # mass = (mx,my,mz,Ixx,Iyy,Izz)
print("Total:", fem.nodes.masses.total_mass())
```

Re-declare mass explicitly on the bridge:

```python
ops.mass(pg="Body", values=(mx, my, mz, 0.0, 0.0, 0.0))
# ops.mass(nodes=[...], values=(...))   # explicit-node alternative
```

### 4.3  Boundary conditions / SP — explicit on the bridge

`g.constraints` fixities and `g.loads.face_sp` resolve into the
snapshot but are **not** auto-emitted. Re-declare on `ops`:

```python
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))      # homogeneous SP
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.sp(pg="LoadingPin", dof=3, value=0.01)     # prescribed (non-zero) SP
```

---

## 5. OpenSees bridge — `apeSees(fem)`

OpenSees is driven **after** the session, from the `FEMData`
snapshot, with typed primitives (no string types). The bridge
does **not** ingest `g.loads` / `g.masses` / `g.constraints` —
loads / masses / SPs are re-declared explicitly here (§4);
multi-point constraints are deferred (§3.3).

```python
from apeGmsh.opensees import apeSees

fem = g.mesh.queries.get_fem_data(dim=3)
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)

conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
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
returns an immutable `BuiltModel`; it is usually implicit because
`ops.tcl/py/h5/run` build internally. Post-build inspection is
broker-side (`fem.inspect.*`) or via
`apeGmsh.opensees.emitter.h5_reader` on the emitted `model.h5`.

---

## 6. Viewer

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

---

## 7. Pitfalls & gotchas

### 7.1  `remove_duplicates` tolerance is unit-dependent
mm models: `tolerance=1e-3`. Metre models: `tolerance=1e-6`.

### 7.2  `set_size_sources(from_points=False)` after CAD import
CAD files bake tiny per-vertex `lc` values. Global sizing won't
override unless you disable point sources.

### 7.3  `renumber(dim=, base=1)` before `get_fem_data`
Gmsh tags are non-contiguous. OpenSees needs dense 1-based IDs.
Call `g.mesh.partitioning.renumber(dim=N, method="rcm", base=1)` —
`dim` is the element dimension you will later extract. `method="rcm"`
additionally minimises bandwidth for direct solvers.

### 7.4  Don't modify FEMData after extraction
FEMData is an immutable snapshot. Re-declare on session and re-extract.

### 7.5  `generate(dim=2)` on a solid is valid
Imports solid for BRep robustness, meshes only surfaces. Viewer
auto-skips unmeshed dimensions.

### 7.6  Don't hold tags past `fragment_all()`
OCC renumbers entities. Re-query via `g.parts.instances` or
`g.model.select(...)` after fragmentation.

### 7.7  Tie penalty default is 1e18
Drop to 1e10-1e12 if Newton fails to converge. The element only
needs K >> parent element stiffness.

### 7.8  `in_box` is half-open `[lo, hi)` (point family)
Every point-family `.select().in_box(...)` (and the retained
`filter_set(in_box=)`) excludes a coordinate / centroid lying exactly
on the **upper** box face by default (matches the results side;
adjacent boxes don't double-count). Pass `inclusive=True` for the
closed box `[lo, hi]`. The entity family (`g.model.select().in_box`)
has **no** `inclusive=` knob — passing it raises `TypeError` (use
`.on_plane(...)` / `.crossing_plane(spec, mode=...)` for an exact
predicate).

### 7.9  These selection paths fail loud (no silent wrong answer)
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

## 8. Anti-patterns

### 8.1  ❌ `equal_dof` for non-matching meshes → use `tie`
`equal_dof` needs co-located nodes. `tie` uses shape function
interpolation for non-matching interfaces.

### 8.2  ❌ `g.mesh.generate()` → use `g.mesh.generation.generate()`
No shortcut methods on parent composites. Sub-composite prefix required.

### 8.3  ❌ Skip `make_conformal` after STEP import
Touching bodies need `remove_duplicates` + `make_conformal` for
shared interfaces.

### 8.4  ❌ Store tags in local dicts → use `g.physical`
Physical groups are the only way to address mesh subsets by name across
the pipeline. Local dicts don't survive fragmentation.

### 8.5  ❌ `fem.node_ids` (old API) → use `fem.nodes.ids`
The FEMData API was rebuilt as composites. Old flat attributes no
longer exist. Use `fem.nodes.*` and `fem.elements.*`.

### 8.6  ❌ Hand-stitching spatial filters → use `.select()` chain
For a composed/spatial selection don't manually `np.isin` /
intersect results, and don't assume the entity-family
`g.model.select().in_box` matches a point-family box. Use the one
canonical chain (§1.4): `fem.nodes.select(pg=...).in_box(...)
.on_plane(...)`, set algebra with `| & - ^`. The old single-shot
`fem.*.get/.resolve` / `g.mesh_selection.add_nodes` accessors were
**removed** — `.select()` is the only accessor (its no-arg / `pg=`
seed covers the simple case too).

### 8.7  ❌ `results.elements.gauss.select(...)`
Not shipped yet. Results sub-composites (`gauss`/`fibers`/`layers`/
`line_stations`/`springs`) have **no** `.select()` — use their
existing `.in_box/.nearest_to/.on_plane` helpers.
(Mesh-selection name-seed, by contrast, **is** shipped:
`g.mesh_selection.select(name="my_set")` seeds id-for-id from an
existing set — see §1.4. Seeding directly from a raw gmsh PG name /
label is still not a `select()` parameter; use the two-step
`from_physical(...)` then `select(name=...)`. The one-step
`from_geometric` bridge was removed with no v2 successor.)

---

## 9. Quick reference

```python
from apeGmsh import apeGmsh, Part
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="model", verbose=True) as g:
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
    fem = g.mesh.queries.get_fem_data(dim=3)
    base = fem.nodes.select(pg="Base")          # -> MeshSelection
    ids, coords = base.ids, base.coords         # bulk arrays
    # canonical fluent selection (all 4 levels; same verbs everywhere):
    sel = (fem.nodes.select(pg="Body")
               .in_box((0, 0, 0), (1, 1, 1))   # half-open; inclusive=True → closed
               .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6))
    nodes = sel.result()                        # → NodeResult
    both = fem.nodes.select(ids=a) | fem.nodes.select(ids=b)   # set algebra
    faces = g.model.select("box", dim=2).in_box(lo, hi)        # entity family
    print(fem.inspect.summary())               # introspection

    # Viewer with FEM overlays (sees session loads/masses/constraints)
    g.mesh.viewer(fem=fem)

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
ops.tcl("model.tcl")
ops.py("model.py")
```

---

## 10. Lower-level skills

| Skill | Use when |
|---|---|
| `gmsh-structural` | Raw Gmsh API, meshing algorithms, OCC kernel, transfinite internals |
| `opensees-expert` | Constraint handlers, analysis commands, element internals, convergence |
| `fem-mechanics-expert` | FEM theory, shape functions, constitutive models, penalty vs Lagrange |
| `stko-to-python` | Loading STKO/MPCO HDF5 recorder output for post-processing |
