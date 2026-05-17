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

**Selection API** — get subsets by physical group or label:
```python
# NodeResult yields (id, xyz) pairs on iteration — ids are Python ints
for nid, xyz in fem.nodes.get(pg="Base"):
    ops.node(nid, *xyz)

# Attribute access for bulk arrays
result = fem.nodes.get(pg="Base")
result.ids              # ndarray (object dtype — yields Python int on iter)
result.coords           # ndarray (N, 3) float64
result.to_dataframe()   # pandas DataFrame
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
`g.model.selection` after fragmentation.

### 7.7  Tie penalty default is 1e18
Drop to 1e10-1e12 if Newton fails to converge. The element only
needs K >> parent element stiffness.

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
    ids, coords = fem.nodes.get(pg="Base")    # selection
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
