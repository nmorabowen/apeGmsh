---
name: apegmsh-helper
description: Use whenever the user is working with apeGmsh — the structural-FEM wrapper around Gmsh with OpenSees integration. Triggers on building FEM models from CAD/STEP imports, Part-based assembly workflows, composite-based geometry/mesh/constraint APIs (g.model, g.mesh, g.physical, g.constraints, g.opensees, etc.), non-matching-mesh ties via ASDEmbeddedNodeElement, loads/masses/constraints resolution into the FEMData broker, and exporting models to OpenSees Tcl or openseespy scripts. Covers apeGmsh's own abstractions on top of Gmsh and OpenSees. For raw gmsh API questions see the gmsh-structural skill; for raw OpenSees analysis commands see opensees-expert; for FEM theory first principles see fem-mechanics-expert.
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

`g` is an **apeGmsh session** — one `gmsh.initialize()`, one model, and 16
composite objects as attributes. Each composite owns a focused slice of the
API. You never call `gmsh.*` directly.

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
| `g.opensees` | OpenSees bridge (materials, elements, ingest, inspect, export) |
| `g.loader` | `.msh` file loader — build a `FEMData` snapshot from a saved mesh |
| `g.inspect` | Session diagnostics |
| `g.plot` | Matplotlib (optional dep) |
| `g.view` | Gmsh post-processing views |

### 1.2  Sub-composites are required — no shortcuts

The three biggest composites have focused sub-namespaces:

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

**`g.opensees.*`** — solver bridge sub-composites:

| Sub-composite | Examples |
|---|---|
| `g.opensees.materials` | `add_nd_material`, `add_uni_material`, `add_section` |
| `g.opensees.elements` | `add_geom_transf`, `assign`, `fix` |
| `g.opensees.ingest` | `loads(fem)`, `masses(fem)`, `constraints(fem)` |
| `g.opensees.inspect` | `node_table`, `element_table`, `summary` |
| `g.opensees.export` | `tcl(path)`, `py(path)` |

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

**Grouped emission** (preferred over `pairs()` for OpenSees):
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

### 3.3  Tie via ASDEmbeddedNodeElement

```python
g.opensees.ingest.constraints(fem, tie_penalty=1e12)
```

apeGmsh emits ties as `ASDEmbeddedNodeElement` (penalty element).
Default K=1e18. Drop to 1e10-1e12 if conditioning issues arise.
Quad4 master faces are split into triangles automatically.

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

After get_fem_data(), loads split:
- **`fem.nodes.loads`** — `NodalLoadRecord` (point forces + moments)
- **`fem.elements.loads`** — `ElementLoadRecord` (pressure, body force)

```python
# 3D frame (ndf=6). Pick the slice matching your model's DOF space —
# apeGmsh stores spatial vectors only and does not know ndf.
for load in fem.nodes.loads:
    fx, fy, fz = load.force_xyz  or (0.0, 0.0, 0.0)
    mx, my, mz = load.moment_xyz or (0.0, 0.0, 0.0)
    ops.load(load.node_id, fx, fy, fz, mx, my, mz)
for eload in fem.elements.loads:
    ops.eleLoad(eload.element_id, eload.load_type, **eload.params)
```

### 4.2  Masses — accumulated per node

```python
g.masses.volume("Body", density=2400)
```

After resolution: `fem.nodes.masses`

```python
for m in fem.nodes.masses:
    ops.mass(m.node_id, *m.mass)  # mass = (mx,my,mz,Ixx,Iyy,Izz)
print("Total:", fem.nodes.masses.total_mass())
```

---

## 5. OpenSees pipeline

```python
g.opensees.set_model(ndm=3, ndf=3)
g.opensees.materials.add_nd_material(
    "Concrete", "ElasticIsotropic", E=30e9, nu=0.2, rho=2400)
g.opensees.elements.assign("Body", "FourNodeTetrahedron", material="Concrete")
g.opensees.elements.fix("Base", dofs=[1, 1, 1])

fem = g.mesh.queries.get_fem_data(dim=3)
(g.opensees.ingest
    .loads(fem)
    .masses(fem)
    .constraints(fem, tie_penalty=1e12))
g.opensees.build()
g.opensees.export.tcl("model.tcl").py("model.py")
```

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

    # OpenSees
    g.opensees.set_model(ndm=3, ndf=3)
    g.opensees.materials.add_nd_material("C", "ElasticIsotropic", E=30e9, nu=0.2)
    g.opensees.elements.assign("Body", "FourNodeTetrahedron", material="C")
    g.opensees.elements.fix("Base", dofs=[1, 1, 1])
    g.opensees.ingest.loads(fem).masses(fem).constraints(fem)
    g.opensees.build()
    g.opensees.export.tcl("model.tcl").py("model.py")

    # Viewer with FEM overlays
    g.mesh.viewer(fem=fem)
```

---

## 10. Lower-level skills

| Skill | Use when |
|---|---|
| `gmsh-structural` | Raw Gmsh API, meshing algorithms, OCC kernel, transfinite internals |
| `opensees-expert` | Constraint handlers, analysis commands, element internals, convergence |
| `fem-mechanics-expert` | FEM theory, shape functions, constitutive models, penalty vs Lagrange |
| `stko-to-python` | Loading STKO/MPCO HDF5 recorder output for post-processing |
