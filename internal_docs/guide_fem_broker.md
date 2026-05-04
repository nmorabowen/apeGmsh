# The FEM Broker — `FEMData`

## Why the broker exists

Gmsh is a wonderful mesher, but a live Gmsh session is a *stateful* thing: tags shift as you regenerate, physical groups live behind API calls, and every solver that wants to consume the mesh has to re-learn the same query dance. apeGmsh's FEM broker — the `FEMData` object returned by `g.mesh.queries.get_fem_data(dim=...)` — breaks that coupling.

The broker is a **frozen snapshot** of everything a solver needs to build a model. Once you hold a `FEMData` object you can close Gmsh, pickle it, ship it to another process, or loop over it years later — no live session required. The broker is deliberately **solver-agnostic**: OpenSees, Abaqus, Code_Aster, or a hand-rolled Python assembler all consume the same shapes.

## Architecture at a glance

The broker is organized by what the engineer needs — **Nodes** and **Elements**:

```
fem
  |-- .nodes              NodeComposite
  |     |-- .ids          ndarray(N,) object       all node IDs
  |     |-- .coords       ndarray(N, 3) float64    all coordinates
  |     |-- .physical     PhysicalGroupSet          solver-facing PGs
  |     |-- .labels       LabelSet                  geometry-time labels
  |     |-- .constraints  NodeConstraintSet         equal_dof, rigid, etc.
  |     |-- .loads        NodalLoadSet              point forces
  |     |-- .sp           SPSet                     prescribed displacements / fix
  |     +-- .masses       MassSet                   lumped nodal masses
  |
  |-- .elements           ElementComposite
  |     |-- .ids          ndarray(E,) object        all element IDs
  |     |-- .connectivity ndarray(E, npe) object    connectivity
  |     |-- .physical     PhysicalGroupSet          (shared ref with nodes)
  |     |-- .labels       LabelSet                  (shared ref with nodes)
  |     |-- .constraints  SurfaceConstraintSet      tie, mortar, etc.
  |     +-- .loads        ElementLoadSet             pressure, body force
  |
  |-- .info               MeshInfo                  n_nodes, n_elems, bandwidth
  +-- .inspect            InspectComposite          summaries, tables
```

Construction:

```python
fem = g.mesh.queries.get_fem_data(dim=3)       # from live session
fem = FEMData.from_gmsh(dim=3, session=g)      # equivalent
fem = FEMData.from_msh("bridge.msh", dim=2)    # from .msh file
```


## Nodes

The `fem.nodes` composite provides two ways to access node data:

**Raw arrays** for low-level consumers (VTK export, Results, custom code):

```python
fem.nodes.ids       # ndarray(N,)    dtype=object, yields Python ints
fem.nodes.coords    # ndarray(N, 3)  dtype=float64
```

The `object` dtype on IDs is intentional: iterating yields plain Python `int`s, which OpenSees and other C-extension solvers accept without complaint.

**Selection API** to get subsets by physical group or label:

```python
# Bundled — returns a NodeResult that yields (id, xyz) pairs on iteration
for nid, xyz in fem.nodes.get(pg="Base"):
    ops.node(nid, *xyz)                          # by physical group
for nid, xyz in fem.nodes.get(label="col.web"):  # by label
    ...
for nid, xyz in fem.nodes.get():                 # all domain nodes
    ...

# Individual arrays
ids    = fem.nodes.get_ids(pg="Base")
coords = fem.nodes.get_coords(pg="Base")

# Shorthand (searches PGs first, then labels)
for nid, xyz in fem.nodes.get("Base"):
    ...
```

The return type `NodeResult` iterates as `(id, xyz)` pairs and also exposes bulk
array attributes — pick whichever fits the call site:

```python
result = fem.nodes.get(pg="Base")
result.ids            # ndarray (object dtype — yields Python int on iter)
result.coords         # ndarray (N, 3) float64
result.to_dataframe() # pandas DataFrame with [node_id, x, y, z]
len(result)           # number of nodes
```

**Random access** by node ID:

```python
idx = fem.nodes.index(42)           # O(1) after first call
fem.nodes.coords[idx]               # -> [x, y, z]
```


## Elements

The `fem.elements` composite mirrors the node API:

**Raw arrays:**

```python
fem.elements.ids           # ndarray(E,)       dtype=object
fem.elements.connectivity  # ndarray(E, npe)   dtype=object
```

`npe` is "nodes per element" — a property of the mesh. For linear tets `npe == 4`; for linear quads `npe == 4`; for linear hexes `npe == 8`. The connectivity references match `fem.nodes.ids`.

**Selection API:**

```python
# .get() returns a GroupResult — a collection of per-type ElementGroups
result = fem.elements.get(pg="Body")
ids    = result.ids                  # all element IDs concatenated
conn   = result.connectivity         # only valid if homogeneous (single type)

# Flatten to (ids, conn) when you know the selection is single-type
ids, conn = fem.elements.resolve(pg="Body")          # by PG
ids, conn = fem.elements.resolve(label="col.web")    # by label
ids, conn = fem.elements.resolve()                   # all elements (single-type)

# Or pick one type from a mixed selection
ids, conn = fem.elements.resolve(pg="Body", element_type="tet4")

# Iterate per-type when the selection mixes types
for group in fem.elements.get(pg="Body"):
    for eid, conn_row in group:
        ops.element(group.element_type.name, eid, *conn_row, mat_tag)

# Individual
eids = fem.elements.get_ids(pg="Body")
```

`.get()` returns a `GroupResult` (`_element_types.py:232`); it is
**not** a NamedTuple and unpacking it via `ids, conn = ...` will
fail. Use `.resolve()` for the flat tuple form, or access `.ids` /
`.connectivity` attributes on the `GroupResult`. There is no
`get_connectivity(...)` shortcut — use
`fem.elements.resolve(...).` `[1]` or `fem.elements.get(...).connectivity`.

**Random access:**

```python
idx = fem.elements.index(10)
fem.elements.connectivity[idx]    # -> [n1, n2, n3, ...]
```


### Higher-order meshes

Nothing is hard-coded to linear elements. When you use `gmsh.model.mesh.setOrder(2)`, the connectivity shape changes (`tri3` becomes `tri6`, `tet4` becomes `tet10`). Node ordering follows Gmsh convention — corner nodes first, then edge midside, then face, then volume. Solver-specific permutation happens at the adapter layer, not in the broker.


## Named group sets — Physical Groups and Labels

Both composites carry references to the same two group sets:

```python
fem.nodes.physical      # PhysicalGroupSet — solver-facing PGs
fem.nodes.labels        # LabelSet — geometry-time labels (Tier 1)
```

Both inherit from `NamedGroupSet` and expose the same API:

```python
fem.nodes.physical.names()                 # ['Base', 'Body', ...]
fem.nodes.physical.node_ids("Base")        # ndarray(N,)
fem.nodes.physical.node_coords("Base")     # ndarray(N, 3)
fem.nodes.physical.element_ids("Body")     # ndarray(E,)
fem.nodes.physical.connectivity("Body")    # ndarray(E, npe)
fem.nodes.physical.summary()               # DataFrame

# Dict-like access
"Base" in fem.nodes.physical               # True/False
fem.nodes.physical["Base"]                 # info dict
```

Labels work identically via `fem.nodes.labels`.

**When to use which:**
- **Physical groups** are explicit solver-facing declarations (`g.physical.add_volume(..., name="Body")`)
- **Labels** are geometry-time internal names that survive boolean operations (`g.labels.add(...)`)
- The `.get(pg=..., label=...)` selection API on the composites lets you pick either without touching the group sets directly


## Constraints

Constraints are split across two composites based on what solver commands they produce:

**Node-level constraints** (`fem.nodes.constraints`):
equal_dof, rigid_beam, rigid_rod, rigid_diaphragm, node_to_surface

```python
K = fem.nodes.constraints.Kind    # linter-friendly constants

# Create phantom nodes first
for nid, xyz in fem.nodes.constraints.phantom_nodes():
    ops.node(nid, *xyz)

# Emit node-pair constraints (compound records expanded automatically)
for c in fem.nodes.constraints.pairs():
    if c.kind == K.RIGID_BEAM:
        ops.rigidLink("beam", c.master_node, c.slave_node)
    elif c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
```

**Surface-level constraints** (`fem.elements.constraints`):
tie, mortar, tied_contact, distributing

```python
K = fem.elements.constraints.Kind

for interp in fem.elements.constraints.interpolations():
    # build MP constraint from interpolation weights
    ...

for coupling in fem.elements.constraints.couplings():
    # mortar operator
    ...
```

**Kind constants** — no more magic strings:

```python
from apeGmsh.mesh import ConstraintKind

# Instead of: c.kind == "rigid_beam"     (typo-prone, no autocomplete)
# Use:        c.kind == K.RIGID_BEAM     (linter catches typos)
```

Compound constraints like `NodeToSurfaceRecord` (which creates phantom nodes + rigid links + equalDOF pairs) live in `fem.nodes.constraints` and are expanded automatically by `.pairs()` and `.phantom_nodes()`.


## Loads

Loads are split by type:

**Nodal loads** (`fem.nodes.loads`) — point forces on nodes:

```python
# 3D frame (ndf=6). Pick the slice matching your model's DOF space.
for load in fem.nodes.loads:
    fx, fy, fz = load.force_xyz  or (0.0, 0.0, 0.0)
    mx, my, mz = load.moment_xyz or (0.0, 0.0, 0.0)
    ops.load(load.node_id, fx, fy, fz, mx, my, mz)

# Pattern grouping
for pat in fem.nodes.loads.patterns():
    recs = fem.nodes.loads.by_pattern(pat)
    # open a solver pattern, emit records...
```

**Element loads** (`fem.elements.loads`) — surface pressure, body forces:

```python
for eload in fem.elements.loads:
    ops.eleLoad(eload.element_id, eload.load_type, **eload.params)
```


## Masses

Mass is intrinsic to the model — no pattern grouping. Lives under `fem.nodes.masses`:

```python
for m in fem.nodes.masses:
    ops.mass(m.node_id, *m.mass)

# Sanity check
print("Total mass:", fem.nodes.masses.total_mass())

# Lookup by node
m = fem.nodes.masses.by_node(42)
```


## Introspection

The `fem.inspect` composite provides summaries and tables:

```python
print(fem.inspect.summary())             # one-line stats + sub-composite counts

fem.inspect.node_table()                 # DataFrame: node_id, x, y, z
fem.inspect.element_table()              # DataFrame: elem_id, n0, n1, ...
fem.inspect.physical_table()             # DataFrame: dim, tag, name, n_nodes
fem.inspect.label_table()                # DataFrame: dim, tag, name, n_nodes

# Constraint/load/mass breakdowns with source tracing
print(fem.inspect.constraint_summary())
print(fem.inspect.load_summary())
print(fem.inspect.mass_summary())
```

The constraint summary tells you not just *what* you have, but *why* — which label, PG, or definition produced each record.


## Mesh statistics

```python
fem.info.n_nodes         # total nodes
fem.info.n_elems         # total elements
fem.info.bandwidth       # semi-bandwidth (max node spread per element)
fem.info.nodes_per_elem  # e.g. 4 for tet4
fem.info.elem_type_name  # e.g. "Tetrahedron 4"
fem.info.summary()       # one-line string
```


## Snapshot identity and persistence

The broker carries a content-addressed identifier and three persistence
entry points (`FEMData.py:1124, 1181, 1192, 1204`):

```python
fem.snapshot_id          # str — deterministic content hash, cached
```

`snapshot_id` is the linking contract for the Results module:
`Results.bind()` uses it to refuse pairing a results file against a
mesh it didn't come from. Computed on first access from
`_femdata_hash.compute_snapshot_id`.

```python
import h5py

# Embed this FEMData inside a results HDF5 group
with h5py.File("run.h5", "a") as f:
    fem.to_native_h5(f.require_group("model"))

# Round-trip through the same group later
with h5py.File("run.h5", "r") as f:
    fem2 = FEMData.from_native_h5(f["model"])
assert fem2.snapshot_id == fem.snapshot_id    # same hash
```

`from_native_h5` reconstructs nodes, elements (per type), physical
groups, and labels. Loads / masses / constraints are not round-tripped
because they don't influence `snapshot_id` and the viewer doesn't need
them.

```python
# Synthesize a partial FEMData from an MPCO MODEL/ group
with h5py.File("results.mpco", "r") as f:
    fem_mpco = FEMData.from_mpco_model(f["MODEL"])
```

`from_mpco_model` carries nodes, elements (keyed by negated OpenSees
class tag instead of Gmsh codes), and physical groups derived from
MPCO Regions. Labels and pre-mesh declarations are absent. The
`snapshot_id` will not match a native FEMData built from the same
mesh — that's expected.


## SP records (prescribed displacements)

`g.loads.face_sp(...)` produces `SPRecord` entries on `fem.nodes.sp`
(`FEMData.py:270`). The set behaves like the other record sub-composites:

```python
for rec in fem.nodes.sp.homogeneous():
    # rec.node_id, rec.dof   -- emit ops.fix(...)
    ...

for rec in fem.nodes.sp.prescribed():
    # rec.node_id, rec.dof, rec.value   -- emit ops.sp(...)
    ...
```

Ingest into the OpenSees bridge via `g.opensees.ingest.sp(fem)`.


## Selection shorthand and dim filter

`.get(...)` accepts a positional `target=` shorthand (auto-resolved
against PGs first, then labels, then parts) and a `dim=` filter to
narrow by entity dimension (`FEMData.py:300-309`):

```python
# Positional shorthand — searches PGs, then labels
result = fem.elements.get("Body")

# Restrict to a dimension when a name exists at multiple dims
result = fem.elements.get("interface", dim=2)    # only the 2-D entities

# Combine with element_type for a single-type slice
ids, conn = fem.elements.get("Body", dim=3).resolve(element_type="tet4")
```

A list passed to `target=` is interpreted as a **union** of targets.


## Complete solver workflow

```python
from apeGmsh import apeGmsh

g = apeGmsh("cantilever")
g.begin()

# Geometry + labels
g.model.geometry.add_box(0, 0, 0, 1, 1, 10, label="beam")
g.physical.add_surface([...], name="Base")
g.physical.add_volume([...], name="Body")

# Constraints, loads, masses (pre-mesh definitions)
g.constraints.equal_dof("Base", dofs=[1, 2, 3])
with g.loads.pattern("Gravity"):
    g.loads.body("Body", force_xyz=(0, 0, -9810))
g.masses.volume("Body", density=2400)

# Mesh
g.mesh.generation.generate(dim=3)
fem = g.mesh.queries.get_fem_data(dim=3)

# Build solver model
import openseespy.opensees as ops
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 3)

# 1. Domain nodes
for nid, xyz in fem.nodes.get():
    ops.node(nid, *xyz)

# 2. Phantom nodes from constraints
for nid, xyz in fem.nodes.constraints.phantom_nodes():
    ops.node(nid, *xyz)

# 3. Elements
for group in fem.elements.get():
    for eid, conn in group:
        ops.element("FourNodeTetrahedron", eid, *conn, mat_tag)
# Or if you know the mesh is single-type:
# ids, conn = fem.elements.resolve()
# for eid, row in zip(ids, conn):
#     ops.element("FourNodeTetrahedron", eid, *row, mat_tag)

# 4. Supports
for nid in fem.nodes.get_ids(pg="Base"):
    ops.fix(nid, 1, 1, 1)

# 5. Node constraints
K = fem.nodes.constraints.Kind
for c in fem.nodes.constraints.pairs():
    if c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)

# 6. Loads (3D frame, ndf=6)
ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
for load in fem.nodes.loads:
    fx, fy, fz = load.force_xyz  or (0.0, 0.0, 0.0)
    mx, my, mz = load.moment_xyz or (0.0, 0.0, 0.0)
    ops.load(load.node_id, fx, fy, fz, mx, my, mz)

# 7. Masses
for m in fem.nodes.masses:
    ops.mass(m.node_id, *m.mass)

# Introspection
print(fem.inspect.summary())
print(fem.inspect.constraint_summary())

g.end()
```

## The payoff

The broker is the one place in the pipeline where the mesh stops being a live Gmsh conversation and becomes plain data. It is organized by what you, the structural engineer, actually need: **nodes** and **elements**, with named selections and boundary conditions as sub-composites you can reach into. Kind constants prevent typos. Pair-iterating result objects give you clean one-liners plus array access and DataFrames. Introspection tells you not just what you have, but why.

Plain data is the only thing that ever ports cleanly between tools.
