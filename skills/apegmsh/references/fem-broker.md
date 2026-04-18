# FEMData — the solver broker

`FEMData` is the solver-agnostic snapshot that apeGmsh hands to any
FEM backend.  Once you have `fem = g.mesh.queries.get_fem_data(...)`
you no longer need a live Gmsh session to query the mesh — the
snapshot is self-sufficient.

All signatures below are read from `src/apeGmsh/mesh/FEMData.py`.

## Construction

```python
# From a live session (usual path)
fem = g.mesh.queries.get_fem_data(dim=3)

# Direct factory (equivalent)
from apeGmsh import FEMData
fem = FEMData.from_gmsh(dim=3, session=g, ndf=6, remove_orphans=False)
fem = FEMData.from_gmsh(session=g)           # all dimensions

# From a standalone .msh file (no live session)
fem = FEMData.from_msh("bridge.msh", dim=2)
```

`from_gmsh` auto-resolves any pre-mesh declarations (`g.loads.*`,
`g.masses.*`, `g.constraints.*`) into resolved record sets attached
to the snapshot.  `ndf` controls the padding of load / mass vectors
(6 = full 3-D frame/shell, 3 = 3-D solid or 2-D frame, 2 = 2-D solid).

`remove_orphans=True` drops mesh nodes that aren't connected to any
element at the requested `dim`.  Nodes referenced by constraint,
load, or mass records are **always** kept, even with orphan removal
on — so you won't lose a master node just because the mesh didn't
land on it.

## Top-level layout

```
fem.nodes            → NodeComposite
fem.elements         → ElementComposite
fem.info             → MeshInfo
fem.inspect          → InspectComposite
fem.partitions       → list[int]      (shortcut for fem.nodes.partitions)
fem.mesh_selection   → MeshSelectionStore | None
```

`repr(fem)` prints the full `inspect.summary()`, so just printing the
object is a fast way to sanity-check what's in the snapshot.

## `fem.nodes` — NodeComposite

### Iteration / bulk access

```python
# Pair-iteration — the clean path for solver emission
for nid, xyz in fem.nodes.get():
    ops.node(nid, *xyz)

# Raw arrays
fem.nodes.ids       # ndarray(N,) dtype=object — iterates as plain int
fem.nodes.coords    # ndarray(N, 3) float64
len(fem.nodes)
```

IDs are stored with `dtype=object` on purpose so that iterating yields
plain Python `int` — OpenSees (and other C-extension solvers) reject
numpy integer scalars on some paths.  Take this at face value; do not
cast to int.

### Selection API

Every selection returns a `NodeResult` — iterable of `(id, xyz)`
pairs, with `.ids`, `.coords`, `.to_dataframe()` for array access.

```python
fem.nodes.get(target="Base")                  # label or PG or part
fem.nodes.get(pg="Base")                       # explicit PG
fem.nodes.get(label="control_node")            # explicit label
fem.nodes.get(tag=42)                          # raw Gmsh PG tag
fem.nodes.get(tag=(2, 17))                     # raw DimTag
fem.nodes.get(target=["Top", "Bottom"])        # union
fem.nodes.get(target="Body", partition=3)      # AND-intersected with partition
fem.nodes.get_ids(pg="Base")                   # IDs only
fem.nodes.get_coords(label="tip")              # coords only
```

Resolution order for `target=` (strings): **label → physical group →
part label**.  That matches the `LoadsComposite` auto-resolve so the
same name works everywhere.  A name that exists in none of the three
raises `KeyError` with the available candidates printed.

### Lookups

```python
fem.nodes.index(17)          # O(1) after first call — array index for a node ID
fem.nodes.partitions         # sorted list of partition IDs (empty if unpartitioned)
```

### Sub-composites on `fem.nodes`

```
fem.nodes.physical      → PhysicalGroupSet   (per-PG node/element slices)
fem.nodes.labels        → LabelSet           (per-label slices)
fem.nodes.constraints   → NodeConstraintSet  (equal_dof, rigid_beam, etc.)
fem.nodes.loads         → NodalLoadSet
fem.nodes.masses        → MassSet
fem.nodes.sp            → SPSet              (single-point prescribed)
```

Each record set iterates records, exposes `.patterns()` / `.by_pattern(name)`
where applicable, and `.summary()` for a DataFrame breakdown.

Node-constraint kinds live on `NodeConstraintSet.Kind`:

```python
K = fem.nodes.constraints.Kind
for c in fem.nodes.constraints.pairs():
    if c.kind == K.RIGID_BEAM:
        ops.rigidLink("beam", c.master_node, c.slave_node)
    elif c.kind == K.EQUAL_DOF:
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
```

## `fem.elements` — ElementComposite

### Iteration

Iterating `fem.elements` yields `ElementGroup` objects, one per
element type present in the mesh:

```python
for group in fem.elements:
    print(group.type_name, len(group), group.dim, group.npe)
    for eid, conn in group:
        ops.element(group.type_name, eid, *conn, mat_tag)
```

`len(fem.elements)` is the total element count across all groups.

### Raw access

```python
fem.elements.ids            # ndarray(E,) int64 — all element IDs concatenated
fem.elements.types          # list[ElementTypeInfo]
fem.elements.is_homogeneous # True if one element type
fem.elements.partitions     # sorted list of partition IDs

# Raises TypeError for multi-type meshes — use .resolve(element_type=...)
# or iterate groups instead
fem.elements.connectivity
```

### Selection API

```python
# General selection — same selectors as nodes, plus dim + element_type
result = fem.elements.get(
    target="col.web",           # or label=, pg=, tag=
    dim=2,                       # filter by element dimension
    element_type="tet4",         # alias, Gmsh code, or Gmsh name
    partition=0,                 # intersect with partition
)

# result is a GroupResult — iterable of ElementGroup
for group in result:
    ...

# Flatten when you know the result is a single type
ids, conn = result.resolve()                    # single type
ids, conn = result.resolve(element_type="tet4") # pick one when multiple

# Shortcut for flat (ids, conn) tuple
ids, conn = fem.elements.resolve(label="cols", element_type="beam2")
ids_only = fem.elements.get_ids(pg="Skin")
```

### Sub-composites on `fem.elements`

```
fem.elements.physical     → PhysicalGroupSet
fem.elements.labels       → LabelSet
fem.elements.constraints  → SurfaceConstraintSet  (tie interpolations, etc.)
fem.elements.loads        → ElementLoadSet        (beamUniform, surfacePressure)
```

### Type table

`fem.elements.type_table()` returns a DataFrame with one row per
element type (`code`, `name`, `gmsh_name`, `dim`, `order`, `npe`,
`count`) — useful before you start iterating to know what element
formulations you need to declare.

## `fem.info` — MeshInfo

```python
fem.info.n_nodes
fem.info.n_elems
fem.info.bandwidth          # semi-bandwidth max(max_node - min_node) per element
fem.info.types              # list[ElementTypeInfo]
fem.info.summary()          # "N nodes, M elements (tet4:4711), bandwidth=1234"
```

The bandwidth is computed from the raw connectivity as stored.  If
you want a small bandwidth you need to call
`g.mesh.partitioning.renumber(method="rcm")` **before**
`get_fem_data` — renumbering happens on the Gmsh side and the
snapshot just reads what Gmsh hands it.

## `fem.inspect` — InspectComposite

Cheap human-readable and table-oriented views.  Everything here is
computed on demand; the inspect composite stores no state.

```python
print(fem.inspect.summary())           # multi-line: types, PGs, labels, counts
fem.inspect.node_table()               # DataFrame of all nodes
fem.inspect.element_table()            # DataFrame with 'type' column
fem.inspect.physical_table()           # PG breakdown
fem.inspect.label_table()              # label breakdown
print(fem.inspect.constraint_summary())  # per-kind breakdown
print(fem.inspect.load_summary())        # per-pattern breakdown
print(fem.inspect.mass_summary())        # total mass + count
```

## Solver-agnostic emission pattern

Even without the OpenSees bridge you can feed `fem` into any backend
in a handful of lines:

```python
# Nodes
for nid, xyz in fem.nodes.get():
    solver.node(nid, xyz[0], xyz[1], xyz[2])

# Elements — one loop per element type
for group in fem.elements:
    for eid, conn in group:
        solver.element(group.type_name, eid, conn, mat_id)

# Supports
for nid in fem.nodes.get_ids(pg="Base"):
    solver.fix(nid, [1, 1, 1])

# Nodal loads (already resolved by pattern)
for pat in fem.nodes.loads.patterns():
    solver.begin_pattern(pat)
    for r in fem.nodes.loads.by_pattern(pat):
        solver.load(r.node_id, r.force_xyz, r.moment_xyz)
    solver.end_pattern()

# Node-pair constraints
K = fem.nodes.constraints.Kind
for c in fem.nodes.constraints.pairs():
    if c.kind == K.EQUAL_DOF:
        solver.equal_dof(c.master_node, c.slave_node, c.dofs)
```

The OpenSees bridge (`g.opensees.ingest.loads(fem).masses(fem)`)
is a thin wrapper over this same shape.

## Common traps

- `fem.elements.connectivity` raises `TypeError` when the mesh has
  more than one element type.  Use `fem.elements.resolve(element_type=...)`
  or iterate `for group in fem.elements:`.
- `fem.nodes.ids` has `dtype=object`; `fem.elements.ids` is
  `int64`.  This asymmetry is intentional — node IDs are consumed
  by Python solvers one at a time, element IDs are used in bulk.
- After `g.end()`, resolving a raw `(dim, tag)` DimTag through
  `get(target=(dim, tag))` raises `RuntimeError` because Gmsh is
  gone.  Use labels / PGs baked into the snapshot instead.
- `get_fem_data(dim=None)` returns every element present — you
  will see both `dim=2` surface elements (used by loads / tied
  contact / rigid diaphragms) and `dim=3` volume elements in the
  same snapshot.  Filter with `get(dim=3)` if you want the volume
  mesh only.
