# Selection in apeGmsh — OCC entities and mesh entities

apeGmsh has two complementary selection systems that sit on opposite sides of the meshing step. They intentionally look alike so that downstream code (constraints, loads, solver adapters) does not care which one a group came from — but they answer fundamentally different questions about the model.

| System | Lives on | Operates on | Created | Exposed on the broker |
|---|---|---|---|---|
| Geometric selection | `g.model.selection` | BRep / OCC entities — points, curves, surfaces, volumes | **Before** `g.mesh.generation.generate()` | *indirectly*, via `fem.nodes.physical` or `fem.mesh_selection` |
| Mesh selection | `g.mesh_selection` | Mesh nodes and elements | **After** `g.mesh.generation.generate()` | *directly*, via `fem.mesh_selection` |

The guiding idea: **OCC selection is geometry, mesh selection is topology**. One talks in terms of `(dim, tag)` BRep entries and is invariant to how you mesh. The other talks in terms of node IDs and element IDs and only becomes meaningful once a mesh exists.

Both systems feed the same FEM broker. This guide walks through both ends, then shows how to bridge them.


## 1. Geometric selection — the OCC side

The geometric selection composite is attached to the model as `g.model.selection`. It is a *query engine* over the currently synchronised OCC topology. Every call returns a frozen `Selection` snapshot that behaves like a mathematical set with some refinement and conversion helpers attached.

### 1.1 The five query entry points

There is one method per BRep dimension plus an `all`:

```python
sel = g.model.selection

pts  = sel.select_points(**filters)      # dim = 0
crvs = sel.select_curves(**filters)      # dim = 1
srfs = sel.select_surfaces(**filters)    # dim = 2
vols = sel.select_volumes(**filters)     # dim = 3
any_ = sel.select_all(dim=-1, **filters) # across all dims
```

The composite implicitly calls `gmsh.model.occ.synchronize()` before querying, so you do not have to remember to sync yourself — but the geometry has to actually exist in the OCC kernel for the query to see it.

### 1.2 The filter vocabulary

All query methods take the same keyword-argument vocabulary. Filters are **AND-combined**, so passing several at once narrows the result.

**Identity filters**

- `tags=[...]` / `exclude_tags=[...]` — keep or drop by raw entity tag
- `labels="col_*"` — glob match against apeGmsh's registry labels (labels are the `name=...` you attach when creating geometry)
- `kinds="box"` — match against the kind recorded by the geometry factory (`"box"`, `"cylinder"`, `"disk"`, ...)
- `physical="fixed"` — members of a physical group by name or tag

**Spatial filters**

- `in_box=(x0, y0, z0, x1, y1, z1)` — delegates to `getEntitiesInBoundingBox`
- `in_sphere=(cx, cy, cz, r)` — centroid distance test
- `on_plane=("z", 0.0, 1e-6)` — entity's bounding box touches an axis-aligned plane
- `on_axis=("z", 1e-6)` — entity centroid lies on a coordinate axis
- `at_point=(x, y, z, atol)` — entity bounding box contains the point

**Metric ranges**

- `length_range=(lo, hi)` — dim=1 only, uses `occ.getMass`
- `area_range=(lo, hi)` — dim=2 only
- `volume_range=(lo, hi)` — dim=3 only

**Orientation (curves only)**

- `horizontal=True` — curve lies in a plane normal to z
- `vertical=True` — curve is parallel to z
- `aligned=("x", 5.0)` — parallel to the given axis within `atol_deg`

**Escape hatch**

- `predicate=lambda dim, tag: ...` — arbitrary Python predicate evaluated last

A typical use after importing an IGES frame:

```python
sel = g.model.selection

columns = sel.select_curves(vertical=True)
beams   = sel.select_curves(horizontal=True, length_range=(0.5, 12.0))
base    = sel.select_points(on_plane=("z", 0.0, 1e-3))

print(columns)   # <Selection dim=1 n=24 [C1, C2, …]>
```

### 1.3 Working with a `Selection`

A `Selection` is an immutable snapshot but supports rich manipulation. Set algebra uses the normal Python operators:

```python
all_curves = sel.select_curves()
diagonals  = all_curves - columns - beams      # set difference
edges      = columns | beams                   # union
corners    = columns & sel.select_curves(in_box=(0, 0, 0, 0.5, 0.5, 100))
```

Refinement re-applies filters to an existing set without re-running the full universe query:

```python
short_cols = columns.filter(length_range=(0, 4.0))
leftmost   = columns.sorted_by("x").limit(5)
```

Geometry helpers work on the whole set:

```python
columns.bbox()      # (xmin, ymin, zmin, xmax, ymax, zmax)
columns.centers()   # ndarray(N, 3) — entity centroids
columns.masses()    # ndarray(N,)  — length/area/volume per entity
```

### 1.4 Topology helpers

Some queries do not fit the filter vocabulary because they cross dimensions. For those, the composite exposes three helpers:

```python
# Boundary of a set — delegates to gmsh.model.getBoundary
faces_of_block = sel.boundary_of(block_volumes)

# Entities of dim_target whose boundary touches the source set
bounding_volumes = sel.adjacent_to(some_surfaces, dim_target=3)

# Nearest-n entities to a point
pin_points = sel.closest_to(0.0, 0.0, 10.0, dim=0, n=4)
```

`boundary_of` and `adjacent_to` are the right way to get "the surfaces of this block" or "the volumes that share this interface surface" — they respect the OCC topology instead of guessing from bounding boxes.

### 1.5 Persisting a geometric selection

A geometric `Selection` by itself is just a tuple of `DimTag`s held in Python. To make it survive outside the current selection session, promote it to a **physical group**:

```python
columns.to_physical("columns")
beams.to_physical("beams", tag=101)
base.to_physical("fixed_support")
```

This calls `g.physical.add(dim, tags, name=...)` for you and returns the physical-group tag. Physical groups are the *only* mechanism that carries named entity groupings through the mesher and into the msh/vtu outputs, so any selection you want the solver to see should be promoted before `g.mesh.generation.generate()`.


## 2. Mesh selection — the post-mesh side

Once `g.mesh.generation.generate(dim)` has run, the picture changes. The BRep entities are still there, but solvers need to talk about **node IDs and element IDs**. That is what `g.mesh_selection` is for.

`MeshSelectionSet` has the same identity contract as physical groups — a `(dim, tag)` key plus an optional `name` — but `dim` now means "dimensionality of the selected mesh entities":

- `dim=0` → node set
- `dim=1` → 1-D element set (line elements)
- `dim=2` → 2-D element set (tris / quads)
- `dim=3` → 3-D element set (tets / hexes)

### 2.1 Spatial queries on mesh entities

The two creation methods mirror the OCC side but work on mesh data:

```python
g.mesh.generation.generate(3)

# Node sets — filters work on node coordinates
base = g.mesh_selection.add_nodes(
    on_plane = ("z", 0.0, 1e-3),
    name     = "base",
)

interior = g.mesh_selection.add_nodes(
    in_box   = (-5, -5, 1, 5, 5, 10),
    name     = "core_nodes",
)

top5 = g.mesh_selection.add_nodes(
    closest_to = (0.0, 0.0, 10.0),
    count      = 5,
    name       = "roof_monitor",
)

# Element sets — filters work on element centroids (in_box) or
# on "all nodes lie on plane" (on_plane)
slab = g.mesh_selection.add_elements(
    dim      = 2,
    on_plane = ("z", 10.0, 1e-3),
    name     = "slab_surf",
)

core = g.mesh_selection.add_elements(
    dim    = 3,
    in_box = (-5, -5, 0, 5, 5, 10),
    name   = "core_solid",
)
```

Each creation call returns the allocated tag. Tags are auto-allocated per-dim and independent from physical-group tags.

### 2.2 Explicit lists and predicates

If you already know the IDs — for example, from a solver query or from a post-processing pipeline — you can register them directly:

```python
g.mesh_selection.add(dim=0, tags=[12, 18, 22, 41], name="instr_nodes")
g.mesh_selection.add(dim=2, tags=elem_id_list,      name="damage_zone")
```

For anything the built-in filters do not cover, pass a `predicate`:

```python
def above_z(coords):              # coords is (N, 3)
    return coords[:, 2] > 5.0

g.mesh_selection.add_nodes(predicate=above_z, name="upper_half")
```

### 2.3 Refining and combining sets

Once a set exists you can refine it into a new set, sort its entries in place, or combine sets with set algebra:

```python
# Refine base -> keep only the rightmost corner
rhs_base = g.mesh_selection.filter_set(
    dim=0, tag=base,
    in_box=(4.9, -5, -0.1, 5.1, 5, 0.1),
    name="rhs_base",
)

# Sort entries along x for deterministic iteration
g.mesh_selection.sort_set(dim=0, tag=base, by="x")

# Set algebra (unions, intersections, differences)
corner = g.mesh_selection.intersection(0, base, rhs_base, name="rhs_base_corner")
```

### 2.4 Introspection

```python
g.mesh_selection.summary()                     # DataFrame of every set
g.mesh_selection.get_all(dim=0)                # [(0, 1), (0, 2), ...]
g.mesh_selection.get_tag(0, "base")            # 1
g.mesh_selection.get_nodes(0, 1)               # {'tags': ..., 'coords': ...}
g.mesh_selection.get_elements(2, 3)            # {'element_ids': ..., 'connectivity': ...}
g.mesh_selection.to_dataframe(0, 1)            # per-entry DataFrame
```

The `get_nodes` / `get_elements` return shapes are **identical** to the ones returned by `g.physical`; that is the whole point — downstream code never has to care which source a group came from.


## 3. Bridging the two systems

Geometric and mesh selections are complementary, not alternatives. Real workflows use both, and apeGmsh gives you three explicit bridges.

### 3.1 `Selection.to_physical(...)` — geometric → physical group

The simplest bridge. It writes an OCC selection into Gmsh's physical-group table so the mesher sees it and the msh/vtu output carries it. This is the standard way to make a geometric selection persist across meshing:

```python
g.model.selection.select_points(on_plane=("z", 0, 1e-3)).to_physical("base")
g.mesh.generation.generate(3)
# Now fem.nodes.physical will contain 'base'
```

### 3.2 `MeshSelectionSet.from_physical(...)` — physical group → mesh selection

The reverse direction — take an existing physical group and materialise it as a mesh selection of node IDs:

```python
g.mesh.generation.generate(3)
g.mesh_selection.from_physical(dim=0, name_or_tag="base", ms_name="base_nodes")
```

Why would you want this? Because a physical group lives in Gmsh's world — it is still geometry-flavoured. Converting it to a mesh selection gives you the cached node array and makes it addressable uniformly next to your other mesh selections.

### 3.3 `MeshSelectionSet.from_geometric(...)` — the one-step bridge

This is the workhorse bridge. It takes a geometric `Selection` (pre-mesh) and builds a mesh selection out of it **after the mesh has been generated**, without requiring you to create a physical group first:

```python
# Pre-mesh: build the geometric query
top_faces = g.model.selection.select_surfaces(on_plane=("z", 10.0, 1e-3))

# Mesh
g.mesh.generation.generate(3)

# Post-mesh: extract the corresponding node set
g.mesh_selection.from_geometric(
    top_faces,
    kind="nodes",              # or "elements"
    name="roof_nodes",
)
```

Under the hood this calls `top_faces.to_mesh_nodes()` (or `.to_mesh_elements()`), which uses `gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)` on each entity in the selection and deduplicates.

Two rules of thumb:

- Use **`to_physical` + `from_physical`** when you want the group to appear in the msh/vtu output or be visible to another tool.
- Use **`from_geometric`** when the group is purely an internal handle and you just want the mesh IDs on the apeGmsh side.


## 4. Selection on the FEM broker

When you call `g.mesh.queries.get_fem_data(dim=...)`, apeGmsh captures a frozen snapshot of the current state: nodes, elements, physical groups, mesh selections, constraints, loads, and masses. The broker then becomes the single object you hand to a solver adapter.

Selections show up on the broker under two mirror accessors with the same API:

```python
fem = g.mesh.queries.get_fem_data(dim=3)

fem.nodes.physical   # PhysicalGroupSet  — snapshot of Gmsh physical groups
fem.mesh_selection   # MeshSelectionStore — snapshot of g.mesh_selection
```

Both are immutable and both expose the same query methods. The broker-side classes live in `apeGmsh.mesh.FEMData.PhysicalGroupSet` and `apeGmsh.mesh.MeshSelectionSet.MeshSelectionStore`, and they share this contract (note: physical groups are accessed via `fem.nodes.physical`):

```python
store.get_all(dim=-1)                 # list of (dim, tag)
store.get_name(dim, tag)              # "base"
store.get_tag(dim, "base")            # 1
store.summary()                       # DataFrame
store.get_nodes(dim, tag)             # {'tags': ndarray, 'coords': ndarray(N,3)}
store.get_elements(dim, tag)          # {'element_ids': ndarray, 'connectivity': ndarray(E,npe)}
```

This is the `FEMData` source-agnostic contract in action: a constraint handler or load applicator can receive either a `fem.nodes.physical` key or a `fem.mesh_selection` key and use exactly the same code to resolve it to node or element IDs.

### 4.1 A complete round-trip

Here is what the two sides look like in one session, so you can see where everything lands on the broker:

```python
from apeGmsh import apeGmsh

g = apeGmsh(model_name="selection_demo", verbose=True)

# --- Geometry (pre-mesh) -------------------------------------------
g.model.geometry.add_box(0, 0, 0, 10, 10, 10, label="blk")
g.model.sync()

# Geometric selection → physical group (survives meshing)
g.model.selection.select_surfaces(on_plane=("z", 0, 1e-3)).to_physical("base")
g.model.selection.select_surfaces(on_plane=("z", 10, 1e-3)).to_physical("top")

# Geometric selection → mesh selection bridge (prepared, not yet run)
monitor_curves = g.model.selection.select_curves(
    vertical=True,
    in_box=(-0.1, -0.1, -0.1, 0.1, 0.1, 10.1),
)

# --- Meshing -------------------------------------------------------
g.mesh.generation.generate(3)

# --- Mesh selection (post-mesh) ------------------------------------
# Spatial query directly on mesh nodes
g.mesh_selection.add_nodes(
    in_sphere=(5, 5, 5, 1.0),
    name="core_probe",
)

# Bridge the pre-mesh geometric selection into the mesh world
g.mesh_selection.from_geometric(monitor_curves, kind="nodes", name="monitor")

# Pull a physical group in as a mesh selection too, if you need a
# uniform handle for downstream code
g.mesh_selection.from_physical(dim=2, name_or_tag="top", ms_name="top_nodes")

# --- FEM broker ----------------------------------------------------
fem = g.mesh.queries.get_fem_data(dim=3)

# Physical groups from Gmsh
fem.nodes.physical.summary()
base_nodes = fem.nodes.physical.get_nodes(dim=2, tag=fem.nodes.physical.get_tag(2, "base"))

# Mesh selections from apeGmsh
fem.mesh_selection.summary()
monitor = fem.mesh_selection.get_nodes(0, fem.mesh_selection.get_tag(0, "monitor"))
core    = fem.mesh_selection.get_nodes(0, fem.mesh_selection.get_tag(0, "core_probe"))

# Same dict shape from either side
for nid, xyz in zip(monitor["tags"], monitor["coords"]):
    print(nid, xyz)
```

Notice that once you are on the broker, the code never branches on where a group came from. It reaches into `fem.nodes.physical` or `fem.mesh_selection` with the same call signature and gets back the same `{'tags': ..., 'coords': ...}` dict. That uniformity is what lets solver adapters stay short.


## 5. Mental model and rules of thumb

It is worth stepping back from the API and keeping a few principles in mind.

**Choose the side that matches the question you are asking.** If the thing you want to refer to is a geometric feature of the CAD — a named face, a column line, an import label — start on the OCC side with `g.model.selection`. The queries survive remeshing, they are cheap to re-run, and promoting to a physical group gets them into the output file. If the thing you want to refer to only makes sense after discretisation — "the five nodes closest to the sensor", "all elements whose centroid lies inside this damage box" — start on the mesh side with `g.mesh_selection`.

**Let physical groups carry the named geometry across the mesher.** That is what they were designed for, and that is what the msh/vtu writers and every third-party post-processor expect. If your workflow exports the mesh to another tool, the groups must be physical groups.

**Use mesh selections as the apeGmsh-internal handle.** They are where spatial, topology-blind, or post-processing-derived queries live. They are also the only system that cleanly expresses set algebra over node and element IDs.

**Do not create both sides for the same concept unless you need to.** Either promote a geometric selection to a physical group (and let `fem.nodes.physical` carry it), or bridge it into a mesh selection (and let `fem.mesh_selection` carry it). Both directions are supported, but maintaining two mirror handles for the same concept is just extra book-keeping.

**The broker does not care which source you used.** `fem.nodes.physical` and `fem.mesh_selection` share the same query surface by design, so you can mix sources freely when writing a solver adapter. Downstream consumers — constraints, loads, solver adapters — should stay source-agnostic and take a `(dim, tag)` plus a "which store" reference, not hardcode one side.

Between the two systems you have a path for any grouping question: geometric and topological queries on the OCC side before meshing, spatial and ID-based queries on the mesh side after meshing, and an immutable broker at the end where both converge into a single solver-ready object.
