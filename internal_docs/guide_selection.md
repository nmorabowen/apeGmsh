# Selection in apeGmsh — OCC entities and mesh entities

apeGmsh has two complementary selection systems that sit on opposite sides of the meshing step. They intentionally look alike so that downstream code (constraints, loads, solver adapters) does not care which one a group came from — but they answer fundamentally different questions about the model.

| System | Lives on | Operates on | Created | Exposed on the broker |
|---|---|---|---|---|
| Geometric selection | `g.model.select(...)` | BRep / OCC entities — points, curves, surfaces, volumes | **Before** `g.mesh.generation.generate()` | *indirectly*, via `fem.nodes.physical` or `fem.mesh_selection` |
| Mesh selection | `g.mesh_selection` | Mesh nodes and elements | **After** `g.mesh.generation.generate()` | *directly*, via `fem.mesh_selection` |

The guiding idea: **OCC selection is geometry, mesh selection is topology**. One talks in terms of `(dim, tag)` BRep entries and is invariant to how you mesh. The other talks in terms of node IDs and element IDs and only becomes meaningful once a mesh exists.

Both systems feed the same FEM broker. This guide walks through both ends, then shows how to bridge them.


## 0. `.select()` — the unified fluent idiom (start here)

There is now **one canonical, daisy-chainable selection idiom**,
`.select()`, available at every level — geometry, the FEM broker,
results, and the live mesh:

```python
g.model.select("box", dim=2).in_box(lo, hi).on_plane(p, n, tol=1e-6)   # geometry
fem.nodes.select(pg="Base").in_box(lo, hi)                             # broker
results.nodes.select(pg="Base").in_box(lo, hi).values(component="displacement_z")
g.mesh_selection.select().in_box(lo, hi).on_plane(p, n, tol=1e-6)      # live mesh
```

Every `.select()` returns a chain with the same verbs
(`.in_box / .in_sphere / .on_plane / .crossing_plane / .nearest_to /
.where`), the same set algebra (`| & - ^` and
`.union / .intersect / .difference`), and a domain terminal —
`.result()` everywhere, except results which reads with
`.values(component=, time=, stage=)`. Name resolution inside
`.select()` is **never re-implemented**: it delegates to the same
contract-locked resolver, so the resolved selection is exactly what
the locked resolution contract returns.

> [!warning] `.select()` is the **only** selection surface (v2 removed the rest)
> Selection-unification v2 **hard-removed** the entire legacy surface:
> `g.model.selection` / `g.model.selection.select_*`
> (`SelectionComposite`), `g.model.queries.select(on=/crossing=)` /
> `queries.line` / `queries.select_all*`, `g.mesh_selection.add_nodes`
> / `add_elements` / `from_geometric`,
> `fem.nodes/elements.get/get_ids/get_coords/resolve`, and the chain
> `results.*.select(...).get(...)` terminal. There is **no shim** —
> calling them raises `AttributeError` / `ImportError`. Use `.select()`
> exclusively. The `core._selection.Selection` and `viz.Selection`
> *classes* are retained **by architecture** (the `.result()` payload
> and the viewer pick-result type respectively); only their package
> exports were dropped. Two capability gaps have no v2 successor — see
> §3.3 and §1.2. The maintainer invariants live in
> [The Selection Chain](guide_selection_chain.md).


## 1. Geometric selection — the OCC side

The entity-selection entry is `g.model.select(target, *, dim=)`. It is
a *query engine* over the currently synchronised OCC topology and
returns an `EntitySelection` — a daisy-chainable chain==terminal.
`.result()` yields the retained `Selection` payload (a frozen
`(dim, tag)` set with refinement / conversion helpers).

### 1.1 Seeding

`select()` takes a name (label / physical-group / part), an explicit
`(dim, tag)` set, or nothing-plus-`dim=` for *every* entity at a
dimension:

```python
faces   = g.model.select("TopFaces", dim=2)   # by PG / label / part
all_vol = g.model.select(dim=3)               # every volume
subset  = g.model.select([(1, 4), (1, 5)])    # explicit dimtags
```

Name resolution delegates verbatim to the contract-locked geometry
resolver (label → physical group → part); `select()` re-implements no
tier logic and adds no scoping of its own. The OCC kernel is
synchronised implicitly before querying.

### 1.2 The verb surface (and a capability gap)

`EntitySelection` composes spatial verbs, set-algebra, and direct
terminals:

- `.in_box(lo, hi)` — gmsh BRep containment (closed; `inclusive=`
  raises `TypeError` — it is point-family only)
- `.in_sphere(center, radius)` — bbox-centre distance test
- `.on_plane(point, normal, *, tol)` — all 8 bbox corners within `tol`
- `.crossing_plane(spec, *, tol, mode)` — `on` / `crossing` /
  `not_on` / `not_crossing` straddle (`spec` is an axis dict, a
  2/3-point list, or `m.model.queries.plane(...)`)
- `.nearest_to(point, n=)` / `.where(predicate)`
- set-algebra `| & - ^` and `.union / .intersect / .difference`
- terminals `.to_label(name)` / `.to_physical(name)` /
  `.to_dataframe()` / `.result()`

After `.result()` the `Selection` payload additionally offers the
position predicates `.select(on=/crossing=/not_on=/not_crossing=,
tol=)`, the direction filters `.parallel_to(...)` (curves) /
`.normal_along(...)` (surfaces), `.tags()`, and the same set-algebra.

> [!warning] Capability gap — the rich filter grammar has no successor
> The removed `SelectionComposite.select_*` exposed a rich
> keyword-filter grammar (`labels=` fnmatch, `kinds=`,
> `length/area/volume_range=`, `predicate=fn`, `exclude_tags=`,
> `physical=`, `at_point=`, `on_axis=`, `horizontal=`/`vertical=`/
> `aligned=`). `EntitySelection` has **no** equivalent — only the
> spatial verbs, set-ops, and `.to_*` terminals above. The retained
> `viz.Selection.filter()` carries that grammar but is
> **viewer-pick-only**, not a `g.model.select` migration path. For the
> common cases: use `dim=` seeding + spatial verbs; resolve a named
> physical group / label directly; or `.where(predicate)` /
> `.result().parallel_to(...)` for orientation.

```python
# Vertical column curves on the x = 0 wall, named as a physical group
(g.model.select("frame", dim=1).result()
    .parallel_to("z")
    .select(on={"x": 0})
    .to_physical("left_columns"))
```

### 1.3 Set algebra

Set algebra uses the normal Python operators on either the chain or
the `Selection` payload:

```python
all_curves = g.model.select(dim=1).result()
diagonals  = all_curves - columns - beams      # set difference
edges      = columns | beams                   # union
corners    = columns & g.model.select(dim=1).in_box(
    (0, 0, 0), (0.5, 0.5, 100)).result()
```

`Selection` subclasses `list`, so `|`/`&`/`-`/`^` (set-with-dedup),
not `+` (list concat), are the combining operators.

### 1.4 Topology queries

Cross-dimensional topology queries live on `g.model.queries` (the
removed `boundary_of`/`adjacent_to`/`closest_to` helpers have no
direct successor; use these instead):

```python
# Boundary entities — delegates to gmsh.model.getBoundary
faces_of_block = g.model.queries.boundary(block_vol_tag, dim=3)  # -> Selection

# Volumes adjacent to a set (share a boundary entity)
adj_vols = g.model.queries.adjacencies(some_vol_tag, dim=3)

# Nearest entities to a point: g.model.select(dim=N).nearest_to(p, n=)
pin_points = g.model.select(dim=0).nearest_to((0.0, 0.0, 10.0), n=4)
```

`g.model.queries.boundary` / `adjacencies` respect the OCC topology
instead of guessing from bounding boxes.

### 1.5 Persisting a geometric selection

An `EntitySelection` / `Selection` is just `(dim, tag)`s held in
Python. To make it survive meshing, register it — either as a **Tier-2
physical group** (carried into the msh/vtu output and visible to other
tools) or a **Tier-1 label** (`_label:`-prefixed, boolean-op-stable):

```python
g.model.select("Columns", dim=1).to_physical("columns")
g.model.select("Beams",   dim=1).to_label("beams")
g.model.select(dim=0).on_plane((0,0,0), (0,0,1), tol=1e-3).to_physical(
    "fixed_support")
```

`.to_physical` calls `g.physical.add(dim, tags, name=...)`;
`.to_label` calls `g.session.labels.add(...)`. Physical groups are the
*only* mechanism that carries named entity groupings through the
mesher into the msh/vtu outputs, so anything the solver must see
should be promoted before `g.mesh.generation.generate()`. (Tier-1 and
Tier-2 are **separate registries that are never merged** — ADR 0015.)


## 2. Mesh selection — the post-mesh side

Once `g.mesh.generation.generate(dim)` has run, the picture changes. The BRep entities are still there, but solvers need to talk about **node IDs and element IDs**. That is what `g.mesh_selection` is for.

`MeshSelectionSet` has the same identity contract as physical groups — a `(dim, tag)` key plus an optional `name` — but `dim` now means "dimensionality of the selected mesh entities":

- `dim=0` → node set
- `dim=1` → 1-D element set (line elements)
- `dim=2` → 2-D element set (tris / quads)
- `dim=3` → 3-D element set (tets / hexes)

### 2.1 Spatial queries on mesh entities

`g.mesh_selection.select(...)` returns a point-family `MeshSelection`;
the live-engine terminal `.save_as(name)` registers it as a named set
(the removed `add_nodes` / `add_elements` spatial registrars have no
direct equivalent — this is the replacement):

```python
g.mesh.generation.generate(3)

# Node sets — spatial verbs work on node coordinates
base = (g.mesh_selection.select()                       # node level
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-3)
    .save_as("base"))

interior = (g.mesh_selection.select()
    .in_box((-5, -5, 1), (5, 5, 10))
    .save_as("core_nodes"))

top5 = (g.mesh_selection.select()
    .nearest_to((0.0, 0.0, 10.0), n=5)
    .save_as("roof_monitor"))

# Element sets — verbs work on element centroids
slab = (g.mesh_selection.select(level="element", dim=2)
    .on_plane((0, 0, 10.0), (0, 0, 1), tol=1e-3)
    .save_as("slab_surf"))

core = (g.mesh_selection.select(level="element", dim=3)
    .in_box((-5, -5, 0), (5, 5, 10))
    .save_as("core_solid"))
```

`.save_as` returns the chain (so it stays fluent); the named set's tag
is auto-allocated per-dim, independent from physical-group tags.

> [!warning] point-family `in_box` is half-open (S2)
> The point-family `.in_box(lo, hi)` is **half-open on the upper side**
> by default (`xmin <= xyz < xmax` per axis), matching the `results`
> side. A coordinate exactly on an upper bound is **excluded** so
> adjacent boxes do not double-count a shared face. Pass
> `.in_box(lo, hi, inclusive=True)` for the closed `[lo, hi]` box (the
> retained `filter_set(..., inclusive=True)` does the same). See
> [MIGRATION_v1](MIGRATION_v1.md).

### 2.2 Explicit lists and predicates

If you already know the IDs — for example, from a solver query or from a post-processing pipeline — you can register them directly with the retained `add`:

```python
g.mesh_selection.add(dim=0, tags=[12, 18, 22, 41], name="instr_nodes")
g.mesh_selection.add(dim=2, tags=elem_id_list,      name="damage_zone")
```

For anything the spatial verbs do not cover, use `.where(predicate)`
on the chain:

```python
def above_z(coords):              # coords is (N, 3)
    return coords[:, 2] > 5.0

g.mesh_selection.select().where(above_z).save_as("upper_half")
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
g.model.select(dim=0).on_plane((0,0,0), (0,0,1), tol=1e-3).to_physical("base")
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

### 3.3 The one-step `from_geometric` bridge — REMOVED (capability gap)

> [!warning] No v2 successor
> `MeshSelectionSet.from_geometric(...)` (and its
> `Selection.to_mesh_nodes()` / `to_mesh_elements()` backing) — the
> one-step "geometric entity selection → named mesh selection without
> a physical group" bridge — was **removed** by selection-unification
> v2 along with `viz.Selection.to_mesh_*`. There is **no direct v2
> replacement** (a documented capability gap).

The remaining bridge is **`to_physical` + `from_physical`** (§3.1 →
§3.2): promote the geometric selection to a physical group pre-mesh,
then materialise that physical group as a mesh selection post-mesh:

```python
# Pre-mesh: geometric selection -> physical group
g.model.select(dim=2).on_plane((0,0,10.0), (0,0,1), tol=1e-3).to_physical(
    "roof_faces")

# Mesh
g.mesh.generation.generate(3)

# Post-mesh: physical group -> mesh selection
g.mesh_selection.from_physical(dim=2, name_or_tag="roof_faces",
                               ms_name="roof_nodes")
```

This costs one extra physical group versus the removed one-step path,
but it is the only supported route and also makes the group visible to
the msh/vtu output and other tools.


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
g.model.select(dim=2).on_plane((0,0,0),  (0,0,1), tol=1e-3).to_physical("base")
g.model.select(dim=2).on_plane((0,0,10), (0,0,1), tol=1e-3).to_physical("top")

# Geometric selection → physical group, pre-mesh (the from_geometric
# one-step bridge was removed — promote to a PG and pull it back below)
(g.model.select(dim=1).in_box((-0.1,-0.1,-0.1), (0.1,0.1,10.1)).result()
    .parallel_to("z")                       # vertical curves (no filter grammar)
    .to_physical("monitor_curves"))

# --- Meshing -------------------------------------------------------
g.mesh.generation.generate(3)

# --- Mesh selection (post-mesh) ------------------------------------
# Spatial query directly on mesh nodes
g.mesh_selection.select().in_sphere((5, 5, 5), 1.0).save_as("core_probe")

# Bridge the pre-mesh geometric selection in via its physical group
g.mesh_selection.from_physical(dim=1, name_or_tag="monitor_curves",
                               ms_name="monitor")

# Pull another physical group in as a mesh selection too, for a
# uniform downstream handle
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

**Choose the side that matches the question you are asking.** If the thing you want to refer to is a geometric feature of the CAD — a named face, a column line, an import label — start on the OCC side with `g.model.select(...)`. The queries survive remeshing, they are cheap to re-run, and promoting to a physical group gets them into the output file. If the thing you want to refer to only makes sense after discretisation — "the five nodes closest to the sensor", "all elements whose centroid lies inside this damage box" — start on the mesh side with `g.mesh_selection`.

**Let physical groups carry the named geometry across the mesher.** That is what they were designed for, and that is what the msh/vtu writers and every third-party post-processor expect. If your workflow exports the mesh to another tool, the groups must be physical groups.

**Use mesh selections as the apeGmsh-internal handle.** They are where spatial, topology-blind, or post-processing-derived queries live. They are also the only system that cleanly expresses set algebra over node and element IDs.

**Do not create both sides for the same concept unless you need to.** Either promote a geometric selection to a physical group (and let `fem.nodes.physical` carry it), or bridge it into a mesh selection (and let `fem.mesh_selection` carry it). Both directions are supported, but maintaining two mirror handles for the same concept is just extra book-keeping.

**The broker does not care which source you used.** `fem.nodes.physical` and `fem.mesh_selection` share the same query surface by design, so you can mix sources freely when writing a solver adapter. Downstream consumers — constraints, loads, solver adapters — should stay source-agnostic and take a `(dim, tag)` plus a "which store" reference, not hardcode one side.

Between the two systems you have a path for any grouping question: geometric and topological queries on the OCC side before meshing, spatial and ID-based queries on the mesh side after meshing, and an immutable broker at the end where both converge into a single solver-ready object.
