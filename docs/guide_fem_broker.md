# The FEM Broker — `FEMData`

## Why the broker exists

Gmsh is a wonderful mesher, but a live Gmsh session is a *stateful* thing: tags shift as you regenerate, physical groups live behind API calls, and every solver that wants to consume the mesh has to re-learn the same query dance. pyGmsh's FEM broker — the `FEMData` container returned by `g.mesh.get_fem_data(dim=...)` — breaks that coupling.

The broker is a **frozen snapshot** of everything a solver needs to build a model:

- Nodes (IDs and coordinates)
- Elements (IDs and connectivity)
- Physical groups (named sets of nodes and elements)
- Mesh-selection sets (the pyGmsh-side grouping primitive)
- Constraints, loads, and masses (already resolved to node-level records)
- Mesh statistics (counts and bandwidth)

Once you hold a `FEMData` object you can close Gmsh, pickle it, ship it to another process, or loop over it years later — no live session required. The broker is also deliberately **solver-agnostic**: OpenSees, Abaqus, Code_Aster, or a hand-rolled Python assembler all consume the same shapes.

## Nodes and coordinates

The broker exposes nodes as two parallel arrays:

```python
fem = g.mesh.get_fem_data(dim=3)

fem.node_ids      # ndarray(N,)    dtype=object, Python ints
fem.node_coords   # ndarray(N, 3)  dtype=float64, [x, y, z]
```

The `object` dtype on IDs is intentional: iterating yields plain Python `int`s, which OpenSees and other C-extension solvers accept without complaint. The coordinates are kept as a normal `float64` matrix because linear algebra on them is common.

Two access patterns are supported. For sequential emission to a solver, use the iterator:

```python
for nid, xyz in fem.nodes():
    ops.node(nid, *xyz)
```

For random access by node ID, the broker builds a cached `id -> index` map on first use:

```python
idx = fem.node_index(42)
fem.node_coords[idx]          # O(1) after first lookup
x, y, z = fem.get_node_coords(42)
```

The lookup maps are lazy, so if you only iterate you never pay for building them.

## Elements and connectivity

Elements mirror the node layout:

```python
fem.element_ids    # ndarray(E,)       dtype=object
fem.connectivity   # ndarray(E, npe)   dtype=object
```

`npe` is "nodes per element" and is a property of the mesh, not of the broker. The broker simply stores whatever Gmsh produced. For a linear tetrahedral mesh `npe == 4`; for a linear quadrilateral shell mesh `npe == 4`; for a linear hex mesh `npe == 8`. The connectivity references are expressed in terms of the same `node_ids` you see in `fem.node_ids`, so everything round-trips.

The element iterator hands back Python-native types:

```python
for eid, conn in fem.elements():
    ops.element("tri31", eid, *conn, thk, "PlaneStrain", 1)
```

And, symmetric to the node API:

```python
idx = fem.elem_index(10)
fem.connectivity[idx]             # -> [n1, n2, n3, ...]
n1, n2, n3 = fem.get_elem_connectivity(10)
```

### How this maps to higher-order meshes

Nothing in the broker is hard-coded to linear elements. When you ask Gmsh for a second-order mesh (`gmsh.model.mesh.setOrder(2)`), the only thing that changes on the broker side is the shape of `connectivity`: a `tri3` becomes a `tri6`, a `tet4` becomes a `tet10`, a `hex8` becomes a `hex20` (or `hex27` for full serendipity). The shape goes from `(E, 3)` to `(E, 6)`, from `(E, 4)` to `(E, 10)`, and so on.

Node ordering inside each row follows Gmsh's own convention — corner nodes first, then edge-midside nodes, then face nodes, then volume nodes. If your target solver uses a different ordering (e.g., OpenSees `tri6` expects a specific corner/edge interleave, Abaqus uses its own), you handle the permutation *at the solver adapter*, not inside the broker. This keeps `FEMData` truthful to the mesh and pushes solver quirks to the thin adapter layer where they belong.

The same applies to node coordinates: `node_coords` already contains the extra midside nodes that Gmsh inserted during `setOrder`, because they are just regular nodes in the Gmsh node list. No special handling, no separate "higher-order" code path — the broker is order-agnostic by construction.

## Physical groups and mesh-selection sets

The broker carries two parallel grouping sources:

```python
fem.physical         # PhysicalGroupSet — snapshots of Gmsh physical groups
fem.mesh_selection   # MeshSelectionStore — pyGmsh native selection sets
```

Both answer the same kinds of questions — "give me the node IDs of this set", "give me the element connectivity of that set" — and both produce the same dict shapes:

```python
nodes = fem.physical.get_nodes(dim=0, tag=1)
# {'tags': ndarray(N,), 'coords': ndarray(N, 3)}

elems = fem.physical.get_elements(dim=2, tag=3)
# {'element_ids': ndarray(E,), 'connectivity': ndarray(E, npe)}
```

Because the shapes are identical, downstream code (constraint handlers, load applicators, OpenSees adapters) does not care which source a group came from. This is the *source-agnostic* contract: add a new grouping primitive later and it slots in without touching any solver code.

## Constraints, loads, and masses

The broker is not just geometry — it is also where the resolved boundary-condition world lives. Three solver-ready snapshots hang off `FEMData`:

```python
fem.constraints   # ConstraintSet
fem.loads         # LoadSet
fem.masses          # MassSet
```

Each one is populated by the matching composite (`g.constraints`, `g.loads`, `g.masses`) calling `.resolve(...)` during FEM extraction. By the time the broker exists, every constraint has been expanded to node-level records, every load has been pinned to the nodes or elements it applies to, and every mass contribution has been accumulated per node.

### Constraints

`ConstraintSet` hides the difference between the high-level constraint types (equal-DOF, rigid link, node-to-surface, tie, mortar) behind two flat iterators:

```python
# Any phantom nodes the constraint machinery needs — emit first
for nid, xyz in fem.constraints.extra_nodes():
    ops.node(nid, *xyz)

# Every constraint as a node pair, regardless of original kind
for c in fem.constraints.node_pairs():
    if c.kind == "equal_dof":
        ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
    elif c.kind == "rigid_beam":
        ops.rigidLink("beam", c.master_node, c.slave_node)

# Interpolation-style constraints (tie, distributing, mortar slaves)
for ir in fem.constraints.interpolations():
    ...
```

A compound constraint — a node-to-surface rigid link, for example — expands automatically through `expand_to_pairs()` so the solver adapter only sees simple pairs.

### Loads

`LoadSet` groups records by pattern name, which maps naturally to solver load patterns:

```python
for pat in fem.loads.patterns():
    # open a pattern in the solver
    for rec in fem.loads.by_pattern(pat):
        ...

# Or split by record type
for r in fem.loads.nodal():
    ops.load(r.node_id, *r.values)

for r in fem.loads.element():
    ops.eleLoad("-ele", r.element_id, "-type", r.type, *r.values)
```

### Masses

Mass is not pattern-scoped — it is intrinsic to the model — so `MassSet` is a flat collection of per-node records. The composite already summed contributions from multiple mass definitions, so what you see is the final assembled mass at each node:

```python
for r in fem.masses.records():
    ops.mass(r.node_id, *r.mass)

print("Total translational mass:", fem.masses.total_mass())
```

`total_mass()` is a useful sanity check against the expected `Σ density × volume`.

## The payoff

What you get from this design is a single, quiet object that you can hand to any solver adapter and get a complete model out. The extraction step happens once; the adapter is a few dozen lines; and because every snapshot class (`PhysicalGroupSet`, `MeshSelectionStore`, `ConstraintSet`, `LoadSet`, `MassSet`) shares the same "dict of arrays" contract, you can mix and match grouping sources, pickle the broker between runs, and unit-test solver code without ever touching Gmsh.

That is the beauty of the FEM broker: it is the one place in the pipeline where the mesh stops being a live Gmsh conversation and becomes plain data — and plain data is the only thing that ever ports cleanly between tools.
