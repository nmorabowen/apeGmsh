# Selection & geometric queries

This page teaches you how to *point at things* — how one fluent `.select()`
idiom turns coordinates and topology into names, on both sides of the mesher,
so the rest of your model never has to touch a raw tag.

The [mental model](mental-model.md) already gave you the naming layers: labels
at geometry time, physical groups for anything the solver must see. That covers
every entity you created yourself and thought to name. Selection covers the
rest — the top face of a box you didn't build face-by-face, every surface on a
plane, the five nodes nearest a sensor. You describe *where* something is, and
a terminal bakes the answer into a name. From then on it's just a name like any
other.

## One idiom, everywhere

There is exactly one selection idiom in apeGmsh, and it shows up at every level
of the workflow:

```python
g.model.select("box", dim=2).in_box(lo, hi).on_plane(p, n, tol=1e-6)   # geometry
fem.nodes.select(pg="Base").in_box(lo, hi)                             # broker
results.nodes.select(pg="Base").in_box(lo, hi).values(component="displacement_z")
g.mesh_selection.select().in_box(lo, hi).on_plane(p, n, tol=1e-6)      # live mesh
```

Every `.select()` starts a chain with the same shape: seed a starting set,
narrow it with spatial verbs — `.in_box`, `.in_sphere`, `.on_plane`,
`.crossing_plane`, `.nearest_to`, or an arbitrary `.where(predicate)` —
combine chains with set algebra (`| & - ^`, or the spelled-out
`.union / .intersect / .difference`), and finish with a terminal. The terminal
is `.result()` everywhere except results, which reads values directly with
`.values(component=, time=, stage=)`. Name seeding inside `.select()` uses the
same resolver as everything else in the library — the set you get is exactly
the set a load or constraint targeting that name would get.

Learn the chain once and you have learned it four times. What changes between
levels is only *what the atoms are* — CAD entities before meshing, node and
element IDs after — and that distinction is the real subject of this page.

## Selecting geometry

Before the mesh exists, `g.model.select(target, *, dim=)` queries the live OCC
topology. You can seed it three ways — with a name, with explicit
`(dim, tag)` pairs, or with nothing but a dimension to start from everything:

```python
faces   = g.model.select("TopFaces", dim=2)   # by PG / label / part
all_vol = g.model.select(dim=3)               # every volume
subset  = g.model.select([(1, 4), (1, 5)])    # explicit dimtags
```

The verbs then filter geometrically. `.in_box` keeps entities whose bounding
box is contained in yours; `.on_plane(point, normal, tol=)` keeps entities
whose whole bounding box sits within tolerance of a plane — the workhorse for
"the base of the model" or "the x = 0 wall"; `.crossing_plane` distinguishes
entities *on* a plane from those *straddling* it; `.nearest_to(point, n=)`
does proximity. A chain is cheap to build and cheap to re-run, and because it
queries the geometry rather than any mesh, it gives the same answer no matter
how — or whether — you mesh.

What makes a geometric selection *useful* is the terminal. `.to_label(name)`
registers it as a label; `.to_physical(name)` promotes it to a physical group.
Physical groups are the only naming that rides through the mesher into the
output files, so anything the solver or a downstream tool must see gets
promoted before you generate:

```python
g.model.select("Columns", dim=1).to_physical("columns")
g.model.select("Beams",   dim=1).to_label("beams")
g.model.select(dim=0).on_plane((0,0,0), (0,0,1), tol=1e-3).to_physical(
    "fixed_support")
```

That last line is the canonical move of the whole system: a spatial question
("which points sit on the ground plane?") becomes a name (`"fixed_support"`)
that [supports](../how-to/supports-bcs.md) and
[loads](../how-to/point-load.md) target without ever seeing a tag.

For questions the spatial verbs don't express, call `.result()` to get the
selection as a concrete payload. It carries the `(dim, tag)` set plus a few
refinement helpers — direction filters like `.parallel_to(...)` for curves and
`.normal_along(...)` for surfaces, position predicates, and the same
terminals — so orientation-flavoured picks stay one expression:

```python
# Vertical column curves on the x = 0 wall, named as a physical group
(g.model.select("frame", dim=1).result()
    .parallel_to("z")
    .select(on={"x": 0})
    .to_physical("left_columns"))
```

Set algebra works on chains and payloads alike, and it is often the cleanest
way to define a group *negatively* — everything that isn't already something
else:

```python
all_curves = g.model.select(dim=1).result()
diagonals  = all_curves - columns - beams      # set difference
edges      = columns | beams                   # union
```

## Selecting the mesh

Once `g.mesh.generation.generate(dim)` has run, a second kind of question
becomes meaningful: questions about *nodes and elements*. That is what
`g.mesh_selection` answers. Its `.select()` chain has the same verbs, but the
atoms are now node IDs (the default) or element IDs
(`level="element"`, filtered by centroid), and the terminal is
`.save_as(name)`, which registers the result as a named set:

```python
g.mesh.generation.generate(3)

base = (g.mesh_selection.select()                       # node level
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-3)
    .save_as("base"))

slab = (g.mesh_selection.select(level="element", dim=2)
    .on_plane((0, 0, 10.0), (0, 0, 1), tol=1e-3)
    .save_as("slab_surf"))
```

Here `dim` means the dimensionality of the mesh entities in the set — `dim=0`
is a node set, `dim=1` through `dim=3` are element sets — and each named set
gets its own tag, independent of physical-group tags.

`.where(predicate)` covers anything the built-in verbs don't; the predicate
receives the coordinate array and returns a boolean mask:

```python
def above_z(coords):              # coords is (N, 3)
    return coords[:, 2] > 5.0

g.mesh_selection.select().where(above_z).save_as("upper_half")
```

And when you already *have* the IDs — from a solver query, a post-processing
pipeline — you register them directly:

```python
g.mesh_selection.add(dim=0, tags=[12, 18, 22, 41], name="instr_nodes")
```

Existing sets can be refined (`filter_set`), sorted for deterministic
iteration (`sort_set`), and combined with the same set algebra — mesh
selections are the one place where union, intersection, and difference over
raw node and element IDs is expressed cleanly.

One deliberate asymmetry is worth knowing before it surprises you. On the mesh
side, `.in_box(lo, hi)` is **half-open** on the upper bound — a node exactly
on `hi` is excluded, so two adjacent boxes never double-count a shared face;
pass `inclusive=True` for the closed box. On the geometry side, `.in_box` is a
closed BRep containment test and *has no* `inclusive=` knob — passing one
raises `TypeError` rather than being silently ignored. Points can split a
boundary fairly; bounding boxes can't.

## Geometry or mesh?

The two sides are complementary, and choosing between them is easier than it
looks: **match the side to the question**. If the thing you mean is a feature
of the CAD — a face, a column line, an import label — select it on the
geometry side. Those queries survive remeshing and belong in the model
definition. If the thing you mean only exists after discretisation — "the
five nodes closest to the sensor", "every element whose centroid falls in this
damage box" — it *has* to be a mesh selection, because before meshing there
is nothing to select.

When one concept needs to cross from geometry to mesh, the route is the
physical group. Promote the geometric selection before meshing, then
materialise the group as a mesh selection after:

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

This is the only bridge, and it's a feature: the physical group it creates is
also what the msh/vtu writers and every third-party tool see, so the crossing
leaves a visible, exportable name behind. Don't create both handles for a
concept that only needs one — either the physical group carries it or the mesh
selection does.

## Topology queries

Selection reasons about *where* things are. Its complement, `g.model.queries`,
reasons about *how they connect* and *how big they are* — the OCC topology and
measures, no bounding-box guessing:

```python
# Boundary entities — the surfaces bounding a volume
faces_of_block = g.model.queries.boundary(block_vol_tag, dim=3)  # -> Selection

# Volumes adjacent to a set (share a boundary entity)
adj_vols = g.model.queries.adjacencies(some_vol_tag, dim=3)
```

`boundary` returns the lower-dimensional entities that bound an entity — the
natural way to find the faces of a volume you're about to constrain.
`adjacencies` finds the entities that share a boundary — which parts touch,
and whether a fragment produced the interfaces you expected. Alongside them
live the measurement queries — `bounding_box`, `center_of_mass`, and `mass`
(the geometric measure: volume, area, or length by dimension) — and
`registry()`, a DataFrame of every entity in the model with its type, bounds,
centroid, and measure. `boundary` returns a `Selection`, so a topology answer
feeds straight back into the selection machinery and out to a name.

## What the solver sees

Everything above converges at the snapshot. When you call
`g.mesh.queries.get_fem_data(...)`, both kinds of named group are frozen onto
the broker, side by side:

```python
fem = g.mesh.queries.get_fem_data(dim=3)

fem.nodes.physical   # PhysicalGroupSet  — snapshot of Gmsh physical groups
fem.mesh_selection   # MeshSelectionStore — snapshot of g.mesh_selection
```

The two stores expose the same query surface — `get_nodes(dim, tag)` returns
the same `{'tags': ..., 'coords': ...}` dict from either — so downstream code
never branches on where a group came from. A constraint handler, a load
applicator, a solver adapter: each takes a name, resolves it against either
store, and moves on. That uniformity is the payoff of the whole system, and it
is why idiomatic apeGmsh scripts read as a sequence of names — the coordinates
appear once, inside a `.select()` chain, and never again.

The full chain API — every verb, every terminal, every signature — is in the
[selection API reference](../api/selection.md).

---

*Next: [Parts & assembly](../internal_docs/guide_parts_assembly.md).*
