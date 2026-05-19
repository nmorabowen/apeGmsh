# Model — `g.model`

OCC geometry composite. Five focused sub-composites: geometry, boolean,
transforms, io, queries.

## `g.model`

::: apeGmsh.core.Model.Model

## Sub-composites

### `g.model.geometry`

::: apeGmsh.core._model_geometry._Geometry

### `g.model.boolean`

::: apeGmsh.core._model_boolean._Boolean

### `g.model.transforms`

::: apeGmsh.core._model_transforms._Transforms

### `g.model.io`

::: apeGmsh.core._model_io._IO

### `g.model.queries`

::: apeGmsh.core._model_queries._Queries

### Fluent selection — `g.model.select()`

`g.model.select(...)` is the **single** entity-selection surface and
the geometry entry of the unified, daisy-chainable
[selection idiom](selection.md). The former
`g.model.queries.select(...)` predicate selector and the former
`g.model.selection.select_*` entity composite have been **removed**;
their behaviour is folded into the verbs below. `select()` returns an
[`EntitySelection`][apeGmsh.core._selection.EntitySelection] (entity
family) with direct terminals `.to_label()` / `.to_physical()` /
`.to_dataframe()`; `.result()` is a zero-cost identity alias yielding
the [`Selection`][apeGmsh.core._selection.Selection] payload (retained
by architecture as the entity-side terminal type).

```python
(g.model.select("Faces")                          # tiered name resolve
    .in_box((-0.1, -0.1, -0.1), (1.1, 1.1, 1.1))  # gmsh BRep containment
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    .to_physical("Base"))
```

Entity-family `in_box` is gmsh BRep containment and rejects
`inclusive=` with a `TypeError` (it is point-family only). See
[Selection](selection.md) for the full idiom, the verb surface, and
the point-vs-entity family contract.

### Geometric predicates — cheat sheet

The straddle predicates are reached two ways: as the
`.crossing_plane(spec, *, mode=)` verb on the
[`EntitySelection`][apeGmsh.core._selection.EntitySelection] chain, or
as the `.select(on=/crossing=/not_on=/not_crossing=)` refinement method
on the [`Selection`][apeGmsh.core._selection.Selection] payload (after
`.result()`):

| Predicate | Where | Dim | Example | Keeps entities that… |
|---|---|---|---|---|
| `mode="on"` / `on=` | `.crossing_plane()` verb / `.select()` kwarg | any | `on={"z": 0}` | lie **entirely on** the plane |
| `mode="crossing"` / `crossing=` | `.crossing_plane()` verb / `.select()` kwarg | any | `crossing={"z": 0}` | **straddle** the plane |
| `mode="not_on"` / `not_on=` | `.crossing_plane()` verb / `.select()` kwarg | any | `not_on={"z": 0}` | are **not entirely on** the plane |
| `mode="not_crossing"` / `not_crossing=` | `.crossing_plane()` verb / `.select()` kwarg | any | `not_crossing={"z": 0}` | lie **entirely on one side** |
| `.parallel_to(...)` | method on `Selection` | 1 (curves) | `edges.parallel_to("z")` | are curves whose chord direction is **parallel** to it |
| `.normal_along(...)` | method on `Selection` | 2 (surfaces) | `faces.normal_along("z")` | are surfaces whose **normal** is along it |

#### Primitive (plane / line) spec formats

| Form | Meaning |
|---|---|
| `{"z": 0}` / `{"x": 5}` / `{"y": -3}` | Axis-aligned plane |
| `[(x1,y1,z1), (x2,y2,z2)]` | Infinite line through 2 points (for curves in 2-D) |
| `[(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]` | Infinite plane through 3 points (for surfaces / volumes) |
| `m.model.queries.plane(...)` | `Plane` object — axis-aligned, 3-point, or `normal=`/`through=` |

#### Direction formats accepted by `.parallel_to(...)` / `.normal_along(...)`

| Form | Meaning |
|---|---|
| `"x"`, `"y"`, `"z"` | Axis alias |
| `(1, 0, 0)` / `(1, 1, 0)` | Any non-zero 3-vector (normalized internally) |
| `angle_tol=2.0` | Tolerance in **degrees**; default `1.0`. Anti-parallel counts as parallel. |

#### Seeding a selection

| Call | Returns |
|---|---|
| `g.model.select(dim=N)` | every entity at dimension `N` (point=0, curve=1, surface=2, volume=3) |
| `g.model.select(name_or_dimtags, dim=N)` | by PG / label / part name, or from an explicit `(dim, tag)` set |

### Selection — `.result()` payload of `select()`

`g.model.select(...)` returns an `EntitySelection`; its `.result()`
yields a `Selection` — a chainable list of `(dim, tag)` pairs. No
import is needed. The `Selection` payload still carries the position
predicates (`.select(on=/crossing=/not_on=/not_crossing=, tol=)`),
direction filters, and set-algebra.

```python
curves = m.model.queries.boundary(surf, oriented=False)  # -> Selection

# axis-aligned plane
bottom = curves.select(on={'y': 0})

# 2-point line
mid    = curves.select(crossing=[(0,5,0),(5,5,0)])

# chain to narrow further
left_bottom = curves.select(on={'y': 0}).select(on={'x': 0})

# extract bare tags for downstream calls
m.mesh.structured.set_transfinite_curve(bottom.tags(), n=11)
```

#### Starting from every entity of a dimension

When parsing an imported `.geo` / STEP file with no labels yet, seed
with `g.model.select(dim=N)` (no target):

```python
# Every volume in the model
g.model.select(dim=3).to_physical("solids")

# Volumes the plane z = -15 slices through
(g.model.select(dim=3)
    .crossing_plane({"z": -15}, mode="crossing")
    .to_physical("crossers"))

# The floor (surfaces lying on z = 0)
(g.model.select(dim=2)
    .crossing_plane({"z": 0}, mode="on")
    .to_physical("base"))
```

`dim=0` points, `dim=1` curves, `dim=2` surfaces, `dim=3` volumes.

#### Direction-based filters — `parallel_to` and `normal_along`

For dim-restricted filtering by *direction* (not position), the
`Selection` payload offers two methods:

```python
# Curves: keep only edges whose chord is along a direction
edges = g.model.select("box", dim=1).result()
verticals = edges.parallel_to("z")                    # axis alias
diagonals = edges.parallel_to((1, 1, 0), angle_tol=2) # arbitrary vector

# Surfaces: keep only faces whose normal is along a direction
faces = g.model.select("box", dim=2).result()
horizontals = faces.normal_along("z")
```

Both accept axis aliases (`"x"` / `"y"` / `"z"`) or any non-zero 3-vector
(normalized internally; anti-parallel counts as parallel). Default
`angle_tol` is 1.0°. The methods are **dim-restricted**: `parallel_to`
raises if the Selection contains non-curve entities, `normal_along` raises
for non-surface entities — with a fix-it suggestion in the error.

They chain with the position predicates:

```python
# Vertical edges on the x = 0 wall
(g.model.select("box", dim=1).result()
    .parallel_to("z")
    .select(on={"x": 0}))
```

#### Combining selections — `|`, `&`, `-`

Two Selections can be combined with set-algebra operators. Semantics are
**set-like with deduplication** — a `(dim, tag)` pair never appears
twice in the result, so downstream calls like `to_physical` register each
entity once.

Each operation has both an **operator** form (terse, for one-liners)
and a **named-method** form (discoverable via autocomplete, keeps the
chain fluent — important when you don't want to break out to a
variable).

| Operator | Method | Meaning | Example |
|---|---|---|---|
| `a \| b` | `a.union(b)` | entities in either | `sides = nx.union(ny)` |
| `a & b` | `a.intersect(b)` | entities in both | `edge = top.intersect(front)` |
| `a - b` | `a.difference(b)` | in `a`, not in `b` | `lateral = all.difference(horizontal)` |

```python
surf = g.model.select(dim=2).result()

# Three equivalent ways to grab the lateral sides of an axis-aligned box:
(surf.normal_along("x") | surf.normal_along("y")).to_physical("sides")
(surf - surf.normal_along("z")).to_physical("sides")
surf.normal_along("x").union(surf.normal_along("y")).to_physical("sides")

# Intersection — curves shared by two faces (the edge between them)
top_edges   = m.model.queries.boundary("top",   dim=2, oriented=False)
front_edges = m.model.queries.boundary("front", dim=2, oriented=False)
shared_edge = top_edges.intersect(front_edges)
```

**Why `|` and not `+`?** `Selection` subclasses `list`, where `+`
already means *concatenation with duplicates preserved*. The `|` family
follows Python's `set` / `dict` convention for combining-with-dedup,
which is the right semantics for selection sets — combining the xmin
faces with the ymin faces should give each shared corner edge once, not
twice.

#### Resolve-only `select(...)` — no predicate required

`g.model.select("name", dim=N)` with **no** spatial verb returns
the entities under that name as a chainable selection — useful as an
entry point into the method-style filters:

```python
g.model.select("box", dim=1).result().parallel_to("z").to_physical("verticals")
```

::: apeGmsh.core._selection.Selection

### Geometric primitives (internal)

These classes are constructed automatically by `select()` from raw input.
You never instantiate them directly, but their docstrings describe the
accepted formats.

::: apeGmsh.core._selection.Plane

::: apeGmsh.core._selection.Line
