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

### Geometric predicates — cheat sheet

The full set of filters available on `queries.select(...)` and `Selection`:

| Predicate | Where | Dim | Example | Keeps entities that… |
|---|---|---|---|---|
| `on=` | kwarg on `select()` | any | `on={"z": 0}` | lie **entirely on** the plane |
| `crossing=` | kwarg on `select()` | any | `crossing={"z": 0}` | **straddle** the plane |
| `not_on=` | kwarg on `select()` | any | `not_on={"z": 0}` | are **not entirely on** the plane |
| `not_crossing=` | kwarg on `select()` | any | `not_crossing={"z": 0}` | lie **entirely on one side** |
| `.parallel_to(...)` | method on `Selection` | 1 (curves) | `edges.parallel_to("z")` | are curves whose chord direction is **parallel** to it |
| `.normal_along(...)` | method on `Selection` | 2 (surfaces) | `faces.normal_along("z")` | are surfaces whose **normal** is along it |

#### Primitive formats accepted by `on=` / `crossing=` / `not_on=` / `not_crossing=`

| Form | Meaning |
|---|---|
| `{"z": 0}` / `{"x": 5}` / `{"y": -3}` | Axis-aligned plane |
| `[(x1,y1,z1), (x2,y2,z2)]` | Infinite line through 2 points (for curves in 2-D) |
| `[(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]` | Infinite plane through 3 points (for surfaces / volumes) |
| `m.model.queries.plane(...)` | `Plane` object — axis-aligned, 3-point, or `normal=`/`through=` |
| `m.model.queries.line(p1, p2)` | `Line` object — explicit 2-point construction |

#### Direction formats accepted by `.parallel_to(...)` / `.normal_along(...)`

| Form | Meaning |
|---|---|
| `"x"`, `"y"`, `"z"` | Axis alias |
| `(1, 0, 0)` / `(1, 1, 0)` | Any non-zero 3-vector (normalized internally) |
| `angle_tol=2.0` | Tolerance in **degrees**; default `1.0`. Anti-parallel counts as parallel. |

#### Entry-point methods on `queries` (build a Selection from scratch)

| Method | Returns |
|---|---|
| `select_all()` | every entity, all dims |
| `select_all_points()` | dim=0 |
| `select_all_curves()` | dim=1 |
| `select_all_surfaces()` | dim=2 |
| `select_all_volumes()` | dim=3 |
| `select(name_or_dimtags, dim=N)` | by PG / label name, or from an explicit set |

### Selection — result type for `select()`

`select()` returns a `Selection` — a chainable list of `(dim, tag)` pairs.
No import is needed; you receive one whenever you call `select()`.

```python
curves = m.model.queries.boundary(surf, oriented=False)

# axis-aligned plane
bottom = m.model.queries.select(curves, on={'y': 0})

# 2-point line
mid    = m.model.queries.select(curves, crossing=[(0,5,0),(5,5,0)])

# chain to narrow further
left_bottom = (m.model.queries
    .select(curves, on={'y': 0})
    .select(on={'x': 0}))

# extract bare tags for downstream calls
m.mesh.structured.set_transfinite_curve(bottom.tags(), n=11)
```

#### Starting from every entity of a dimension

When parsing an imported `.geo` / STEP file with no labels yet, use the
`select_all_*` shortcuts as your starting point:

```python
# Every volume in the model, as a chainable Selection
m.model.queries.select_all_volumes().to_physical("solids")

# Volumes the plane z = -15 slices through
(m.model.queries
    .select_all_volumes()
    .select(crossing={"z": -15})
    .to_physical("crossers"))

# The floor (surfaces lying on z = 0)
(m.model.queries
    .select_all_surfaces()
    .select(on={"z": 0})
    .to_physical("base"))
```

Symmetric helpers exist for each dimension:
`select_all_points()`, `select_all_curves()`, `select_all_surfaces()`,
`select_all_volumes()`. Use `select_all()` (no args) to get **every**
entity across all dimensions.

#### Direction-based filters — `parallel_to` and `normal_along`

For dim-restricted filtering by *direction* (not position), Selection
offers two methods:

```python
# Curves: keep only edges whose chord is along a direction
edges = m.model.queries.select("box", dim=1)
verticals = edges.parallel_to("z")                    # axis alias
diagonals = edges.parallel_to((1, 1, 0), angle_tol=2) # arbitrary vector

# Surfaces: keep only faces whose normal is along a direction
faces = m.model.queries.select("box", dim=2)
horizontals = faces.normal_along("z")
```

Both accept axis aliases (`"x"` / `"y"` / `"z"`) or any non-zero 3-vector
(normalized internally; anti-parallel counts as parallel). Default
`angle_tol` is 1.0°. The methods are **dim-restricted**: `parallel_to`
raises if the Selection contains non-curve entities, `normal_along` raises
for non-surface entities — with a fix-it suggestion in the error.

They chain with the existing position predicates:

```python
# Vertical edges on the x = 0 wall
(m.model.queries
    .select("box", dim=1)
    .parallel_to("z")
    .select(on={"x": 0}))
```

#### Resolve-only `select(...)` — no predicate required

`queries.select("name", dim=N)` with **no** geometric predicate returns
the entities under that name as a chainable Selection — useful as an
entry point into the method-style filters:

```python
m.model.queries.select("box", dim=1).parallel_to("z").to_physical("verticals")
```

::: apeGmsh.core._selection.Selection

### Geometric primitives (internal)

These classes are constructed automatically by `select()` from raw input.
You never instantiate them directly, but their docstrings describe the
accepted formats.

::: apeGmsh.core._selection.Plane

::: apeGmsh.core._selection.Line
