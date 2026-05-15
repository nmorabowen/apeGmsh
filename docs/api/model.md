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
`select_all_volumes()`.

::: apeGmsh.core._selection.Selection

### Geometric primitives (internal)

These classes are constructed automatically by `select()` from raw input.
You never instantiate them directly, but their docstrings describe the
accepted formats.

::: apeGmsh.core._selection.Plane

::: apeGmsh.core._selection.Line
