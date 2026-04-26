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

::: apeGmsh.core._selection.Selection

### Geometric primitives (internal)

These classes are constructed automatically by `select()` from raw input.
You never instantiate them directly, but their docstrings describe the
accepted formats.

::: apeGmsh.core._selection.Plane

::: apeGmsh.core._selection.Line
