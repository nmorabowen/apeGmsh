# Mesh — `g.mesh`

Meshing composite. Seven focused sub-composites.

## `g.mesh`

::: apeGmsh.mesh.Mesh.Mesh

## Sub-composites

### `g.mesh.generation`

::: apeGmsh.mesh._mesh_generation._Generation

### `g.mesh.sizing`

::: apeGmsh.mesh._mesh_sizing._Sizing

### `g.mesh.field`

::: apeGmsh.mesh._mesh_field.FieldHelper

### `g.mesh.structured`

::: apeGmsh.mesh._mesh_structured._Structured

#### One-call recipe — `set_transfinite()`

For the common case of "apply transfinite + recombine to one entity, many,
or the whole model," use the unified
[`set_transfinite()`](#apeGmsh.mesh._mesh_structured._Structured.set_transfinite)
method. It infers the dim from the target and runs the appropriate cascade
(curves → faces → recombine → volume):

```python
# Whole model, uniform — typical "just give me hexes"
g.mesh.structured.set_transfinite(n=11)

# Axis-aligned hex, per-axis sizing (most readable)
g.mesh.structured.set_transfinite("layer_top",
                                   n={"x": 101, "y": 101, "z": 6})

# Rotated hex, per principal axis (works for any orientation)
g.mesh.structured.set_transfinite("rotated_box", n=(11, 11, 21))

# Length-based sizing (per axis)
g.mesh.structured.set_transfinite("layer_top",
                                   size={"x": 100, "y": 100, "z": 10})
```

**Sizing forms** — all three coexist; pick by readability:

| Form | Use when | Example |
|---|---|---|
| **scalar** `n=11` | uniform on every edge; any orientation | `n=11` |
| **dict** `n={"x":..., "y":..., "z":...}` | axis-aligned hex, want per-axis counts | `n={"x":11, "y":11, "z":21}` |
| **tuple** `n=(n1, n2, n3)` | any rotation; counts in `(X-closest, Y-closest, Z-closest)` order | `n=(11, 11, 21)` |

**Behavior on incompatible geometry** — entities whose edges don't cluster
into the expected principal-axis count (e.g. a face split by a boolean op
into a 5-sided patch) are **warned and skipped** rather than failing the
whole call. Use
[`set_transfinite_automatic()`](#apeGmsh.mesh._mesh_structured._Structured.set_transfinite_automatic)
if you want silent skipping with no warning.

For per-edge bias (Progression / Bump / Beta with `coef=`), explicit
corner lists, or custom triangle arrangement, drop to the granular methods:
`set_transfinite_curve()`, `set_transfinite_surface()`,
`set_transfinite_volume()`.

### `g.mesh.editing`

::: apeGmsh.mesh._mesh_editing._Editing

### `g.mesh.queries`

::: apeGmsh.mesh._mesh_queries._Queries

### `g.mesh.partitioning`

::: apeGmsh.mesh._mesh_partitioning._Partitioning

## Supporting types

::: apeGmsh.mesh.PhysicalGroups.PhysicalGroups

::: apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet

::: apeGmsh.mesh.MeshSelectionSet.MeshSelectionStore

::: apeGmsh.mesh.MshLoader.MshLoader

!!! warning "Legacy"
    `Partition` below is the standalone, pre-composite class. New code
    should use the live `g.mesh.partitioning` composite documented above.

::: apeGmsh.mesh.Partition.Partition

::: apeGmsh.mesh.View.View

## Algorithms & enums

::: apeGmsh.mesh._mesh_algorithms

::: apeGmsh.mesh._mesh_partitioning.RenumberResult

::: apeGmsh.mesh._mesh_partitioning.PartitionInfo

## Group sets

::: apeGmsh.mesh._group_set.PhysicalGroupSet

::: apeGmsh.mesh._group_set.LabelSet
