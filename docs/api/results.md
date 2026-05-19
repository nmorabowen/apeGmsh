# Results

Post-processing container.

## Fluent selection — `.nodes.select()` / `.elements.select()`

`results.nodes.select(...)` and `results.elements.select(...)` are the
results entries of the unified, daisy-chainable
[selection idiom](selection.md). `.select()` returns a
[`MeshSelection`][apeGmsh.mesh._mesh_selection.MeshSelection] (point
family); the terminal is `.values(component=...)`, which forwards to
the **retained** `results.<level>.get(...)` reader and returns the
same slab (`NodeSlab` / `ElementSlab`) with id/value parity.

```python
slab = (results.nodes.select(pg="Base")
    .in_box(lo, hi)                                # half-open [lo, hi)
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    .values(component="displacement_x"))           # -> NodeSlab
```

A bare results selection needs a component — `.result()` raises
`RuntimeError`; use `.values(component=...)`. Element spatial verbs
operate on element centroids.

!!! warning "S5 — formerly-silent results paths now raise"
    `results` with `selection=` on an import-origin
    (`from_msh`/MPCO/native) `FEMData` now raises `RuntimeError`
    instead of resolving to an empty set; results element-centroid
    computation raises `KeyError` on an unknown connectivity node,
    which also makes the legacy
    `results.elements.in_box`/`nearest_to`/`on_plane` helpers fail
    loud. See the [changelog](../changelog.md).

See [Selection](selection.md) for the full idiom; results
sub-composite `.select()` (`gauss`/`fibers`/`layers`/`line_stations`/
`springs`) is a tracked, not-yet-shipped follow-up.

::: apeGmsh.results.Results.Results

## Slabs

Tabular dataclasses returned by reader queries.

::: apeGmsh.results._slabs

## Readers

Reader protocol and supporting types shared by every backend.

::: apeGmsh.results.readers._protocol

## Live capture

Recorder wiring used during a live OpenSees analysis.

### `LiveRecorders`

::: apeGmsh.results.live._recorders.LiveRecorders

### `LiveMPCO`

::: apeGmsh.results.live._mpco.LiveMPCO

### `DomainCapture`

::: apeGmsh.results.capture._domain.DomainCapture

## Inspect

::: apeGmsh.results._inspect.ResultsInspect

## Vocabulary

Canonical result names and shorthand expansion.

::: apeGmsh.results._vocabulary
    options:
      members:
        - expand_shorthand
        - is_canonical
        - ALL_CANONICAL
