# Results

Post-processing container.

## Construction — `model=` is required

Every `Results` constructor requires a model, so the post-processing
container always has a read-side broker to resolve names, PGs, and
connectivity against:

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel

# Native apeGmsh HDF5 (a Composed file carrying results + model is common)
model = OpenSeesModel.from_h5("run.h5")
results = Results.from_native("run.h5", model=model)     # model= REQUIRED

# STKO .mpco — model_h5= points at the sibling model archive
results = Results.from_mpco("run.mpco", model_h5="model.h5")  # model_h5= REQUIRED

# Live recorders — fem= and model=
results = Results.from_recorders(spec, "out/", fem=fem, model=model)
```

Omitting `model=` (or `model_h5=` for `from_mpco`) raises `TypeError`.

`results.model` is **never `None`** on a constructed `Results` (ADR 0020
INV-1). Reach the neutral `FEMData` zone through the broker chain:

```python
osm = results.model            # OpenSeesModel broker (never None)
fem = results.model.fem        # neutral FEMData zone
```

`results.fem` is the locally-bound snapshot and may differ from
`results.model.fem` after a `.bind()`.

### Lineage — `results.lineage` warns, never raises

`results.lineage` returns a `Lineage(fem_hash, model_hash,
results_hash, warnings)` describing the git-style `fem → model →
results` hash chain. Stored-vs-recomputed mismatches surface as
`[lineage] ...` strings in `lineage.warnings`; the property itself
**never raises**. Call `lineage.assert_clean()` to escalate any
warnings to `LineageError`.

!!! note "`BindError` is gone"
    Construction no longer rejects a mismatched FEM with `BindError`
    (deleted in the three-broker refactor). `results.bind(fem)`
    performs no hash validation — pairing a FEM with the right run is
    the user's responsibility, reported through `lineage.warnings`.

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
    options:
      inherited_members: true

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

::: apeGmsh._vocabulary
    options:
      members:
        - expand_shorthand
        - is_canonical
        - ALL_CANONICAL
