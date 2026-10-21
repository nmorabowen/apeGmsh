# Results

Post-processing container.

## Construction — every `Results` has a model

Every `Results` carries a model, so the post-processing container always
has a read-side broker to resolve names, PGs, and connectivity against.
An in-process run recorded with `spec.emit_default(...)` opens with
`run.results()`, which picks the reader for the route the build took and
passes the model for you. Each reader can also be called directly:

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel

# The default route: run.results() after `with spec.emit_default(...) as run:`
results = run.results()

# Ladruno fork .ladruno: model_h5= binds the archive's PGs and labels
results = Results.from_ladruno("run.ladruno", model_h5="model.h5")

# STKO .mpco: model_h5= points at the sibling model archive
results = Results.from_mpco("run.mpco", model_h5="model.h5")  # model_h5= REQUIRED

# Classic recorder files: fem= and model=
model = OpenSeesModel.from_h5("model.h5")
results = Results.from_recorders(spec, "out/", fem=fem, model=model)  # model= REQUIRED

# Native apeGmsh HDF5, e.g. the modes file EigenResult.to_native() writes
p = ops.eigen(3).to_native()          # the file embeds the bridge's model
results = Results.from_native(p, model=OpenSeesModel.from_h5(p))  # model= REQUIRED
```

Omitting `model=` (`from_recorders`, `from_native`) or `model_h5=`
(`from_mpco`) raises `TypeError`. `from_ladruno` builds its broker from
the file's own `MODEL` group when `model_h5=` is left out, but then the
read has no physical groups or labels.

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

### `LiveDefault`

The default route, returned by `ResolvedRecorderSpec.emit_default`.

::: apeGmsh.results.live._default.LiveDefault

### `read_deck_mpco`

::: apeGmsh.results.live._default.read_deck_mpco

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
