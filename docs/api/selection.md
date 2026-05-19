# Selection — the unified `.select()` idiom

One fluent, daisy-chainable selection idiom across all four levels —
geometry, the live mesh, the FEM broker, and results. Every `.select()`
returns a chain whose spatial verbs (`in_box`, `in_sphere`, `on_plane`,
`crossing_plane`, `nearest_to`, `where`) and set algebra (`|`, `&`,
`-`, `^`) compose, ending in a level-appropriate terminal.

!!! danger "v2 — BREAKING: the legacy selection surface was removed"
    The pre-v2 selectors — `fem.nodes.get`/`fem.elements.get`/
    `.resolve`, the `results.*.select(...).values()` chain path,
    `g.mesh_selection.add_nodes`/`add_elements`/`from_geometric`,
    `g.model.queries.select`/`queries.line`/`select_all*`,
    `g.model.selection` (`SelectionComposite`), and the four legacy
    `*Chain` classes + `GeometryChain` — have been **removed with no
    deprecation shim** (project-owner-ratified full removal; no
    backward-compat). `.select()` is now the *only* idiom. See
    [Migration](#migration-from-the-legacy-surface-v2-breaking) and
    [Known capability gaps](#known-capability-gaps) below. The classes
    `core._selection.Selection` and `viz.Selection` are **retained by
    architecture** (terminal-payload / viewer-pick-result types) — but
    only as internal payloads, **not** user entry points; their package
    exports were dropped.

## Entry points

`.select()` is available at six entry points across the four levels.
The first call seeds the chain (by name, ids, or the full universe);
subsequent verbs refine it.

| Entry point | Level | Returns | Atoms |
|---|---|---|---|
| [`g.model.select(...)`][apeGmsh.core.Model.Model.select] | Geometry | [`EntitySelection`][apeGmsh.core._selection.EntitySelection] | `(dim, tag)` dimtags |
| [`g.mesh_selection.select(...)`][apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet.select] | Live mesh | [`MeshSelection`][apeGmsh.mesh._mesh_selection.MeshSelection] | node / element ids |
| `fem.nodes.select(...)` | FEM broker | [`MeshSelection`][apeGmsh.mesh._mesh_selection.MeshSelection] | node ids |
| `fem.elements.select(...)` | FEM broker | [`MeshSelection`][apeGmsh.mesh._mesh_selection.MeshSelection] | element ids |
| `results.nodes.select(...)` | Results | [`MeshSelection`][apeGmsh.mesh._mesh_selection.MeshSelection] | node ids |
| `results.elements.select(...)` | Results | [`MeshSelection`][apeGmsh.mesh._mesh_selection.MeshSelection] | element ids |

`fem` is the [`FEMData`](fem.md) snapshot from
`g.mesh.queries.get_fem_data(dim)`; `results` is a
[`Results`](results.md) container. There are **two** concrete terminals
— `EntitySelection` (entity family) and `MeshSelection` (point family);
the five point-level `.select()` entry points all return
`MeshSelection`.

## The verb surface

Every chain — regardless of level — exposes the **same verb names**
with the **same signatures** (the lone exception is the entity family's
`in_box`, see [Two families](#two-families-point-vs-entity)). Each
refining verb returns a new chain of the *same* concrete type, so calls
daisy-chain:

| Verb | Signature | Keeps atoms that… |
|---|---|---|
| `in_box` | `in_box(lo, hi, *, inclusive=False)` | lie in the box `[lo, hi)` (half-open; point family) |
| `in_sphere` | `in_sphere(center, radius)` | lie within the closed ball of `radius` about `center` |
| `on_plane` | `on_plane(point, normal, *, tol)` | lie within `tol` of the plane (`tol` is **required**, keyword-only) |
| `crossing_plane` | `crossing_plane(spec, *, tol=1e-6, mode="crossing")` | (entity family) straddle/`on` test of the entity bbox vs a plane/line spec |
| `nearest_to` | `nearest_to(point, *, count=1)` | are the `count` nearest to `point` (deterministic, lowest-index tie-break) |
| `where` | `where(predicate)` | satisfy `predicate(xyz)` on their coordinate row |

`lo` / `hi` / `center` / `point` / `normal` are 3-sequences.
`crossing_plane` is an **entity-family** predicate; on the point family
it raises `TypeError` (a node/element id has no bounding box — the
straddle test is inexpressible and is rejected loudly, never silently
empty).

!!! warning "`on_plane` has no default tolerance"
    `tol` is keyword-only with **no default** — always pass it
    explicitly, e.g. `.on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)`.

### Set algebra

Two chains of the **same type bound to the same engine** combine with
set operators or their named-method equivalents. The dedup law is
insertion-order-preserving: `a`'s order first, then `b`'s new atoms,
each atom once.

| Operator | Method | Meaning |
|---|---|---|
| `a \| b` | `a.union(b)` | atoms in either |
| `a & b` | `a.intersect(b)` | atoms in both |
| `a - b` | `a.difference(b)` | in `a`, not in `b` |
| `a ^ b` | `a.symmetric_difference(b)` | in exactly one |

Combining two **different** terminal types (an `EntitySelection` with a
`MeshSelection`), or two `MeshSelection`s bound to **different** engines
(two different `FEMData` / `Results`), raises `TypeError` — set algebra
is loud across incompatible spaces. Pair selections from one engine.

### Terminals

| Level | Terminal | Returns |
|---|---|---|
| `g.model.select(...)` | `.result()` / `.to_label(name)` / `.to_physical(name)` / `.to_dataframe()` | `.result()` → the retained-by-architecture [`Selection`][apeGmsh.core._selection.Selection] payload (`.to_label`/`.to_physical`/`.tags()`); `.to_label`/`.to_physical` register a Tier-1 label / Tier-2 physical group |
| `g.mesh_selection.select(...)` | `.result()` / `.ids` / `.coords` / `.save_as(name)` | `.result()` → the same-shape `dict` `MeshSelectionSet.get_nodes`/`get_elements` return; `.ids` the raw ids; `.coords` node coords (node level) or element centroids (element level, fail-loud); `.save_as` persists a named set (**live-mesh engine only**) |
| `fem.nodes.select(...)` | `.result()` / `.ids` / `.coords` | a `NodeResult` (id/coord parity with the pre-v2 broker read) |
| `fem.elements.select(...)` | `.result()` / `.groups()` | a `GroupResult` (per-type element payload; the `GroupResult` itself exposes `.resolve()` / `.groups()`) |
| `results.<nodes\|elements>.select(...)` | `.values(*, component, time=None, stage=None)` | the existing `NodeSlab` / `ElementSlab` — the chain forwards verbatim onto the **retained typed reader** `results.<level>.get(component=, ids=, pg=, label=, selection=)` |

!!! note "Results selections need a component"
    A results chain identifies *where* to read; a slab read still needs
    *what*. `results.<level>.select(...).values(component=...)` is the
    terminal. Calling `.result()` on a results chain raises
    `RuntimeError` (a bare results selection is meaningless without a
    component). The typed reader `results.<level>.get(component=...)` is
    **retained** and is itself the successor to the removed chain
    `.values()`/`ResultChain.get` path.

## Two families: point vs entity

The two terminals have **honestly different** spatial contracts. The CI
contract test asserts per-family laws; it never claims cross-family
identical behavior.

### Point family — `MeshSelection`

`fem.*`, `results.*`, `g.mesh_selection`. Spatial verbs test node
**coordinates** (or element **centroids** for element-level chains; the
centroid is fail-loud — a connectivity id absent from the node set
raises `KeyError`, never a silent row-0 substitution):

- `in_box` is **half-open `[lo, hi)`** by default — an atom exactly on
  an upper face is excluded.
- `in_box(..., inclusive=True)` switches to the **closed box
  `[lo, hi]`** — upper-face atoms are kept.
- `in_sphere` is a closed ball; `on_plane` keeps atoms within `tol` of
  the plane (`|(c - point)·n̂| <= tol`; the normal is normalised by the
  caller).

### Entity family — `EntitySelection`

`g.model.select()`. Atoms are `(dim, tag)` CAD dimtags; there is no
single coordinate per entity, so the contract is entity-typed:

- `in_box` delegates to `gmsh.model.getEntitiesInBoundingBox` — BRep
  bounding-box **CONTAINMENT** (the *whole* entity bbox must lie inside
  the query box, expanded by `Geometry.Tolerance` ≈ 1e-8). It is
  closed-ish, **not** an intersect and **not** half-open: a query box
  exactly equal to an entity's own extent will *not* contain it —
  enclose it comfortably.
- `in_box` therefore **cannot** honor the half-open / `inclusive=`
  knob. Passing `inclusive=` (or any keyword) raises `TypeError` — the
  knob is inexpressible here and is rejected loudly, never silently
  ignored.
- `in_sphere` / `nearest_to` / `where` use the entity bbox **centre**;
  `on_plane` keeps an entity iff *all 8* bbox corners are within `tol`.
  For exact straddle / `on` semantics use `crossing_plane(spec,
  mode="on"|"crossing")` — the v2 successor to the removed
  `g.model.queries.select(on=/crossing=)` predicate (it folds the same
  `Plane` / 2-point `Line` / 3-point `Plane` spec parsing).

## Examples per level

### Geometry — `g.model.select()`

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="frame") as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.model.sync()
    faces = g.model.queries.boundary("box", dim=3, oriented=False)
    g.physical.add_surface([int(t) for _d, t in faces], name="Faces")

    # Seed by PG name (tiered name resolution, contract-locked),
    # refine with entity-family spatial verbs, register a Tier-2 PG.
    (g.model.select("Faces")                          # -> EntitySelection
        .in_box((-0.1, -0.1, -0.1), (1.1, 1.1, 1.1))  # gmsh BRep containment
        .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
        .to_physical("Base"))

    # Exact straddle predicate (the v2 successor to queries.select):
    g.model.select("box", dim=3).crossing_plane(
        {"point": (0, 0, 0.5), "normal": (0, 0, 1)}, mode="crossing")
```

`select(target=None, *, dim=None)` accepts anything the locked geometry
resolver accepts — a label / PG / part name, a bare int tag, a
`(dim, tag)` pair, or a list. `dim` is the resolver's `default_dim`
(used for bare ints and `target=None`), **not** a post-filter.
`.result()` returns the retained `Selection` payload (`.to_label` /
`.to_physical` / `.to_dataframe` are also direct terminals).

### Live mesh — `g.mesh_selection.select()`

```python
# After meshing, still in the live session
node_set = (g.mesh_selection.select()                  # node level
    .in_box((0, 0, 0), (1, 1, 1))                      # half-open [lo, hi)
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-9)
    .result())                                         # {'tags', 'coords'}

# Element level — atoms are element ids of `dim`, verbs use centroids
hexes = (g.mesh_selection.select(level="element", dim=3)
    .in_box((0, 0, 0), (1, 1, 1), inclusive=True)      # closed box
    .ids)

# Named, round-tripping persistence (v2): build via the idiom and
# .save_as (live-mesh engine only), or register explicit ids with the
# retained g.mesh_selection.add(dim, ids, name=).
g.mesh_selection.select().in_box((0, 0, 0), (1, 1, 1)).save_as("base")
shell = (g.mesh_selection.select(name="base")          # id-for-id the set
    .in_sphere((0.5, 0.5, 0.5), 0.4)
    .result())
```

`select(*, level="node", dim=2, ids=None, name=None)` —
`level="element"` uses `dim`; `ids=` seeds an explicit list. `name=`
seeds id-for-id from an **existing** `g.mesh_selection` set (its node
ids for `level="node"`, its element ids for `level="element"`); it only
*reads* the set store, and an unknown name fails loud. `ids=` and
`name=` are mutually exclusive; with neither, the full live-mesh
universe is seeded. The retained registrars `g.mesh_selection.add` /
`from_physical` / `filter_set` / `sort_set` / set ops remain.

### FEM broker — `fem.nodes.select()` / `fem.elements.select()`

```python
fem = g.mesh.queries.get_fem_data(dim=3)

# Selectors (target / pg / label / tag / partition / dim) plus ids=;
# .result() is a NodeResult, .ids / .coords the raw arrays.
top = (fem.nodes.select(pg="Body")
    .in_box((0, 0, 0), (1, 1, 1))                      # half-open
    .on_plane((0, 0, 1), (0, 0, 1), tol=1e-6)
    .result())                                         # -> NodeResult

# Set algebra across two node chains on the same FEMData
a = fem.nodes.select(ids=[1, 2, 3])
b = fem.nodes.select(ids=[2, 3, 4])
both = (a | b).result()                                # union, deduped

# Elements — atoms are element ids; spatial verbs use centroids;
# .groups() / .result() yield the per-type GroupResult.
core = (fem.elements.select(pg="Body")
    .in_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.75))
    .groups())                                         # -> GroupResult
```

### Results — `results.nodes.select()` / `results.elements.select()`

```python
# results bound to a fem (Results(..., fem=...) or results.bind(fem))
slab = (results.nodes.select(pg="Base")
    .in_box(lo, hi)                                    # half-open
    .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6)
    .values(component="displacement_x"))               # -> NodeSlab

# Element results — spatial verbs operate on element centroids
forces = (results.elements.select(pg="Beams")
    .in_box(lo, hi)
    .values(component="globalForce"))                  # -> ElementSlab
```

`results.nodes.select` accepts `pg` / `label` / `selection` / `ids`;
`results.elements.select` adds `element_type=`. `.values(component=...)`
forwards verbatim onto the retained typed reader
`results.<level>.get(component=, ids=, pg=, label=, selection=)`, so it
is id/value parity with that reader.

## Migration from the legacy surface (v2, BREAKING)

The legacy surface was removed with **no shim** (owner-ratified full
removal). Map every old call to its v2 successor:

| Removed | v2 successor |
|---|---|
| `fem.nodes.get(...)` / `.get_ids(...)` / `.get_coords(...)` | `fem.nodes.select(...).result()` / `.ids` / `.coords` |
| `fem.elements.get(...)` / `fem.elements.resolve(...)` | `fem.elements.select(...).groups()` / `.result()` → a `GroupResult` (whose `.resolve()` / `.groups()` replace the old `fem.elements.resolve`) |
| `g.model.queries.select(target, on=/crossing=/not_*)`, `queries.line(...)`, `select_all*` | `g.model.select(target).crossing_plane(spec, mode="on"\|"crossing")` (same `Plane` / 2-pt `Line` / 3-pt `Plane` spec) |
| `g.model.queries.select(...).result().to_label/.to_physical` | `g.model.select(...).to_label(name)` / `.to_physical(name)` / `.to_dataframe()` / `.result()` |
| `results.<level>.select(...).values()` chain path / `ResultChain.get` | the **retained** typed reader `results.<level>.get(component=, ids=, pg=, label=, selection=)` — the chain's `.values(component=)` now forwards onto it |
| `g.mesh_selection.add_nodes(in_box=/on_plane=/…, name=)` / `add_elements(...)` | `g.mesh_selection.select(...).<spatial>.save_as(name)` (live-mesh engine only) **or** the retained explicit-ids registrar `g.mesh_selection.add(dim, ids, name=)` |
| `g.model.selection` / `SelectionComposite`; the `Selection` / `SelectionComposite` package exports | `g.model.select(...) → EntitySelection`. The `core._selection.Selection` (`.result()` payload) and `viz.Selection` (viewer pick-result) **classes are retained by architecture** — internal payload types, not user entry points; only their exports were dropped. |
| `fem.nodes.get(...)` `for nid, xyz in …` iteration | `for nid, xyz in fem.nodes.select(...):` (the `MeshSelection` iterates `(id, payload)`); `.result()` preserves the documented terminal shape |
| Retained (NOT removed — do not migrate) | `g.mesh_selection.add` / `from_physical` / `filter_set` / `sort_set` / `union` / `intersection` / `difference`; the typed `results.<sub>.get(component=)` reader; the `core._selection.Selection` / `viz.Selection` classes |

## Known capability gaps

Two removed capabilities have **no v2 successor**. This is an
owner-ratified consequence of full removal (the SC-12 disposition
class: a removed capability with no successor is documented here, not
re-introduced):

1. **Geometric-selection → named mesh-selection
   (`g.mesh_selection.from_geometric` + `viz.Selection.to_mesh_*`).**
   Both ends were removed; there is no v2 path that turns a geometry
   pick into a persisted named mesh-selection. `.save_as(name)` is
   **live-mesh-engine only** and persists the *current chain's ids*, not
   a geometry round-trip. Nearest manual workaround: resolve the
   geometry to ids yourself, then `g.mesh_selection.add(dim, ids,
   name=)` — this is **not** equivalent to the removed
   geometric-selection round-trip.
2. **The `SelectionComposite` filter grammar
   (`g.model.selection.select_*(labels=fnmatch / kinds= /
   length|area|volume_range= / predicate= / exclude_tags= / physical= /
   at_point=)`).** `EntitySelection` (`g.model.select(...)`) exposes
   only spatial verbs + set algebra + `to_label`/`to_physical`/
   `to_dataframe`/`result`; it has **no** declarative filter-grammar
   equivalent. (The retained `viz.Selection.filter()` still exposes a
   similar grammar, but it is the *viewer pick-result* type reached only
   from a viewer pick — it is **not** a `g.model.select(...)` migration
   path.)

## Behavior changes (already shipped)

Two pre-existing, intentionally-pinned behavior changes shipped earlier
in the program (P3-R); they are end-state here, not introduced by this
docs page. Full migration text is in the [changelog](../changelog.md).

!!! warning "S2 — `g.mesh_selection` box default is half-open"
    The point-family `in_box` is **half-open `[lo, hi)`** by default
    (matching `results`); pass `inclusive=True` for the closed box
    `[lo, hi]`.

!!! warning "S5 — formerly-silent paths fail loud"
    `results` with `selection=` on an import-origin `FEMData` raises
    `RuntimeError`; element-centroid computation raises `KeyError` on an
    unknown connectivity node (never a silent row-0 centroid); a
    loads/masses `__ms__` target with no info raises `KeyError` instead
    of binding to zero nodes.

## Reference

The shared base mixin (chaining + set-algebra + definition-time
verb-name enforcement) and the two concrete v2 terminals. The legacy
`*Chain` classes + `GeometryChain` were removed in v2; the per-family
spatial behavior now lives on `EntitySelection` / `MeshSelection` over
the one `apeGmsh._kernel.spatial` mask kernel.

::: apeGmsh._kernel.chain.SelectionChain

### `EntitySelection` — entity family (geometry, `(dim, tag)`)

::: apeGmsh.core._selection.EntitySelection

### `MeshSelection` — point family (FEM / results / live mesh, ids)

::: apeGmsh.mesh._mesh_selection.MeshSelection
