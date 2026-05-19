# The Selection Chain — maintainer invariants

> [!warning] Rewritten for selection-unification **v2** (full removal)
> This page previously documented the v1 "additive, five-chain"
> architecture (`apeGmsh/_chain.py` + `GeometryChain` / `NodeChain` /
> `ElementChain` / `ResultChain` / `MeshSelectionChain`, with the
> legacy surface "untouched"). **Selection-unification v2 collapsed and
> hard-removed all of that.** The authoritative v2 design record — the
> red/blue adversarial exercise, the phase ledger, the ratified
> decisions (R-v2-1..R-v2-8), the removed-vs-retained ledger — is
> `docs/plans/selection-unification-v2.md` (with the caller-migration
> contract `docs/plans/selection-unification-v2-p3r-callers.md`). This
> page is now only the *operational distillation* of the v2 invariants
> a maintainer must not break.

---

## 1. What v2 shipped, in one paragraph

A single fluent, daisy-chainable selection idiom — `.select()` — at
four levels: geometry (`g.model.select()` → `EntitySelection`), the
FEM broker (`fem.nodes.select()` / `fem.elements.select()` →
`MeshSelection`), results (`results.nodes.select()` /
`results.elements.select()` → `MeshSelection`, terminal
`.values(component=...)`), and the live mesh
(`g.mesh_selection.select()` → `MeshSelection`). The **entire legacy
selection surface was hard-removed** (no shim, no deprecation window):
`g.model.selection` / `SelectionComposite`,
`g.model.queries.select/select_all*/line`,
`g.mesh_selection.add_nodes/add_elements/from_geometric`,
`fem.nodes/elements.get/get_ids/get_coords/resolve`, the chain
`results.*.select(...).get(...)` terminal, the five chain *modules*
(`_node_chain` / `_elem_chain` / `_result_chain` /
`_mesh_selection_chain` and `GeometryChain`), and the package exports
of both `Selection` classes. The behavioural deltas are the
`g.mesh_selection` point-family box default flip (S2) and three
formerly-silent paths that now fail loud (S5).

The architecture that survives:

```
apeGmsh/_kernel/chain.py     leaf — stdlib/typing only
  class SelectionChain        chaining + set-algebra + name enforcement
apeGmsh/_kernel/spatial.py    leaf — box/sphere/plane mask math (pure)

core/_selection.py    EntitySelection(SelectionChain)  FAMILY="entity"
mesh/_mesh_selection.py  MeshSelection(SelectionChain)  FAMILY="point"
  (engine-polymorphic: serves the broker-node / broker-element /
   results / live-mesh hosts; per-engine bodies relocated VERBATIM
   from the four deleted chains by P3-K — behaviour-invisible)

results/_result_engine.py  _ResultChainEngine + engine_for (relocated
                            from the deleted results/_result_chain.py)
mesh/_live_engine.py        _LiveMeshEngine  + engine_for (relocated
                            from the deleted mesh/_mesh_selection_chain)
core/_resolution.py   resolve_target()  shared Loads+Masses resolver
```

**Retained by architecture (R-v2-8 / SC-8 — NOT legacy/compat):**
`core/_selection.Selection` is the entity-side `.result()` terminal
payload; `viz.Selection` is the viewer pick-result type. Only their
*package exports* were dropped — the classes stay. Never call either
"legacy", "deprecated", or "removed".

---

## 2. FP-1 — the import-polarity invariant (the one that bites)

### The mechanism

`core` and `mesh` are in a *latent* import cycle on `main`, and the
process only survives because the cross-package edges have a specific
**eager/deferred polarity**:

- `core → mesh` is **eager** (module-level): `core/LoadsComposite.py`,
  `core/MassesComposite.py`, `core/ConstraintsComposite.py` import
  from `apeGmsh.mesh` at module top, and `core/__init__.py` pulls
  those three composites in.
- `mesh → core` is **deferred** (function-body): e.g.
  `mesh/_mesh_structured.py` does
  `from apeGmsh.core._helpers import resolve_to_dimtags` /
  `from apeGmsh.core._selection import …` *inside a method body*.

Eager-`core→mesh` plus deferred-`mesh→core` **terminates**. Flip any
deferred `mesh→core` (or `viz→core`) edge to eager and `import
apeGmsh` crashes with `ImportError`. A static cycle detector cannot
catch this — the cycle is *already there statically*; only the
eager/deferred polarity of the edge set matters.

### Why the kernel leaves are safe

`apeGmsh/_kernel/chain.py` and `apeGmsh/_kernel/spatial.py` are
**package-root leaves**: they import only the standard library /
numpy. `_kernel` is *not* one of `{core, mesh, viz, results}`, so
importing from it adds **no** cross-package edge to the polarity
baseline.

Two structural facts make the chain hooks safe:

1. `core/_selection.py` imports `from .._kernel.chain import
   SelectionChain` at module top — allowed because `_kernel.chain` is
   a root leaf, not a `core↔mesh/viz/results` edge.
2. `core/__init__.py` imports the composites **only** — it does
   **not** import `_selection` / `_kernel` / `_resolution`. So a
   sibling leaf is reachable *without* dragging in the eager
   `core→mesh` chain.

Every `.select()` host hook uses the **deferred-import idiom** —
`from ._mesh_selection import MeshSelection` *inside the `select()`
method body*, mirroring `mesh/_mesh_structured.py`. See
`mesh/FEMData.py` (`select()` for nodes / elements),
`results/_composites.py`, `mesh/MeshSelectionSet.py`,
`core/Model.py` (the docstring there spells out the deferred
rationale).

### The CI tripwire

`tests/test_import_dag_polarity.py` is the lock. It snapshots the
**frozen set of eager cross-package edges** among `{core, mesh, viz,
results}` (widened to also cover `_kernel` + `fem`) in `BASELINE` and
fails on **any** add or remove; it asserts `core/__init__.py` does
not import the selection leaves; and it asserts the leaf + the
point-family chain + a deferred host hook import cleanly.

> [!warning] Maintainer rule
> If you add a new eager cross-package import among `{core, mesh, viz,
> results, _kernel, fem}`, this test goes red — intentionally. If the
> edge is genuinely required, update `BASELINE` **in the same commit**
> so the import-graph change is an explicit, reviewed diff — never a
> silent regression. Adding a new `.select()` host must use the
> deferred-import idiom and must **not** require a `BASELINE` change.

The editable install resolves `apeGmsh` to the **main repo** `src/`,
not a worktree. Every in-process gate must set
`PYTHONPATH=<worktree>\src` and assert `apeGmsh.__file__` is the
worktree, or a green is a false negative
(`docs/plans/selection-unification-v2.md` §7 keystone).

---

## 3. The two retained `Selection` classes (R-v2-8 / SC-8)

There are two classes both named `Selection`, **structurally
incompatible** and **both retained by architecture**:

| | `core/_selection.py` `Selection(list)` | `viz/Selection.py` `Selection` |
|---|---|---|
| Base | `class Selection(list)` (mutable list subclass) | frozen, `__slots__ = ('_dimtags','_dim','_parent')` |
| Tags accessor | `.tags()` — a **method** | `.tags` — a **property** |
| Constructor | `(dimtags, *, _queries=)` | `(dimtags, parent)` |
| Refinement | `.select(on=...)` / `.parallel_to` / `.normal_along` / `.to_label` / `.to_physical` | `.filter` / `.limit` / `.sorted_by` |
| Role | `.result()` terminal payload of `EntitySelection` | interactive-viewer pick-result (`viewer.selection`) |

`.tags()` method vs `.tags` property alone makes any cross-class
identity test impossible — there is no single base they can both
honour. They were proven irreconcilable in v1 and v2 **ratified
retaining both** (R-v2-8 / SC-8).

> [!warning] Maintainer rule
> Do **not** merge, reparent, or "unify" these two `Selection`
> classes, and do **not** describe them as legacy/deprecated/removed.
> `core/_selection.Selection` is the byte-stable `.result()` payload
> of `EntitySelection` (`g.model.select(...)`); `viz.Selection` is the
> viewer pick-result type both viewers construct via a deferred
> in-method import. v2 dropped only their **package exports** — the
> classes are load-bearing. The `viz.Selection.filter()` rich-filter
> grammar is **viewer-pick-only**; it is **not** a `g.model.select`
> migration path (a documented capability gap — §10).

---

## 4. The `__init_subclass__` + `REQUIRED_VERBS` + `FAMILY` contract

`SelectionChain` (`apeGmsh/_kernel/chain.py`) enforces the shared
surface at **class-definition time**, which is strictly stronger than
a CI test (a bad subclass is an `ImportError`-class failure the moment
its module loads).

`__init_subclass__`:

- exempts abstract intermediates (no `FAMILY` set) — only concrete
  leaves are checked;
- rejects a `FAMILY` not in `VALID_FAMILIES = ("entity", "point")`;
- requires every verb in `REQUIRED_VERBS` (`in_box, in_sphere,
  on_plane, nearest_to, where, union, intersect, difference`) to be
  present and callable;
- requires every hook in `_REQUIRED_HOOKS` (`_coords_of,
  _spatial_box, _spatial_sphere, _spatial_plane, _materialize`) to be
  **overridden** (not left as the base `NotImplementedError` stub).

The set-algebra dedup law is **one** law — insertion-order-preserving
`dict.fromkeys`. Every refining verb returns `type(self)(…)` so
chaining is covariant. Cross-type and cross-engine combination is
**loud** (`_compatible` raises `TypeError`).

> [!note] The family is now **two concrete leaves**
> Post-v2 the concrete `SelectionChain` subclasses are
> `EntitySelection` (`FAMILY="entity"`) and `MeshSelection`
> (`FAMILY="point"`, engine-polymorphic over the four point hosts) —
> the five v1 chain classes were collapsed into these two. The
> box/sphere/plane mask math common to the point engines lives once in
> `apeGmsh/_kernel/spatial.py`; the per-engine `_coords_of` / centroid
> / `_materialize` bodies were relocated **verbatim** into
> `MeshSelection` by P3-K (a behaviour-invisible pure move). The
> per-family `in_box` signature split is preserved (see §5).

---

## 5. The two `in_box` families (ratified R3 / R4)

| | POINT family (`MeshSelection`) | ENTITY family (`EntitySelection`) |
|---|---|---|
| Atoms | node ids / element ids | `(dim, tag)` CAD dimtags |
| `in_box` default | half-open `[lo, hi)` per axis (canonical, R4) | gmsh `getEntitiesInBoundingBox` — BRep bbox-**CONTAINMENT**, closed, box expanded by `Geometry.Tolerance`≈1e-8 |
| `inclusive=` | `inclusive=True` → closed `[lo, hi]` | **any** keyword (incl. `inclusive=`) → `TypeError` (fail loud, never silently ignored) |
| Coordinate | node coords / element centroid (mean of node coords) | entity bounding-box centre (sphere/nearest/where), 8 corners (on_plane) |

Point-family box logic lives in `apeGmsh/_kernel/chain.py`
(`_spatial_box`; `inclusive` selects `<= hi` vs `< hi`).
`EntitySelection.in_box` (`core/_selection.py`) overrides it with a
`**kw`-rejecting signature and delegates to
`gmsh.model.getEntitiesInBoundingBox` per distinct dim, intersecting
with the chain (preserving insertion order). This is the one verb the
cross-chain signature test exempts from identity.

> [!note] R3 / R4 are decided behaviour, not bugs
> The point-family box default went **closed → half-open** in S2 (a
> reconciliation: `g.mesh_selection`'s box was closed while `results`'
> box was already half-open). The entity family physically *cannot*
> express a half-open box (gmsh has no such knob), so it rejects the
> kwarg loudly. Do not "fix" either by trying to make them agree.

---

## 6. FP-4 — the deliberate FEMData node-vs-element swallow asymmetry

`FEMData` does **not** call the shared `resolve_target`. It has its
own resolvers with a **deliberate, documented** asymmetry that must
not be touched:

- **Node path** — `FEMData._resolve_nodes` catches **`KeyError`
  only**. A `ValueError` from a wrong-dimension reference propagates
  (fails loud).
- **Element path** — `FEMData._resolve_elem_ids` catches
  **`(KeyError, ValueError)`** — a broader swallow, by design.

This asymmetry is a **correctness invariant** locked by
`tests/test_resolution_contract.py` + `tests/test_target_resolution.py`
(byte-unchanged through P3 per `selection-unification-v2.md` §7).

`fem.nodes.select()` / `fem.elements.select()` **reuse these exact
resolvers** — they delegate verbatim to `_resolve_nodes` /
`_resolve_elem_ids` (the same path the removed `.get()` used), so the
resolved selection is exactly what the locked resolution contract
returns and the asymmetry is preserved *by reuse*. The element-side
auxiliary filters (`dim` / `element_type` / `partition`) go through
the shared private `_filtered_groups` helper (the P3-R M-STOP-1
factoring of the old `ElementComposite.get` filter body) — both the
`select` aux-branch and internal callers use it; never re-implement
`resolve_type_filter`.

> [!warning] Maintainer rule
> A new `.select()` host must reuse the host's existing resolver —
> never add a new name→entity resolver or "harmonise" the FP-4
> asymmetry. `core/_helpers._resolve_string` and the FEMData
> node/element resolvers keep their own paths by design.

---

## 7. S1 — the shared Loads+Masses resolver (unchanged by v2)

`core/_resolution.py` `resolve_target(parent, target, source, *,
expected_dim, not_found_prefix, noun)` is the **one shared engine for
Loads + Masses only** — a pure de-duplication of the byte-identical
`LoadsComposite` / `MassesComposite` `_resolve_target` bodies. It is
itself a **leaf** (gmsh + stdlib; the one intra-`core` symbol imported
deferred inside the function), so it does not perturb the FP-1
baseline. v2 did not touch it.

> [!warning] Maintainer rule
> Do **not** route the FEMData broker resolvers or
> `core/_helpers._resolve_string` through `core/_resolution.py`. The
> contract tests lock the Loads/Masses/Constraints + `core/_helpers`
> fail-loud surface; the broker path is deliberately separate (FP-4).

---

## 8. S2 / S5 behavioural deltas

**S2 — point-family box default closed → half-open.** The point-family
`in_box(lo, hi)` is half-open `[lo, hi)` by default, matching
`results`. `inclusive=True` restores the closed `[lo, hi]` box. The
behaviour-changing `_mesh_filters.py` flip is the reviewed
production+assertion diff that shipped in P3-R.

**S5 — three formerly-silent paths now fail loud:**

1. **Results `selection=` on import-origin fem** —
   `from_msh`/MPCO/native produce `mesh_selection=None`;
   `results/_composites.py` `_resolve_node_ids` / `_resolve_element_ids`
   raise `RuntimeError` instead of silently resolving to an empty set
   (locked by a characterization pin).
2. **Loads/Masses `__ms__` consumer** —
   `core/LoadsComposite.py` `_target_nodes` raises `KeyError` instead
   of silently binding a load to nothing; the `MassesComposite`
   counterpart matches.
3. **Element-centroid fail-loud** — element centroid with a
   connectivity id absent from the node set raises `KeyError` instead
   of an `np.clip` silent corruption. The per-engine fail-loud
   centroid is `MeshSelection._centroid_map_live`
   (`mesh/_mesh_selection.py`), **not** `_kernel/spatial.py` (which
   unifies only the box/sphere/plane mask math). This also makes the
   direct `results.elements.in_box` / `nearest_to` / `on_plane`
   helpers fail loud.

---

## 9. How to extend the chain safely

1. **Prefer extending `MeshSelection`'s engine polymorphism** over a
   new module. The point hosts (broker-node / broker-element /
   results / live-mesh) all return the *same* `MeshSelection`,
   dispatched on the engine kind. A genuinely new point host adds an
   engine + dispatch arm, not a new chain class.

2. If a new concrete `SelectionChain` subclass is truly needed:
   subclass it, set `FAMILY` (`"point"` / `"entity"`), implement every
   `_REQUIRED_HOOKS` hook; import **only** `from .._kernel.chain
   import SelectionChain` (+ `_kernel.spatial`, numpy/stdlib) at
   module top — never `apeGmsh.core` / `mesh` / `viz` / `results`.
   `__init_subclass__` rejects the class at import if you miss a verb,
   a hook, or use a bad `FAMILY`.

3. **Add the `.select()` host hook** with a **deferred** import inside
   the method body, mirroring `mesh/FEMData.py`. Never import the
   chain at the host module top.

4. **Reuse the host's existing resolver** for name seeding (FP-4 /
   §6). Do **not** write a new name→entity resolver. For
   `g.mesh_selection.select(name=N)` the supported route is the
   two-step `from_physical(...)` **then** `select(name=...)`; it
   delegates verbatim to `get_tag`/`get_nodes`/`get_elements` via the
   private `_seed_ids_by_name` (no new resolver, only reads `_sets`),
   and fails loud on an unknown name. (`_seed_ids_by_name`'s `KeyError`
   message — which still names the retained `from_physical`
   register-then-select route — is pinned by
   `tests/test_mesh_selection_chain_name_seed.py`; leave it.)

5. **Materialise to the retained terminal type.** Reuse the existing
   payload via a deferred import inside `_materialize`
   (e.g. `NodeResult` / `GroupResult` / the retained
   `core/_selection.Selection`). Do not invent a new return type and
   do not touch the retained `Selection` classes.

6. **Run the gates** (opensees venv; `PYTHONPATH=<worktree>\src`;
   confirm `apeGmsh.__file__` is the worktree):

   ```
   pytest tests/test_import_dag_polarity.py \
          tests/test_resolution_contract.py \
          tests/test_target_resolution.py \
          tests/test_pin_resolution_v2.py -q
   ```

   `test_import_dag_polarity.py` must stay green **with `BASELINE`
   unchanged** — if it demands a `BASELINE` edit, you added an eager
   cross-package import (step 2 or 3 violated). Fix the import, do not
   edit `BASELINE`.

---

## 10. Capability gaps (no v2 successor)

Two removed-surface capabilities have **no** v2 replacement —
documented, not papered over:

1. **`from_geometric` / `viz.Selection.to_mesh_*`** — the one-step
   "geometric entity selection → named mesh selection without a
   physical group" bridge. Both ends were removed. The supported route
   is the two-step `to_physical` (pre-mesh) + `from_physical`
   (post-mesh).
2. **The `SelectionComposite.select_*` rich filter grammar** —
   `labels=` fnmatch, `kinds=`, `length/area/volume_range=`,
   `predicate=fn`, `exclude_tags=`, `physical=`, `at_point=`,
   `on_axis=`, `horizontal=`/`vertical=`/`aligned=`. `EntitySelection`
   has only spatial verbs + set-ops + `.to_*` terminals; the retained
   `viz.Selection.filter()` carries the grammar but is
   **viewer-pick-only**, not a `g.model.select` migration path.

---

## See also

- `docs/plans/selection-unification-v2.md` — the authoritative v2
  design record (R-v2-1..R-v2-8, the phase ledger, the
  removed-vs-retained ledger, the M-CORRECTION notes).
- `docs/plans/selection-unification-v2-p3r-callers.md` — the
  caller-migration contract (the 15 PROD sites, M-STOP-1..3).
- [Selection in apeGmsh](guide_selection.md) — user-facing geometry +
  mesh selection.
- [Reading & Filtering Results](guide_results_filtering.md) —
  results `.select()` and the retained typed reader.
- [The FEM Broker](guide_fem_broker.md) — `fem.nodes/elements.select()`.
- [apeGmsh model queries](guide_queries.md) — the retained
  `g.model.queries` topology surface.
- [MIGRATION_v1](MIGRATION_v1.md) — §6a is the v2 full-removal
  migration table and the two capability gaps.
