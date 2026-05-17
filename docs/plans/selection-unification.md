# Selection / Resolution Unification — Hardened Plan

Status: **S3 DONE (S3a–S3e landed); S5 pending**

Phase ledger (commit hashes):

| Phase | Status | Commit |
|-------|--------|--------|
| plan  | DONE   | `58f2b58` |
| S0a   | DONE   | `66b2e35` |
| S0b   | DONE   | `7c235d3` |
| S1    | DONE   | `f9af1a1` |
| S2    | DONE   | `b93437e` |
| S3a   | DONE   | `31ba990` |
| S3b   | DONE   | `9045efb` |
| S3c   | DONE   | `02aaf21` |
| S3d   | DONE   | `26c507f` |
| S3e   | DONE (pending commit) | _this change_ |
| S5    | pending | — |
Author: design + 4-wave red/blue adversarial exercise (Opus)
Scope: unify selection / parsing / resolution across the four levels —
Geometry, Mesh (`g.mesh_selection`), FEM Broker (`FEMData`), Results.

This document is the source of truth the implementation builds from.
Every load-bearing claim below was verified at source during the
exercise; file:line references are given so they can be re-checked.

---

## 1. Goal

One shared, intuitive, *daisy-chainable* selection idiom + one shared
resolution/spatial engine across all four levels, without breaking the
locked resolution contract or backward compatibility.

User-stated, non-negotiable requirements:

- Intuitive, fluid, verbose, **daisy-chainable** at all four levels.
- "Share a library logic" — one engine, not N hand-mirrored copies.
- A shared base/inheritance that **forces** naming consistency, with
  geometry and mesh kept as **separate** composites that share that base.
- Existing public methods keep working (facade; minimal caller churn).

---

## 2. What the sweep found (6 hard truths)

1. **Two incompatible geometry `Selection` classes.**
   `core/_selection.py:361` `class Selection(list)` (mutable list
   subclass, `.tags()` **method**, ctor `(dimtags, *, _queries=)`,
   `.to_label/.to_physical`) vs `viz/Selection.py:99` `class Selection`
   (frozen `__slots__`, `.tags` **property**, ctor `(dimtags, parent)`,
   `.filter/.limit/.sorted_by`, and its `labels=`/`physical=` filters
   **bypass** the label→PG→part precedence).

2. **The resolution contract only covers half the library.**
   `tests/test_resolution_contract.py` + `tests/test_target_resolution.py`
   lock label→PG→part fail-loud for `core/_helpers` + Loads/Masses/
   Constraints only. The broker (`mesh/FEMData.py` `_resolve_one_target`,
   `mesh/_group_set.py` `_resolve`) and Results
   (`results/_composites.py` `_resolve_node_ids`) re-implement
   precedence by hand-mirrored code — not contract-locked.

3. **`MassesComposite._resolve_target` is a byte-for-byte clone of
   `LoadsComposite._resolve_target`** (`core/MassesComposite.py:606-729`
   vs `core/LoadsComposite.py:908-1033`; only the noun differs).

4. **Spatial math exists 4×, with a real divergence.**
   `mesh/_mesh_filters.py:62-66` box is **closed-closed**;
   `results/_composites.py:189` box is **half-open upper**;
   `mesh/_constraint_resolver/_geom.py` is a scipy KD-tree;
   `core/_selection.py` is gmsh BB-corner on/crossing. The mesh vs
   results box divergence exists **on `main` today**.

5. **The named-selection chain works** (`g.mesh_selection` →
   `fem.mesh_selection` snapshot at `mesh/_fem_factory.py:408` →
   `results(selection=)` at `results/_composites.py:294`) end-to-end
   through HDF5, **except** import-origin FEMData
   (`from_msh`/MPCO/native) has `mesh_selection=None`
   (`mesh/FEMData.py:1119,1124`), where a `selection=` call can resolve
   to a silent-empty set instead of failing loud.

6. **One class cannot span all four levels.** Geometry mutates live
   gmsh (`.to_label/.to_physical`); broker/results are immutable
   detached snapshots. `core` is the lowest runtime layer;
   `core/_helpers.py` is already the shared resolver already imported
   by `mesh`; `results` has **zero runtime dependency on `mesh`**
   (all `TYPE_CHECKING`).

---

## 3. Adversarial outcome — what changed

The original "shared `@final` ABC reparenting the existing Selections"
was **killed** by the red team and conceded at source by blue:

- **FP-1 (fatal):** `core` is already in a *latent* import cycle with
  `mesh`. Eager `core→mesh` (`core/LoadsComposite.py:40-42`,
  `MassesComposite.py:37-39`, `ConstraintsComposite.py:51-53`;
  `core/__init__.py:1-6` pulls these). Deferred `mesh→core`
  (function-body, `mesh/_mesh_structured.py:562-567`). The system
  only survives because eager-`core→mesh` + deferred-`mesh→core`
  terminates. Reparenting `viz`/`mesh`/`FEMData` Selections onto a
  `core` ABC flips a deferred edge **eager** → `ImportError` at
  `import apeGmsh`. An AST cycle-detector cannot catch this (the cycle
  is already static); only the eager/deferred polarity matters.
- **FP-2:** the two geometry `Selection`s are structurally
  irreconcilable (`.tags()` method vs `.tags` property; `list`
  subclass vs frozen `__slots__`). No cross-class `@final` identity
  test is possible. Do not merge/reparent them.
- **FP-3:** mesh box (closed) vs results box (half-open) already
  diverge on `main`. S2 is a *reconciliation* (a decided behavior
  change), not a relocation.
- **FP-4:** `FEMData` does not call `_resolve_target`; it has a
  structurally different resolver with a **deliberate, documented**
  node-vs-element swallow asymmetry (node path catches `KeyError`
  only — `mesh/FEMData.py:411-419`; element path catches
  `(KeyError, ValueError)` — `:800-811`). S1 unifies Loads+Masses
  only; this asymmetry is a correctness invariant and is **not**
  touched.
- **T15:** `in_box` has three irreconcilable semantics — geometry
  `gmsh.getEntitiesInBoundingBox` (`viz/Selection.py:1051`, no
  half-open knob expressible) vs node-coord vs centroid containment.

### The architecture that survives (keystone-verified)

A **new leaf module `apeGmsh/_chain.py`** holds a `SelectionChain`
mixin. **New per-domain chainable types** subclass it. The **legacy
`Selection` classes are left byte-unchanged as terminals**.

Keystone, verified at source (this is why it is import-safe):

- `core/_selection.py:1-26` — only stdlib/numpy/gmsh at module level;
  `_Queries` is `TYPE_CHECKING`-only. Runtime-package-clean.
- `core/__init__.py:1-14` — imports `Part/Model/_parts_registry/
  Constraints/Loads/Masses` only; **does not import `_selection`**. A
  sibling leaf is therefore reachable without pulling the eager
  `core→mesh` chain.
- `mesh/_mesh_structured.py:562-567` — a **shipped, load-bearing**
  deferred (function-body) `from apeGmsh.core._selection import …`.
  The `.select()` hooks use this exact idiom.
- Empirically: default `import apeGmsh` →
  `C:\Users\nmora\Github\apeGmsh\src\…` (main repo); with
  `PYTHONPATH=<worktree>\src` → the worktree copy. Gates must set
  `PYTHONPATH` and assert `apeGmsh.__file__` is the worktree, or a
  green is a false negative.

Result: daisy-chaining + one shared engine + naming forced by
`__init_subclass__`, with **no eager edge added** and **legacy
Selections untouched** (zero caller churn; facade preserved).

---

## 4. Ratified product decisions

- **R3 — GeometryChain is IN the family.** Same verb names, same set
  algebra, daisy-chainable with mesh/results. But it is the **entity
  family**: `in_box` → `gmsh.getEntitiesInBoundingBox` (BRep
  bbox-**CONTAINMENT** — the whole entity bounding box must lie inside
  the query box, the box expanded by `Geometry.Tolerance`≈1e-8; this is
  *closed*, **not** an intersect and **not** half-open); it cannot
  honor the half-open/`inclusive=` knob and **raises `TypeError` if
  `inclusive=` is passed**. Honest family-typed spatial contract.
- **R2 — naming enforcement = `__init_subclass__` (definition-time)
  AND the CI contract test.** A chainable subclass missing/renaming a
  verb fails at import; the CI test additionally asserts per-family
  behavioral laws (`__init_subclass__` cannot).
- **R4 — box canonical = half-open** (locked); `mesh/_mesh_filters.py`
  moves closed→half-open to match the already-correct
  `results/_composites.py:189`. `inclusive=True` restores closed for
  point-family callers.

---

## 5. Architecture

```
apeGmsh/_chain.py                     [NEW — leaf; imports: stdlib/typing only]
  class SelectionChain:
    # OWNS (identical across all 4 domains):
    #   __or__/__and__/__sub__/__xor__ + union/intersect/difference
    #   one dedup law = insertion-order (pinned in S0b)
    #   chain protocol: every verb returns type(self)(new_items, _engine=)
    #   __init_subclass__: assert the public verb NAMES are present
    #   FAMILY: ClassVar[str] in {"entity","point"}
    # ABSTRACT per-family hooks (contract differs by FAMILY):
    #   _spatial_box / _spatial_sphere / _spatial_plane / _resolve_names
    #   _coords_of / _wrap / _materialize / _atoms

core/_resolution.py   [NEW — leaf] shared tier resolver for Loads+Masses (S1)
core/_spatial.py      [the numpy kernel; mesh/_mesh_filters.py is already
                       package-import-clean and becomes/owns this]   (S2)

# per-domain chainables (each imported DEFERRED by its host)
core/_selection.py  : GeometryChain(SelectionChain)  FAMILY="entity"
                      in_box -> gmsh BRep; inclusive= -> TypeError
                      legacy Selection(list) UNCHANGED (terminal)
mesh/_node_chain.py : NodeChain(SelectionChain)      FAMILY="point"
mesh/_elem_chain.py : ElementChain(SelectionChain)   FAMILY="point"
results/_result_chain.py : ResultChain(SelectionChain) FAMILY="point"
```

T15 is resolved honestly: the base owns chaining + set-algebra + verb
*names*; the spatial verbs are per-family hooks; the contract test
asserts **per-family laws** (entity: gmsh-stable, `inclusive=` raises;
point: half-open + `inclusive=` flip) and **never** cross-family
identity.

`.select()` is added to `NodeComposite`/`ElementComposite`/
`MeshSelectionStore`/results `_SelectionMixin` and to the geometry
entry, each via a **deferred** import. Legacy public entry points keep
returning legacy types unchanged. `.select()` is additive surface.

---

## 6. Phase plan

Order: **S0a → S0b → (S1 ∥ S2) → S3 → S5**. S0a/S0b are blocking.
S1 and S2 are independent of each other and of all product residuals.

Verification protocol for every gate (non-negotiable, see §3 keystone):

```
$env:PYTHONPATH      = "<worktree>\src"
$env:LADRUNO_OPENSEES_QUIET = "1"
C:\Users\nmora\venv\opensees_venv\Scripts\python.exe -c "import apeGmsh; print(apeGmsh.__file__)"   # MUST be the worktree
C:\Users\nmora\venv\opensees_venv\Scripts\python.exe -m pytest <targets> -q
```

### S0a — import-DAG eager/deferred polarity lock + runnable spike (S)

- Guard test: assert the polarity of the load-bearing cross-package
  edges (`core→mesh` eager allowed; `mesh→core`, `viz→core` deferred
  only; `core/__init__.py` does not import `_selection`/`_chain`/
  `_spatial`/`_resolution`). Fails if any deferred edge flips eager.
- Spike: minimal `apeGmsh/_chain.py` + one `NodeChain` + deferred
  `NodeComposite.select()`; then `import apeGmsh` on the real path.
- Gate: `import apeGmsh` clean, `apeGmsh.__file__` = worktree; polarity
  guard green; contract tests green.
- Rollback: 3 additive symbols, zero edits to existing code.
- Cannot silently regress: the polarity invariant is now CI-locked;
  FP-1 cannot be reintroduced without a red build.
- **Fallback:** if `import apeGmsh` fails with the spike on the real
  path → adopt the red pure-function form (`_chain_ops.py` functions +
  CI contract test only, no mixin/`__init_subclass__`). The spike is
  the decider; no re-adjudication.

### S0b — characterization battery on untouched main (M)

Pin **current** behavior so S1/S2 become reviewed pin-flips, not
silent drift: mesh box closed vs results box half-open; mesh-selection
`add_nodes/add_elements/filter_set` boundary counts; FEMData node-path
`ValueError`-not-swallowed vs element-path swallowed; `np.int64`
dimtag through `fem.nodes.get`; multi-target broker union ordering
(`mesh/FEMData.py:484-488` insertion vs `results/_composites.py:306`
sorted vs `viz/Selection.py:207-209`); viz `physical=` vs `label=`
collision; `selection=` on import-origin (`from_msh`/MPCO/native) fem.
Gate: 100% green on `main` HEAD, no production edits.

### S1 — unify Loads + Masses resolver only (M)

`core/_resolution.py` = one shared engine (incl. the `__ms__` tier);
`LoadsComposite._resolve_target` + `MassesComposite._resolve_target`
delegate. **Do not touch** `FEMData._resolve_one_target` /
`_resolve_one_elem_target` (documented swallow asymmetry) or
`_helpers._resolve_string`. No byte-parity tautology test.
Gate: `test_resolution_contract.py` + `test_target_resolution.py`
unchanged & green; S0b Loads/Masses pins unchanged.

### S2 — mesh box → half-open + `inclusive=` escape (S/M)

Change `mesh/_mesh_filters.py:62-66` closed→half-open (matches
`results/_composites.py:189`); `:163` `elements_in_box` inherits via
its delegation. Wire `inclusive=True` (restores closed) at the mesh
selection entry. Update S0b mesh-box pins **in the same commit** (the
diff is the decision). Gate: results pins unchanged; `inclusive=True`
proven to restore closed; contract tests green.

### S3 — the chainable family (L)

Promote the S0a spike: `SelectionChain` mixin (chaining + insertion-
order set-algebra + verb-name `__init_subclass__` + `FAMILY` + abstract
hooks); `NodeChain`/`ElementChain`/`ResultChain`/`GeometryChain`
(entity family, `in_box`→gmsh, `inclusive=`→`TypeError`). Legacy
`core/_selection.Selection` & `viz/Selection.Selection` byte-unchanged
terminals. Hosts get `.select()` via deferred import. Per-family
contract test (never cross-family identity). Gate: S0a polarity guard
green every commit; legacy Selection characterization unchanged; new
chains daisy-chain across all 4 levels in a smoke test.

### S5 — fail-loud sweep (S/M) — THREE distinct items

S0b/S3 characterization pinned three current silent-wrong paths; S5
makes each fail loud (each its own pin-flip + regression test):

1. **Results `selection=` on import-origin fem** — `from_msh`/MPCO/
   native produce `mesh_selection=None` (`mesh/FEMData.py:1119,1124`).
   `results/_composites.py:294-301` already raises `RuntimeError`
   here — S5 keeps/locks it loud (regression test), mirroring the
   already-correct `results/capture/spec.py:936,971` pattern.
2. **Loads `__ms__` consumer** — `LoadsComposite._target_nodes`
   (`~:1052-1053`) does `if info is None: return set()`, silently
   binding a load to nothing. Convert to a raise. (This is the
   `__ms__` *consumer*, distinct from the S1 `core/_resolution.py`
   resolver — the guard belongs in `_target_nodes`, not the shared
   resolver.)
3. **Clip-silent element centroids** — `results/_composites.py`
   `_element_centroids` uses `np.clip` to map a missing connectivity
   node id to the last node (silent centroid corruption) and backs
   the *existing* `results.elements.in_box/nearest_to/...` helpers.
   Make it fail loud (the new chain centroids are already fail-loud;
   this fixes the legacy helper). Flagged via a spawn-task chip
   during S3c.

---

## 7. Residual rulings (for the record)

1. Shared chainable base vs pure functions → **shared base**, gated by
   the S0a spike (engineering call; keystone-confirmed import-safe).
2. Naming enforcement → **`__init_subclass__` + CI test** (ratified).
3. GeometryChain inclusion → **in, entity family, `inclusive=` raises**
   (ratified).
4. S2 direction → **mesh moves to half-open** (locked).

---

## 8. Invariants that must hold at every commit

- `tests/test_resolution_contract.py` + `tests/test_target_resolution.py`
  green (20 tests; baseline confirmed on the worktree).
- No public API signature/behavior change except the deliberate,
  S0b-pinned `g.mesh_selection.in_box` closed→half-open (S2).
- No deferred `mesh→core`/`viz→core` import becomes eager (S0a guard).
- `apeGmsh.__file__` under the worktree for every in-process gate
  (editable install otherwise resolves to the main repo).
- Delivery is via PR onto `origin/main` (worktree → PR).

---

## 9. Deferred / follow-on (decisions for after S3)

S3 delivered the full daisy-chainable family at all four levels with
legacy terminals byte-unchanged. Three items were consciously deferred
(flagged, never silently skipped) and need a product/scope decision:

1. **Persistence parity — `.save_as(name)`.** A chained selection at
   the broker/mesh/results levels is currently query-only; it cannot
   be registered so it round-trips as a named `selection=` through the
   FEMData snapshot. This was explicitly deferred to "S3 with real
   usage"; real usage now exists. Decision: add a detached
   `.save_as(name)` store (registers into the `mesh_selection`-style
   store, round-trips via HDF5) vs leave the broker/results levels
   query-only (named persistence stays a pre-mesh `g.mesh_selection`
   author-time concern).
2. **Results sub-composites `.select()`.** `results.nodes`/`elements`
   have `.select()`; the five element sub-composites
   (`gauss`/`fibers`/`layers`/`line_stations`/`springs`) do not — each
   has a distinct reader/slab and `fibers`/`layers` carry extra
   terminal kwargs (`gp_indices=`/`layer_indices=`). The pattern is
   uniform/extensible; it needs per-terminal kwarg forwarding. Decide:
   a follow-on sub-phase (S3f) vs a tracked task.
3. **`g.mesh_selection.select()` name-seed — DONE for the one clean
   surface; PG/label direct-seed reported, not reimplemented.**
   S3d delivered `ids=`/full-universe seeding + spatial daisy-chain.
   The name-seed follow-on adds `select(name=N)` for an **existing**
   `g.mesh_selection` set: it **delegates verbatim** to the surfaces
   already on `MeshSelectionSet` — `get_tag` (the `_sets` name→tag
   lookup) + `get_nodes`/`get_elements` — via the private
   `_seed_ids_by_name`; **no new resolver**, and `name=` only *reads*
   `_sets` (the locked no-registration invariant still holds, now
   covered for the name path too). `select(name=N).<spatial>` is
   id-for-id `filter_set` over set `N`, proven for an `add_nodes`-built
   set **and** a `from_physical`-built set in
   `tests/test_mesh_selection_chain_name_seed.py`. An unknown name (or
   a node-set name asked at the element level) fails loud — never a
   silent empty / full-universe seed (resolution-contract Rule 6);
   `ids=`+`name=` together raises.

   Seeding *directly* from a raw gmsh PG name or an apeGmsh label was
   investigated and **consciously reported, not reimplemented**:
   `MeshSelectionSet` has no non-registering, non-reimplementing
   resolver for either. Its only PG bridge, `from_physical`,
   *registers* a set + allocates a tag (which `select()` must not do)
   and is node-only; label/geometry resolution lives off
   `MeshSelectionSet` entirely, and
   `tests/test_resolution_contract.py::test_meshselection_name_is_not_a_helpers_tier`
   locks that a mesh-selection name is deliberately *not* a
   geometry-resolver tier. The supported existing-surface route for
   those is the two-step `from_physical(dim, "PG", ms_name="foo")` /
   `from_geometric(sel, name="foo")` **then** `select(name="foo")`.
   Persistence (`.save_as` / round-trip as `selection=`) stays tied
   to (1) and remains out of scope.
