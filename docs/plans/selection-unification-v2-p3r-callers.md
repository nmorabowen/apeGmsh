# §6.3 — P3-R caller-migration contract (the m6-gated artifact)

Committed precondition for the BREAKING, irreversible P3-R
(`selection-unification-v2.md` §6.2 P3-R + RED/BLUE m6). Derived
read-only at the **POST-P3-K** HEAD (`a696bb0`, branch
`guppi/sel-v2-p3k`: `_kernel/spatial.py` present, `_mesh_selection.py`
has no `_delegate(`). Every `file:line` re-derived at source
(ripgrep-backed, never `git grep -E`). Head-verified the load-bearing
STOP-1/STOP-2 at source. **P3-R must not start until this is reviewed.**

## 0. Mandated P3-R actions the §6.2 prose under-specified (fold these in)

These are **resolvable, mandatory, same-commit P3-R actions** (not
blockers) the §6.2 P3-R scope text omitted. P3-R is infeasible without
them.

- **M-STOP-1 (FATAL) — sever `ElementComposite.select`→`self.get`
  FIRST.** `mesh/FEMData.py:962-973`: the aux-filter branch
  (`dim`/`element_type`/`partition`) computes
  `atoms = [int(e) for e in self.get(target,pg=,label=,tag=,dim=,
  element_type=,partition=).ids]` ("reuse `.get()` verbatim ...
  select(...) == get(...)"). P3-K's invisible 2-file scope correctly
  did NOT touch `FEMData.py`; SC-5 is **removal-coupled** and is P3-R's.
  Removing `ElementComposite.get` without first rewiring this breaks
  `fem.elements.select(element_type=|dim=|partition=)` — *the migration
  target itself* — and cascades into proof file
  `test_pin_resolution_v2.py:287,290,312` and the SC-6 sites. **P3-R
  action (same commit, before/with the `get` deletion):** factor the
  `ElementComposite.get` filter body (`FEMData.py:740-782` Step-1/2/3)
  into a private `_filtered_groups(target,pg,label,tag,dim,
  element_type,partition)` helper; `select`'s aux-branch and (the now
  internal-only) filtered path both call it; the public `get`/`get_ids`/
  `resolve` wrappers are deleted, the helper stays. `NodeComposite.select`
  (`FEMData.py:308-315`) is verified clean (calls `_resolve_nodes`/
  `_intersect_partition`, never `self.get`) — node side needs no such
  rewire. Correct the §6.2 P3-K line "Severs … SC-5 …" → "Severs
  SC-1/SC-2/SC-4 (+SC-6 collapse); SC-5's `select→self.get` knot is
  **removal-coupled → resolved in P3-R** (§6.3 M-STOP-1)".
- **M-STOP-2 (SERIOUS) — delete `core/Model.py:89-90`.** `Model.__init__`
  eagerly runs `from apeGmsh.viz.Selection import SelectionComposite;
  self.selection = SelectionComposite(parent=parent, model=self)` on
  **every** `apeGmsh(...)`/`Part(...)`. Hard-removing `SelectionComposite`
  without deleting these lines → `ImportError` at `import apeGmsh;
  apeGmsh(...)` — total break. **P3-R action:** delete `Model.py:89-90`
  + the stale `:84-88` comment; the `g.model.selection` surface is gone
  (superseded by `model.select(...)→EntitySelection`); zero other PROD
  callers of `self.selection` besides `Model.viewer()` (SC-7).
- **M-STOP-3 (SERIOUS) — 3 `fem.elements.resolve` rewire sites, not 1.**
  §6.2 named only `_mesh_selection.py:267` (`_centroid_map_result`).
  Also PROD: `results/_composites.py:139` (`_element_centroids`),
  `results/_composites.py:264`. (`results/_result_chain.py:196` is
  deleted with the chain module — no rewire.) All 3 rewire to direct
  per-type group iteration (the `_centroid_map_element` pattern,
  `_mesh_selection.py:203` — `for grp in self._engine._groups.values()`,
  same `id_to_idx`, same fail-loud `KeyError`, same
  `node_xyz[rows].mean(axis=0)`), **not** `fem.elements.select(
  element_type=)` (which re-enters M-STOP-1). Bound to M-STOP-1's helper.
- **M-MINOR-a** `cuts/_defs.py:201` uses `fem_eids.size` → the
  `.select(...).ids` replacement MUST be `np.asarray(fem.elements.select(
  pg=elements_pg).ids)` (`.ids` is a `list`, no `.size`). Not a bare
  `.get→.select` swap.
- **M-MINOR-b** `core/_model_queries.py:644-667` `_string_ref_to_dimtags`
  becomes an orphan once `select` (`:669-794`) is removed (its only
  callers `:762,774`). Delete it same-commit (orphan-cleanup of P3-R's
  own removal).
- **M-CORRECTION (REVISED — RED-3 CLAIM-A KILLED, BLUE-adjudicated +
  head-reverified at HEAD; supersedes the prior wording AND §6.2's
  ":15 keep Plane,Line").** The symbol *definitions* `_select_impl`
  (`core/_selection.py:166`), `Plane` (`:57`), `Line` (`:94`) **stay
  defined** — load-bearing for the retained `Selection.select`
  (`:513` calls `_select_impl`), `EntitySelection`, and the P2-G chain
  (do **not** delete the defs). **Separately**, P3-R rewrites the
  now-stale module-local import `core/_model_queries.py:15` to
  **exactly** `from ._selection import Plane`. Source-proven (census +
  RED-3 + BLUE + head, 4 independent HEAD checks): after removing
  `line`(:603-615)/`_select_all`+`select_all*`(:621-642)/
  `_string_ref_to_dimtags`(:644-667)/`select`(:669-794), the **only**
  surviving runtime ref is `Plane` (retained `_Queries.plane`
  :561-601, refs :585/:596/:597 — `plane` is **NOT** in removal
  scope); `Selection` (refs only :622/:626/:790 + lazy `select`
  annotations), `_select_impl` (only :792), `Line` (only :615) all
  become dead imports. Nothing imports those four names *from*
  `_model_queries`. Both prior prose lines were wrong (§6.3 left
  `_select_impl`+`Line` dead = SERIOUS; §6.2 left `Line` dead).
- **M-STOP-3 PRECONDITION (hardening — RED-3 CLAIM-B).** The
  `resolve(element_type=name)`→direct-iteration equivalence holds only
  under "`ElementComposite._groups` is one-`ElementGroup`-per-type-code"
  (verified all 4 constructors: `_femdata_native_io.py:102`,
  `_femdata_h5_io.py:904`, `_femdata_mpco_io.py:94`,
  `_fem_factory.py:151`; `_groups` FEMData.py:601; `types` :667-669).
  The 3 rewires iterate `fem.elements._groups.values()` **directly**
  (do NOT re-implement `resolve_type_filter`), emit
  `(grp.ids, grp.connectivity)` per group preserving `_groups` dict
  order (matches the deleted `for type_info in fem.elements.types`);
  the single-type site `results/_composites.py:254-265`
  (`_element_ids_of_type`) **retains its existing
  `element_type not in available` guard** (:261-263) so the
  absent-type→empty contract is unchanged.
- **M-NOTE-save_as (clarifying pin — RED-3 CLAIM-D).** `mesh/
  MeshSelectionSet.py` has **no `def save_as`** (HEAD-verified; the
  retained generic surface is `add()`@:147 + `select()`@:715 only;
  removal set within the file = `add_nodes`@:199/`add_elements`@:259/
  `from_geometric`@:516). The `.save_as` in §5 (cascade-D) / §6.2
  SC-12 is a method of the **returned `MeshSelection`** terminal, not
  of `MeshSelectionSet` — no plan line is defective; this pin only
  prevents inferring a `MeshSelectionSet.save_as`.
- **M-NOTE-oracle-shape (executor guard, reinforces §5 — RED-1
  advisory).** The proof-file rewrites at oracle `:287/:290/:312`
  (`test_pin_resolution_v2.py`) and the element sites in §5 use
  `.groups()`/`.result()` (GroupResult-shaped, matching the deleted
  element-`get` body) — **never** `.ids`. Do not "simplify" to `.ids`.
- **M-NOTE-from_physical (scope clarification — head-reverified at
  HEAD).** §6.2/§6.3's "`g.mesh_selection.add_*/from_*`" is shorthand
  for the **justified** targets only: `MeshSelectionSet.add_nodes`
  (:199-258) + `add_elements` (:259-337) (SC-11 `_mesh_filters`
  silent-row-0 family) and `from_geometric` (:516-584) (SC-12
  `viz.Selection.to_mesh_*` consumer — its dependency is removed).
  **`MeshSelectionSet.from_physical` (:476-514) is RETAINED** — it
  depends only on retained surfaces (`gmsh.model.*` PG APIs +
  `self._alloc_tag`/`self._store_node_set`/`np`), has no removed
  dependency, and no stated removal justification; removing it would be
  unjustified scope-creep on the irreversible phase. This matches the
  RED-2/BLUE removal-set enumeration ({add_nodes, add_elements,
  from_geometric}), which already excluded `from_physical`. (Same
  prose-too-literal class as the `_model_queries.py:15` correction.)
- **M-NOTE-G7-cascade (§6.3 §5 reconciliation — head-adjudicated at
  HEAD after the executor STOP-report; SC-10 anticipated this
  under-enumeration: "an order of magnitude beyond ~75 files; re-derive
  by ripgrep").** Executor's full-suite scan = 278 failed/5 err across
  33 files, **all test-side (zero PROD defects** — G1–G6 surgery
  source-confirmed sound; e.g. `cuts/_drift.py:446` correctly
  `.select`). Dispositions (the §5 A/B/C/D categories stand; these
  refine 3 misclassified/omitted items, behaviour-preserving, no
  PROD/FROZEN edits):
  - **`test_p2i_parity.py` → DELETE** (already pre-ratified §5-B
    "legacy↔v2 parity vacuous once legacy gone; P3-K oracle proved
    equivalence"). Executor done.
  - **`test_p2g_parity.py` → REWRITE to v2-only (NOT "surface-only"
    [§5-B misclassified it], NOT delete).** Source-proven: it is the
    same vacuous-once-legacy-gone legacy↔v2 parity as p2i (28
    `queries.select`/`line` refs), BUT it *uniquely* pins v2-OWNED
    behaviour the RETAINED `EntitySelection.crossing_plane` engine + the
    §6.1 STOP-1 point-family `TypeError` fail-loud — NOT redundantly
    covered post-P3-R (the rewritten `test_pin_spatial_v2.py` is now
    only the SC-11 `_mesh_filters` flip). Rewrite: keep each
    `g.model.select(seed).crossing_plane(spec,mode=m)` assertion with
    its expected `(dim,tag)` set **frozen as a literal** (the
    P2-G-proven value — the legacy oracle is gone, exactly the
    proof-file freeze pattern); keep the STOP-1 point-family `TypeError`
    pin verbatim; drop the now-impossible legacy `queries.select`/`line`
    comparison half. Head owns/verifies this (behaviour-invariant pin,
    like the proof files).
  - **`test_selection_filters.py` (33 tests) + the orphaned
    `g.model.selection.select_*` test sites → RETIRE (delete) with a P4
    capability-gap note.** Source-proven: `SelectionComposite.select_*`
    exposed a rich filter grammar (`labels=`fnmatch, `kinds=`,
    `length/area/volume_range=`, `predicate=fn`, `exclude_tags=`,
    `physical=`, `at_point=`) via the `_apply_filters` engine;
    `EntitySelection` (the ratified `g.model.select(...)→EntitySelection`
    successor) has **no** equivalent (only spatial verbs + set-ops +
    `to_label/to_physical/to_dataframe`). `SelectionComposite` removal
    is **already ratified** §6.2 (census/RED-2 zero PROD callers besides
    `Model.py:89-90`+`Model.viewer():260`/SC-7). Head-ratified via the
    **SC-12 precedent** (a removed-surface user capability with no v2
    successor → P4-documented gap, *not* an internal blocker) — **NOT
    owner-re-ratified** (owner ratified the class removal; SC-12 is the
    established disposition class). **Owner-informed loudly** (this note
    + the P3-R PR body + the memory note): §6.3 §5 omitted the
    `select_*` filter-grammar cascade, and the rich filter API has **no
    v2 successor** (a P4 capability gap, sibling to SC-12's
    `from_geometric`). Not papered over.
  - **Stub/mock cascade (~131 broker-stub + ~7 mock-`_groups` failures)
    → IN-SCOPE mechanical (A/M-STOP-3 consequence; SC-10).** Per-file
    test-doubles that mirrored the removed `fem.*.get` must mirror
    `.select(...)` (node→`.ids/.coords`; element→`.groups()/.result()/
    .resolve()`, M-NOTE-oracle-shape) **byte-faithfully** (the verified
    pattern: stub `.select(...)` ≡ `return self.get(...)`); mocks the
    M-STOP-3 direct-iteration touches must expose `_groups`. Mirror
    only — never weaken/skip an assertion.
  - **P4 (§6.2 P4 scope addition):** document the `SelectionComposite.
    select_*` filter-grammar capability gap beside the SC-12
    `from_geometric` gap.
- **RED-2 MINOR stale-strings (non-blocking).** The stale legacy-idiom
  error/docstring strings (`core/_model_geometry.py:1053`;
  `core/_selection.py:353/:545/:586/:938/:1161`;
  `mesh/_mesh_structured.py:124/:204/:423`) are **P4 /
  optional-same-commit cosmetic** — not PROD callers, not a blocker.

## 1. Zero-PROD-caller census (the P3-R gate) — summary

PROD-load-bearing src callers per removed surface (full file:line table
in the pre-flight transcript; re-run ripgrep at P3-R start as the gate):

- Package exports `Selection`/`SelectionComposite`: drop 4 export lines
  (`apeGmsh/__init__.py:79,187,188`; `viz/__init__.py:2,4`). Classes
  `viz.Selection`/`core/_selection.Selection` **RETAINED** (SC-8/R-v2-8).
  PROD `SelectionComposite` constructors = **2** (`Model.py:89,90` —
  M-STOP-2). Both viewers construct `viz.Selection` via **deferred
  in-method import** (`model_viewer.py:1669-71`, `mesh_viewer.py:
  1454-56`) → survive the export drop, **no repoint**.
- `queries.select`/`select_all*`/`queries.line` bodies
  (`_model_queries.py:603-642,669-794`): **0** PROD src callers outside
  the module.
- `SelectionComposite` class / `g.model.selection`: **0** PROD callers
  besides `Model.py:89-90` + `Model.viewer():260`.
- `fem.*.get/get_ids/get_coords/resolve`: **15** PROD sites (§2).
- chain `results.*.select(...).values()`/`ResultChain.get`: **0** PROD.
- `g.mesh_selection.add_*/from_*`: **0** PROD (SC-12; `from_geometric`
  uses `viz.Selection.to_mesh_*` — both ends removed, P4-documented gap).
  Generic `MeshSelectionSet.add()`/`.select()` **RETAINED**.
- 4 chain modules + `GeometryChain`: DEF-only; importers = the ~10
  chain-test files (top-level) + the 3 deferred `engine_for` sites.
- DYN-clean: no `_COMPOSITES`/getattr/importlib/pickle/HDF5/`__getattr__`
  reference to any removed class name (verified).

## 2. PROD caller migration table (15 sites)

Identity proofs (cited once): **P-NODE** `NodeComposite.select`
(`FEMData.py:308-315`) calls the same `_resolve_nodes`+
`_intersect_partition` as `get`; **P-ELEM-IDS** `select` pure-name path
(`:950-961`) reuses `_resolve_elem_ids`; **P-GROUPRESULT**
`MeshSelection._materialize_element` builds `GroupResult` by
`self._engine._groups.values()` insertion-order + `np.isin` —
byte-identical to `ElementComposite.get` (`:760-782`); `.groups()`/
`.result()`/`.resolve()` identical (`_kernel/payloads.py:306-345`, same
mixed-type `TypeError`); **P-COORD** `MeshSelection.coords`==
`NodeResult.coords` co-indexed; **m3** the `try/except (KeyError,
ValueError)` must wrap the `.select(...)` call (resolution raises there).

| # | file:line | current | replacement | note |
|---|---|---|---|---|
|1|`cuts/_drift.py:446`|`fem.nodes.get_ids(pg=)`|`fem.nodes.select(pg=).ids`|P-NODE; `np.asarray(...).ravel()` already wraps|
|2|`cuts/_planes.py:214`|`fem.nodes.get_coords(pg=)`|`fem.nodes.select(pg=).coords`|P-COORD|
|3|`cuts/_defs.py:201`|`fem.elements.get_ids(pg=)` + `.size`|`np.asarray(fem.elements.select(pg=).ids)`|P-ELEM-IDS; **M-MINOR-a** wrap|
|4|`cuts/_polygons.py:183`|`fem.nodes.get_coords(pg=)`|`fem.nodes.select(pg=).coords`|P-COORD|
|5|`opensees/_internal/build.py:323`|`fem.elements.get(pg=)` (iter groups)|`fem.elements.select(pg=).groups()`|P-GROUPRESULT; m3 wrap `.select`|
|6|`opensees/_internal/build.py:345`|`fem.nodes.get(pg=).ids`|`fem.nodes.select(pg=).ids`|P-NODE; m3|
|7|`opensees/_orientation.py:359`|`fem.elements.get(pg=)` (iter)|`fem.elements.select(pg=).groups()`|P-GROUPRESULT; m3|
|8|`opensees/node.py:314`|`fem.nodes.get(pg=)` → `.ids/.coords`|`fem.nodes.select(pg=)` → `.ids/.coords`|P-NODE+P-COORD; m3|
|9|`results/transcoders/_recorder.py:861`|`fem.elements.get(tag=).resolve()`|`fem.elements.select(tag=).result().resolve()`|P-GROUPRESULT; oracle README ratifies|
|10|`results/transcoders/_recorder.py:870`|`fem.nodes.get_coords(node_a)`|`fem.nodes.select(node_a).coords`|P-COORD (target path)|
|11|`results/transcoders/_recorder.py:871`|`fem.nodes.get_coords(node_b)`|`fem.nodes.select(node_b).coords`|P-COORD|
|12|`results/_composites.py:139`|`fem.elements.resolve(element_type=)`|direct per-type group iteration (M-STOP-3)|**not** `select(element_type=)` (M-STOP-1)|
|13|`results/_composites.py:264`|`fem.elements.resolve(element_type=)`|direct group iteration (M-STOP-3)|M-STOP-1/3|
|14|`mesh/_mesh_selection.py:267`|`fem.elements.resolve(element_type=)`|direct group iteration ≡ `_centroid_map_element`|named SC-6 site; M-STOP-1/3|
|15|`results/_result_chain.py:196`|`fem.elements.resolve(element_type=)`|— deleted with module|N/A|

#1–#11 mechanically behaviour-identical with the stated wraps/order
proofs; #12–#14 bound to the M-STOP-1 `_filtered_groups` helper.

## 3. Chain-deletion bundle

Delete files: `mesh/_node_chain.py`, `mesh/_elem_chain.py`,
`results/_result_chain.py`, `mesh/_mesh_selection_chain.py`; delete
`GeometryChain` class (`core/_selection.py:852`; file stays — holds
retained `Selection`/`EntitySelection`/`Plane`/`Line`/`_select_impl`).

Relocate (both verified pure — `typing` only, NO `_kernel`/numpy/
`SelectionChain` need; → **zero new BASELINE triple**):
- `_ResultChainEngine`+`engine_for`+`VALID_LEVELS`+`_ENGINE_CACHE_ATTR`
  (`_result_chain.py:50-132`) → new `results/_result_engine.py`;
  repoint 3 deferred sites `_composites.py:614,686,907`
  (`from ._result_chain import engine_for` → `from ._result_engine
  import engine_for`).
- `_LiveMeshEngine`+`engine_for`+`VALID_LEVELS`+`_ENGINE_CACHE_ATTR`
  (`_mesh_selection_chain.py:74-155`) → new `mesh/_live_engine.py`;
  repoint `MeshSelectionSet.py:814`.

Import-DAG (`test_import_dag_polarity.py`): **remove exactly 4 BASELINE
triples** `:84` `(mesh,_kernel,mesh/_elem_chain.py)`, `:87`
`(…,_mesh_selection_chain.py)`, `:88` `(…,_node_chain.py)`, `:89`
`(results,_kernel,results/_result_chain.py)`; **add 0** (pure engine
modules). Spike `:197` `import_module("apeGmsh.mesh._node_chain")` →
`"apeGmsh.mesh._mesh_selection"`; `:200` `nodec.NodeChain.FAMILY` →
`<mod>.MeshSelection.FAMILY` (rename local var). Same-commit reviewed
diff (`test_import_dag_polarity.py:18-20`).

## 4. SC-6/7/8 — verified at HEAD

- **SC-6**: 3 PROD sites (M-STOP-3). Equivalent = `_centroid_map_element`
  pattern (proven same centroids/fail-loud at `_mesh_selection.py:184-221`).
- **SC-7**: `Model.viewer():260` `return self.selection.picker(**kwargs)`;
  `picker` (`viz/Selection.py:645-651`) builds `ModelViewer(parent=
  self._parent,model=self._model,…)` — no `Selection` class. Inline →
  `from apeGmsh.viewers.model_viewer import ModelViewer; p=ModelViewer(
  parent=self._parent,model=self,**kwargs); p.show(); return p`.
  `Model.preview` (`:289`) uses `preview_model`, **not** `.picker`/
  `.selection` — untouched. (+ M-STOP-2 deletes `Model.py:89-90`.)
- **SC-8**: both viewers construct `viz.Selection` via deferred
  in-method import → class **RETAINED**, export-only drop, **no
  repoint**. `EntitySelection._materialize` (`core/_selection.py:1368`)
  constructs retained `core/_selection.Selection` (R-v2-8). No
  retained-class contradiction at source.

## 5. Proof/test rewrite

- `test_target_resolution.py` (oracle, `.P3K.frozen`): 5 sites `:62,63,
  64,82,83` `fem.nodes.get(`→`fem.nodes.select(`; expected literals
  **byte-identical** to the frozen copy (P-NODE; `sorted(int(n)…)`
  absorbs list/ndarray). No `dim=`/aux → M-STOP-1 not triggered here.
- `test_pin_resolution_v2.py` (oracle, `.P3K.frozen`): 17 sites; node
  `.ids`→`select`, element→`select(...).groups()`/`.result()`; literals
  byte-identical to frozen. **`:287,290,312`** (`fem.elements.get(...,
  dim=)`) depend on M-STOP-1's `_filtered_groups` preserving the
  silent-empty post-filter (§3.1(a)) byte-identically — verify against
  frozen.
- `test_pin_spatial_v2.py` (SC-3 third file; NOT oracle-equal): Pin 1/2
  surfaces removed → rewritten/retired as the explicit reviewed
  legacy→fail-loud `_mesh_filters.py:159/215` diff (SC-11); P3-S adds
  new-idiom successors.
- `test_resolution_contract.py`: removal-safe (only `g.constraints.
  resolve`) → **byte-unchanged through P3-R** (§7 inv.1 / SC-3).
- ~75-cascade categories: **A** `fem.*.get/...` ~10 (`.get→.select`;
  element `dim=` blocked on M-STOP-1); **B** chain-class ~11 (8
  collection-error files MUST be rewritten/deleted same-commit:
  `test_fem_chain`, `test_geometry_chain`, `test_mesh_selection_chain`,
  `test_mesh_selection_chain_name_seed`, `test_p2i_parity`,
  `test_result_chain`, `test_result_chain_subcomposites`,
  `test_selection_idiom` [`_EXPECTED_CHAINS` 7→2, `_POINT_CHAINS` 5→1];
  + 3 surface-only: `test_p2g_parity`, `test_selection_dtype_contract`,
  `test_characterization_selection`); **C** `queries.select`/`line`/
  `select_all` (`test_selection.py` ~47, `test_selection_direction_
  filters.py` ~17, +~6 others → `model.select(...).crossing_plane(...)`
  P2-G verb); **D** `g.mesh_selection.add_*/from_*` (~2 →
  `.select(...).save_as`; `from_geometric` no v2 path — retire/xfail,
  P4 gap); **E** retained `results.*.get(component=)` ~30+ → **NO
  rewrite** (retained reader, §3.1(c)/SC-2). Genuine rewrite burden
  ≈ A+B+C+D ≈ 30–35 files + 2 oracle + `test_pin_spatial_v2` +
  `test_selection_idiom` locks. `test_p2i_parity` recommended **delete**
  (legacy↔v2 parity vacuous once legacy gone; P3-K oracle already
  proved equivalence) — reviewer call.

## 6. P3-R gate (unchanged from §6.2, + the M-actions)

Full suite green; **no legacy removal-target symbol importable** (assert
`ImportError`); `import apeGmsh; apeGmsh(...)` clean (M-STOP-2);
`fem.*.select(...)` works (M-STOP-1 helper in place);
`test_resolution_contract.py` byte-unchanged; the 2 rewritten proof
files behaviour-equal to `tests/_p3k_oracle/*.P3K.frozen` (literal
diff = resolution-call surface only); import-DAG BASELINE = −4/+0
same-commit; the `_mesh_filters.py:159/215` flip a reviewed
production+assertion diff (SC-11). P3-R stacks on `guppi/sel-v2-p3k`.
