# P3-K execution map (head-derived at HEAD `63a4312`, read-only Phase 0)

Contract: `selection-unification-v2.md` §6.2 P3-K (behaviour-INVISIBLE).
Every line:ref re-derived from source, not plan prose.

## Phase-0 refinement (RECONCILED into §6.2 before any write)

**Finding:** all ~10 chain-specific test files **top-level import** the
legacy chain modules (`tests/test_fem_chain.py:51-52`,
`test_geometry_chain.py:48-49`, `test_mesh_selection_chain.py:53-54`,
`test_mesh_selection_chain_name_seed.py:49`, `test_p2i_parity.py:56-65`,
`test_result_chain.py:65`, `test_result_chain_subcomposites.py:66`,
`test_selection_idiom.py:59-63` + its `_EXPECTED_CHAINS` 7-equality
lock `:95-96,133`, `test_p2g_parity.py`, `test_selection_dtype_contract.py`)
→ deleting the chain modules *in P3-K* would **collection-error** all of
them + force the BASELINE 4-triple delta + spike rewrite, i.e. a large
test-cascade inside the "invisible" PR — contaminating the invisibility
proof.

**Refinement (strictly more invisible; within ratified 3-unit scope —
not a re-ratification item):** **P3-K does NOT delete anything.** It
only makes `MeshSelection` self-contained (relocate the per-engine
bodies in + new `_kernel/spatial.py`; remove `_delegate()`). The 4 chain
modules + `GeometryChain` + the `_ResultChainEngine`/`_LiveMeshEngine`
adapters + `engine_for` stay **defined-but-now-dead** (still importable,
their own tests still pass byte-unchanged because the chain bodies are
left intact). **The chain deletion + adapter relocation + BASELINE
4-triple delta + spike rewrite + the ~10 chain-test-file disposition all
move to P3-R** (which already owns the removal + ~75-file cascade —
SC-10). P3-K's diff is then exactly **2 files** (`mesh/_mesh_selection.py`
+ new `_kernel/spatial.py`); the gate "4 proof files + ALL tests
byte-unchanged & green; full suite 5909/64/0" becomes a clean 2-file
true/false. This mirrors the P1-K precedent (pure relocation; removals
deferred) and R-v2-6.

## The relocation (P3-K writes — 2 files only)

### NEW `src/apeGmsh/_kernel/spatial.py` (pure leaf: numpy/stdlib only)
The 4 chains' `_spatial_box`/`_spatial_sphere`/`_spatial_plane` are
**byte-identical** (verified: `_node_chain.py:48-84`, `_elem_chain.py:124-160`,
`_result_chain.py:247-283`, `_mesh_selection_chain.py:265-301` — same
numpy formulas, half-open `[lo,hi)` default + `inclusive=` closed, closed
ball `<= r`, plane `|((c-p)@n̂)| <= tol` with radius/tol/normal
validation). Extract verbatim as pure array-mask functions on a coords
ndarray. Also a pure fail-loud centroid helper (the array core shared by
the 3 chains' `_centroid_map`: KeyError on a connectivity id absent from
`id_to_idx` — NOT `_mesh_filters`' silent row-0). No `apeGmsh.*` import
(may import `apeGmsh.fem` only if needed — none needed here).

### `src/apeGmsh/mesh/_mesh_selection.py` (collapse `_delegate()`)
`MeshSelection` already carries `_engine_kind()` (`:82-91`: live/result/
element/node) and `_level` (`:166-178`). Replace the `_delegate()`-routed
hooks with self-contained per-engine logic, **relocated verbatim** from
the 4 chains (preserve cache-attr names, iteration order, fail-loud
messages, deferred imports, dtypes), spatial masks via
`_kernel/spatial.py`:

| MeshSelection hook | source body (verbatim) |
|---|---|
| `_coords_of` node (broker) | `_node_chain.py:31-45` (`_row_map` cache `_apegmsh_chain_idrow` on engine=NodeComposite; `self._engine.coords`) |
| `_coords_of` element (broker) | `_elem_chain.py:53-117` (`_centroid_map` cache `_apegmsh_elem_centroid`; sibling NodeComposite via `NODES_REF_ATTR="_apegmsh_nodes_ref"`; **iterate `self._engine._groups.values()` insertion-order** — m4; fail-loud) |
| `_coords_of` result | `_result_chain.py:148-240` (`_fem()`; node `_node_row_map` cache `_apegmsh_rc_node_idrow` on the `_ResultChainEngine` adapter; element `_centroid_map` cache `_apegmsh_rc_elem_centroid` via `fem.elements.types`+`fem.elements.resolve` — SC-6 single site, P3-R rewires when `resolve` is removed) |
| `_coords_of` live | `_mesh_selection_chain.py:171-257` (`_live_nodes` via `self._engine.ms._get_mesh_nodes()`; element `_centroid_map` cache `_apegmsh_lm_elem_centroid` via `ms._get_mesh_elements(dim)`; skip `n<0` padding; fail-loud) |
| `_spatial_box/sphere/plane` | one impl: `c=self._coords_of(atoms); m=_kernel.spatial.X(c,...); return tuple(a for a,k in zip(atoms,m) if k)` + empty-atoms guard |
| `_materialize` node | `_node_chain.py:90-112` → `NodeResult(ids,coords)` (deferred `from .FEMData import NodeResult`) |
| `_materialize` element | `_elem_chain.py:166-200` → `GroupResult` of `ElementGroup` `np.isin`-masked, **iterate `self._engine._groups.values()` insertion-order** (deferred `from ._element_types import ElementGroup, GroupResult`) |
| `_materialize` result | `_result_chain.py:317-330` → **raise** RuntimeError (needs component) |
| `_materialize` live | `_mesh_selection_chain.py:312-348` → node `{tags,coords}` / element mask `{element_ids,connectivity}` dict |
| `values()` (`:280-312`) | repoint from `self._delegate().get(...)` → verbatim `ResultChain.get` body (`_result_chain.py:308-315`): `host=self._engine.host; return host.get(ids=list(self._items),component=component,time=time,stage=stage,**extra)` (the RETAINED typed reader — SC-2) |
| `__iter__`/`connectivity`/`groups`/`result` | already call `self._coords_of` / `self._delegate()._materialize()` — repoint `_delegate()._materialize()`→`self._materialize()` |
| remove | `_delegate()` (`:112-144`) + its 4 deferred chain imports |

**UNTOUCHED in P3-K (oracle / retained / deferred-to-P3-R):**
`mesh/_mesh_filters.py` (no chain consumes it — only `MeshSelectionSet.py:35`;
verified); `core/_selection.py` `Selection`/`EntitySelection._materialize`
(`:1359-1368` — R-v2-8 retained payload, standalone, no collapse) /
`GeometryChain`; `viz/Selection.py`; the 4 chain modules + adapters +
`engine_for`; `results/_composites.py:614/686/907` + `MeshSelectionSet.py:814`
(`engine_for` importers — modules still present); ALL test files; the
import-DAG BASELINE; every legacy removal-target symbol.

## P3-K gate (the invisibility proof — 2-file diff)
- `git diff --stat 63a4312 -- <4 proof files>` → **empty**.
- `git diff --name-only 63a4312` → exactly `mesh/_mesh_selection.py`
  (+ new `_kernel/spatial.py`).
- full suite == **5909 passed / 64 skipped / 0 failed** (zero delta);
  `test_import_dag_polarity` green **with BASELINE unchanged** (no
  deletions → no triple removed); all P0-C pins green; `import apeGmsh`
  WORKTREE-OK.

## Deferred to P3-R (added to its scope; §6.2 reconciled)
Delete the 4 chain modules + `GeometryChain`; relocate
`_ResultChainEngine`/`engine_for`→`results/_result_engine.py` (pure;
repoint `_composites.py:614,686,907`), `_LiveMeshEngine`/`engine_for`→
`mesh/_live_engine.py` (pure; repoint `MeshSelectionSet.py:814`);
same-commit BASELINE remove triples `test_import_dag_polarity.py:84`
(`mesh/_elem_chain.py`), `:87` (`mesh/_mesh_selection_chain.py`), `:88`
(`mesh/_node_chain.py`), `:89` (`results/_result_chain.py`); rewrite
spike `:197,200` (`_node_chain`/`NodeChain`→`_mesh_selection`/
`MeshSelection`); dispose the ~10 chain-test files
(rewrite-to-MeshSelection where behaviour-bearing / delete where
chain-identity-only; `test_selection_idiom._EXPECTED_CHAINS` 7→2
{EntitySelection, MeshSelection}); rewire the one
`fem.elements.resolve` centroid site (SC-6) onto a non-removed path.

## P3-K oracle (Phase 3, committed)
`tests/_p3k_oracle/`: capture script + JSON of resolved id-sets/payloads
for the `test_target_resolution.py`/`test_pin_resolution_v2.py`
scenarios (run against the legacy oracle surface still present at P3-K),
the behaviour-equality reference for P3-R's proof-file rewrites.
