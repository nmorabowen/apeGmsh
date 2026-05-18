# Selection / Resolution Unification v2 — Hardened Plan

Status: **DESIGN RATIFIED — implementation not started.** Supersedes the
scope conclusions of `docs/plans/selection-unification.md` (v1).

Author: head-engineer adjudication of a 4-wave red/blue adversarial
exercise (3× RED Opus + 1× BLUE Opus + adjudication), 2026-05-17/18.

> [!important] What v1 was, and why v2 exists
> v1 (`selection-unification.md`, S0a–S3f shipped, S5 pending) delivered
> a verb-consistent `.select()` chain **additively, beside the legacy
> surface**, under a *non-negotiable backward-compatibility constraint*.
> That constraint is exactly what produced the debt the project owner is
> now resolving: ~11 divergent terminal types, two classes both named
> `Selection`, 4 name-resolvers, 6 spatial copies. The owner has
> **explicitly removed backward-compat** and ratified **full removal**.
> v2 is the cycle-breaking relayer + the single idiom + the gated
> removal that v1's constraint forbade. The v1 doc's *technical hard
> truths* (FP-1/FP-2/FP-4) remain true at source and are re-stated here
> as HT1–HT10 on durable source grounds — but its *conclusions*
> ("therefore do not remove", "facade preserved") were
> backward-compat-driven and no longer bind.

Every load-bearing claim below was verified at source during the
exercise; `file:line` references are given so they can be re-checked.

---

## 1. Goal (ratified)

One fluent idiom, two terminals, one spatial kernel, no import cycle,
legacy surface removed. Concretely, ratified by the project owner:

1. **Full removal**, no backward compatibility — sequenced **last** and
   **gated** (P3), not folded into the invisible relayer.
2. **Two terminals**: `EntitySelection` (CAD dimtags) +
   `MeshSelection` (node|element ids). **chain == terminal** — no
   `.result()` ceremony; `.result()` survives as a **zero-cost identity
   alias** so the documented `for nid,xyz` / `for eid,conn` idiom and
   the 23+ existing callers do not churn.
3. **Relayer split in two**: cycle-break (structural, genuinely
   behaviour-invisible) is a **different kind of work** from
   engine-dedup (behaviour-touching, reviewed pin-flips) and **must not
   be one phase**.
4. **"One engine" = one *spatial* kernel only.** The FEMData
   node/element/broker *name*-resolvers are semantically different and
   the difference is **load-bearing correctness** (HT2/HT3) — they stay
   separate. Spatial's 6 copies unify via **pinned reviewed flips**.
5. `.save_as(name)` (named, round-trips to results) and the geometry
   `crossing` / `Line` predicates fold into the idiom.

---

## 2. The keystone (the whole game)

`NodeResult` (`mesh/FEMData.py:76-134`), `ElementGroup`
(`mesh/_element_types.py:156-225`) / `GroupResult` (`:232-392`,
`:15` self-declared leaf; the coupled `resolve_type_filter`
`:399-440` moves with them) are **lightweight, dependency-free numpy
payloads**. **Relocate that triad into `apeGmsh/_kernel/`** alongside
the pure defs/records/resolver layer. This single move closes three
independent findings at once:

- **HT1 (RED-1 FATAL) dissolves.** `_record_set.py`'s only non-pure
  tie is the deferred `:469 from .FEMData import NodeResult` inside
  `NodeConstraintSet.phantom_nodes()`. With `NodeResult` in `_kernel`,
  that rewrites to a **downward** `mesh→_kernel` edge — not a
  `_kernel⇄mesh` cycle. Class identity is unchanged (only the module
  path moves), so `phantom_nodes()`'s public return is byte-stable.
- **HT8/R3-C (iteration contract).** `MeshSelection` overrides
  `__iter__` to delegate to the relocated `NodeResult`/`GroupResult`,
  yielding `(id,xyz)` / `(eid,conn)`. chain==terminal **and** the 23+
  callers survive.
- **R3-B (element payload).** `GroupResult`/`ElementGroup` are
  **retained intact** (not deleted) — the OpenSees emitter and beam
  viewer need per-type `ElementGroup.element_type`
  (`opensees/_internal/build.py:333-336`,
  `opensees/_orientation.py:368-380`, `phase-8.7-scope.md:98`); a flat
  list is `TypeError`-or-corrupt for mixed-type
  (`_element_types.py:295-313` raises by design).

One relayering move closing the import cycle, the element-typing
problem, and the iteration contract simultaneously is the signal that
this is the correct layering. **P1-K is the highest-leverage phase;
everything else is conventional once it lands.**

### 2.1 P1-K pre-flight corrections (execution-map; ratified before any move)

An exhaustive read-only execution-map + cycle pre-flight (the contract
the relocation follows) corrected the plan **before any file moved**.
Recorded explicitly so the move set is complete and the source-of-truth
matches HEAD:

- **(1) Stale line numbers → corrected to HEAD** (above): `ElementGroup`
  `:156-225`, `GroupResult` `:232-392` (not `:211-313`).
- **(2) The keystone is a closed *four*-symbol triad.**
  `GroupResult.get()` calls `resolve_type_filter`
  (`_element_types.py:399-440`), which reads `ElementGroup`
  attributes — `NodeResult`+`ElementGroup`+`GroupResult`+
  `resolve_type_filter` move together to `_kernel/payloads.py`.
  `ElementTypeInfo`/`make_type_info`/alias machinery **stay** in
  `mesh/_element_types.py` (no back-edge — it never calls the moved
  trio).
- **(3) STOP-condition resolved — `_consistent_quadrature.py` joins
  the move set.** `mesh/_load_resolver.py` imports
  `mesh/_consistent_quadrature.py` at 4 deferred sites
  (`:412/441/472/503`). Relocating `_load_resolver`→`_kernel` without
  it re-forms a `_kernel→mesh` up-edge — the exact cycle P1-K deletes.
  `_consistent_quadrature.py` is **pure** (numpy-only, zero
  `apeGmsh.*` imports — verified) → relocate to
  `_kernel/_consistent_quadrature.py`, rewrite the 4 sites to
  `from .._consistent_quadrature import …`. With this addition the
  pre-flight finds **no surviving (c)-class up-edge**: every move-set
  cross-package import is DISSOLVED (target co-relocates) or
  ALLOWED-DOWNWARD (`apeGmsh.fem`, HT10).
- **(4) Internal re-export decision (Option i — forced by the gate).**
  `mesh/FEMData.py` keeps `from .._kernel.payloads import NodeResult`;
  `mesh/_element_types.py` keeps a downward
  `from .._kernel.payloads import ElementGroup, GroupResult,
  resolve_type_filter` re-export. Legal `mesh→_kernel` **downward**
  edge (the intended direction) and *mandatory* — the P1-K gate
  requires the contract tests + all P0-C pins **byte-unchanged**, and
  some import these via their `apeGmsh.mesh.*` paths. This is an
  internal import-path facade, **not** the user-facing backward-compat
  the v2 mandate forbids; flagged as a P3/P4 internal-cleanup
  candidate (sweep with the legacy surface). Full file:line repoint
  list + topological order live in the execution-map (§A–§E).

---

## 3. Hard truths (source-proven; future implementers must not re-trip these)

| # | Hard truth | Source | Disposition |
|---|---|---|---|
| **HT1** | `_record_set.py` straddle: eagerly imported by all 3 core composites *and* welded back to the non-relocated broker via deferred `phantom_nodes()→FEMData.NodeResult` | `core/LoadsComposite.py:41`, `core/MassesComposite.py:38`, `core/ConstraintsComposite.py:52`; `mesh/FEMData.py:58-61`; `_record_set.py:469` | Dissolved by the §2 keystone (relocate `NodeResult`). |
| **HT2** | node↔element name-resolver divergence = **3 orthogonal axes** (dim-threading, part-gating, exception-breadth), runtime-proven. The element *name-resolver* does not thread `dim`; the public `elements.get(dim=)` is a **separate downstream post-filter that silently empties** on mismatch (no raise) — see §3.1(a) | node `mesh/FEMData.py:456-496` (`:478` dim-threaded label, `:489` dim-gated part, `:480,485` `except KeyError`); elem resolver `:851-880` (`:868` unscoped label, `:875` unconditional part, `:869,873` `except (KeyError,ValueError)`); public `dim=` post-filter `:753`→`:807` | **NOT collapsible.** Resolvers stay separate. Pinned by `test_pin_element_path_no_dim_scoping_and_unconditional_part`. The silent-empty `get(dim=)` post-filter is a **P3 fail-loud candidate**. |
| **HT3** | broker `_group_set._resolve` **silently merges** a multi-dim PG; contract engines **raise**; `test_rule3_pg_multidim_raises_everywhere` parametrize **excludes the broker** | `mesh/_group_set.py:80-95,146-180`; `core/_resolution.py:122-130` (raises); `tests/test_resolution_contract.py:69-82` | Characterize-and-keep (P0-C pin). Do **not** silently "fix". |
| **HT4** | `core/Labels.py` is **not** wholesale-relocatable (eager `import gmsh` + gmsh-driven class); only the 3 predicates are pure | `core/Labels.py:59`, `:68 LABEL_PREFIX`, `:71-87` (`is_label_pg/strip_prefix/add_prefix`); sole eager consumer `mesh/PhysicalGroups.py:10` | Split predicates → `_kernel/_label_prefix.py`; repoint (P1-K work item). |
| **HT5** | import tripwire `PKGS={core,mesh,viz,results}` is **blind** to `_kernel`/`fem`; "strictly stronger invariant" was false advertising | `tests/test_import_dag_polarity.py:31`, `:36-45 BASELINE`, `:105-106` | P0-T: widen `PKGS` to add `_kernel` + `fem`; re-freeze BASELINE. |
| **HT6** | the two `Selection` classes are **structurally irreconcilable** (`list` subclass `.tags()` *method* vs frozen `__slots__` `.tags` *property*); `viz.Selection` is **public-exported** + viewer-constructed | `core/_selection.py:361`; `viz/Selection.py:99-154`; `apeGmsh/__init__.py:79,187-188`; `model_viewer.py:1671`, `mesh_viewer.py:1454` | Removal is a real **migration** (P3), not a reconcile. v1's "keep both" was backward-compat-driven; no longer binds. |
| **HT7** | spatial = **6 copies with real divergence** — axis-aligned `nodes_on_plane(coords,axis,value,atol)` vs `(point,normal,tol)`; silent row-0 centroid bug | `mesh/_mesh_filters.py:23-42`, `:137-162` (`:159` `.get(nid,0)`); `results/_composites.py`; the 4 chain `_spatial_*` hooks | Unify via **pinned reviewed flips** (P3), not relocation. Owns ratified v1-S5 item 3. |
| **HT8** | chain `__iter__` yields **bare atoms**; legacy terminals yield `(id,payload)` | `_chain.py:216-217`; `mesh/FEMData.py:114-116`; `mesh/_element_types.py:211-219` | `MeshSelection.__iter__` override + `.result()` alias (ratified). |
| **HT9** | `GeometryChain` lacks the `crossing` straddle mode **and** the 2-point `Line` primitive that legacy `queries.select` has (~60 test sites) | `_chain.py:47-50 REQUIRED_VERBS`; `core/_selection.py:93-125` (`Line`), `:132-194` (`_parse_primitive`: dict→Plane, 2pts→Line, 3pts→Plane; `on`/`crossing`/`not_*`) | P2-G: add `crossing_plane` + `Line`/2-point verbs **before** legacy removal; parity-test vs the ~60 sites. |
| **HT10** | `_mass_resolver.py` eagerly imports `apeGmsh.fem._hrz/_shape_functions` (leaf-pure sibling pkg) | `mesh/_mass_resolver.py:43-52`; `fem/_hrz.py`, `fem/_shape_functions.py` (zero `apeGmsh.*` eager) | **State explicitly:** `_kernel` MAY depend on `apeGmsh.fem`. `fem` added to tripwire `PKGS` (visible, not hidden). |

> [!warning] HT2/HT3 are the durable replacement for v1's "FP-4 / do
> not touch the broker." The reason is **not** "the old doc says so" —
> it is RED-2's runtime-proven source fact that the node/element/broker
> resolvers compute *different entity sets* for the same reference, and
> that difference is correctness (e.g. multi-dim label scoping). Any
> future "let's unify the resolvers" must re-read HT2/HT3 first.

### 3.1 P0 reconciliations (observed reality — the pins are authoritative)

P0-C empirically characterized the running system and caught **two
places where this plan's prose was wrong**. The pins lock *observed
reality*; this plan is corrected to match them (not vice-versa).
Recorded explicitly so P1-K's invisibility proof and P3's
reviewed-flips are anchored to fact:

- **(a) Element `get(dim=)` is a silent-empty post-filter, not "no dim
  arg".** HT2's *resolver* claim is intact (the element name-resolver
  does not thread `dim`). But the public `fem.elements.get(dim=)`
  parameter **exists** (`mesh/FEMData.py:753`) and is applied as a
  downstream group post-filter (`:807`): on a dim-3-only result,
  `elements.get(label="reg", dim=2)` and
  `elements.get(target="P1", dim=2)` return an **empty `GroupResult`
  with NO raise** — not the dim-2 slice, not an error. Locked by
  `test_pin_element_path_no_dim_scoping_and_unconditional_part`. This
  silent-empty is a **P3 fail-loud candidate** (sibling to HT3/HT7).
- **(b) Results dispatcher silently UNIONS multiple named selectors.**
  The prior assumption ("exactly-one selector; passing both raises")
  is **false**. `results…_resolve_node_ids(pg=X, label=Y)` returns the
  **sorted union** of both, **no raise**
  (`results/_composites.py:312-335`). The exactly-one guard fires
  **only** when `ids=` is combined with a named selector
  (`:293-298`/`:347-352`). Locked by
  `test_pin_results_samename_label_pg_precedence`. The silent
  multi-named-selector union is a **P3 fail-loud candidate**; any P3
  text must say "`ids=` is exclusive vs named selectors; multiple
  *named* selectors currently UNION" — never "passing both raises".

Neither reconciliation changes the ratified architecture or scope —
both refine HT2 / correct a mental-model error and add two
characterized behaviors to the P3 fail-loud review set.

---

## 4. Architecture

```
apeGmsh/_kernel/                 [NEW root-leaf pkg; stdlib/numpy/gmsh + apeGmsh.fem only]
  _label_prefix.py    is_label_pg / strip_prefix / add_prefix / LABEL_PREFIX   (HT4)
  defs/               core.{loads,masses,constraints}.defs  (relocated, pure)
  records/            mesh.records._{loads,masses,constraints,kinds}  (relocated, pure)
  resolvers/          mesh._{load,mass,constraint}_resolver + _constraint_resolver/  (relocated)
  record_sets.py      _record_set.py (relocated; :469 → from .._kernel … NodeResult)
  payloads.py         NodeResult / ElementGroup / GroupResult  ← THE KEYSTONE (HT1, R3-B)
  spatial.py          one point-family kernel + fail-loud centroid (P3; pinned flips)
  chain.py            SelectionChain  (moved from apeGmsh/_chain.py)

core / mesh / viz / results   ── all import DOWNWARD into _kernel ──►  _kernel ──► apeGmsh.fem
                              (no core⇄mesh cycle; tripwire PKGS now includes _kernel, fem)

EntitySelection(SelectionChain)  FAMILY="entity"   — dimtags; .to_label(Tier-1)/.to_physical(Tier-2)
MeshSelection(SelectionChain)    FAMILY="point"    — node|elem ids; __iter__→(id,payload); .save_as
```

`_constraint_resolver/_geom.py` (KDTree/Newton non-matching-mesh tie
kernel) relocates **with** its package but is **OUT of scope** for
selection — it is a different problem (projection, not selection) and
must not be folded into `spatial.py`.

---

## 5. Ratified product decisions

- **R-v2-1** Full removal, no backward compat; sequenced **last**
  (P3), gated on **zero internal callers** of each removed symbol.
- **R-v2-2** Two terminals; chain==terminal; `.result()` = zero-cost
  identity alias (1-line; saves 23+ call-sites).
- **R-v2-3** `EntitySelection.to_label` = `session.labels.add`
  (Tier-1, `_label:`-prefixed, boolean-stable);
  `EntitySelection.to_physical` = `session.physical.add` (Tier-2, raw
  gmsh PG). **Distinct registries; ADR-locked non-mergeable.** A
  `Clash` user-name can coexist as `(d,'Clash')` *and*
  `(d,'_label:Clash')`; merging them silently destroys Tier-1
  boolean-op identity. `EntitySelection` also exposes `.to_dataframe()`.
- **R-v2-4** `GroupResult`/`ElementGroup` **retained intact**
  (relocated, not deleted) as the element payload — parallel to the
  results `*Slab`s.
- **R-v2-5** "One engine" = **spatial-only** unification via pinned
  flips. Name-resolvers (node/element/broker) stay separate (HT2/HT3).
- **R-v2-6** Relayer split: P1-K (cycle-break) is behaviour-invisible
  pure relocation; **no resolver/spatial dedup in P1-K**.
- **R-v2-7** `.save_as(name)` + `crossing_plane`/`Line` verbs folded
  into the idiom (P2-I / P2-G).

---

## 6. Phase ledger

Order: **P0-T ∥ P0-C → P1-K → (P2-I ∥ P2-G) → P3 → P4.** P0 blocking.

| Phase | Status | Commit |
|-------|--------|--------|
| P0-T  | pending | — |
| P0-C  | pending | — |
| P1-K  | pending | — |
| P2-I  | pending | — |
| P2-G  | pending | — |
| P3    | pending | — |
| P4    | pending | — |

### P0-T — tripwire hardening (blocking, additive)

Add `_kernel` **and** `fem` to `tests/test_import_dag_polarity.py:31
PKGS`; re-derive `BASELINE` on untouched HEAD (no relocation yet) so
the frozen set is honest before any move. **Gate:** tripwire green
with widened `PKGS`; BASELINE diff = pure scope-widening, reviewed.
Closes HT5, HT10. Rollback: revert one test file.

### P0-C — characterization battery (blocking, additive, zero prod edits)

Add pins that turn every would-be silent flip into a reviewed
pin-flip: `test_pin_broker_multidim_pg_silently_merges` (HT3),
`test_pin_element_path_no_dim_scoping_and_unconditional_part` (HT2),
`test_pin_meshselection_on_plane_is_axis_aligned` (HT7),
`test_pin_meshfilters_element_centroids_silent_row0` (HT7),
`test_pin_element_samename_label_pg_precedence`,
`test_pin_results_samename_label_pg_precedence`. **Gate:** 100% green
on HEAD; zero production edits.

### P1-K — the keystone relayer (behaviour-INVISIBLE; own PR)

Create `apeGmsh/_kernel/`. Relocate, as a **pure module move** (no
behaviour change, no resolver merge, no spatial merge, no
`tolerate_wrong_dim`) — per the §2.1 corrections and the execution-map
§A–§E: the keystone triad `NodeResult`/`ElementGroup`/`GroupResult`/
`resolve_type_filter` → `_kernel/payloads.py`;
`core.{loads,masses,constraints}.defs`; `mesh.records._*`; the 3
resolvers + `_constraint_resolver/`; **`_consistent_quadrature.py`**
(§2.1-(3)); `_record_set.py` (rewrite `:469`→`_kernel`); the 3
`Labels` predicates → `_kernel/_label_prefix.py`;
`apeGmsh/_chain.py`→`_kernel/chain.py`. Repoint every consumer per
execution-map §B (the 3 composites, `FEMData.py:58-65`,
`PhysicalGroups.py:10`, `core/Labels.py` internals, the 5 chain
imports, all deferred/test sites); keep the §2.1-(4) Option-i
`mesh→_kernel` downward re-exports so the invisibility-proof tests
stay byte-unchanged.
**Gate:** widened tripwire green, BASELINE updated **in the same
commit** (the relocation diff *is* the reviewed decision —
`test_import_dag_polarity.py:18-20`); `test_resolution_contract.py` +
`test_target_resolution.py` (20) **byte-unchanged & green**; **all
P0-C pins unchanged** (this is the proof of invisibility);
`import apeGmsh` clean with `apeGmsh.__file__` under the worktree.
Closes HT1, HT4, R3-B. Rollback: the move is one reviewable diff;
revert the PR.

### P2-I — the idiom (additive; own PR)

`EntitySelection` + `MeshSelection` as `SelectionChain` subclasses on
the P1-K kernel. `MeshSelection.__iter__` overridden → `(id,payload)`
via relocated payloads; `.ids`/`.coords`/`.connectivity`/`.groups()`
accessors; `.save_as(name)` (registers into the mesh-selection store →
round-trips via FEMData HDF5 → addressable as `selection=`);
`.values(component=,time=,stage=,**slab_kw)` forwarding **verbatim** to
the spawning sub-composite's typed `.get` (the locked
`test_result_chain_subcomposites.py` fail-loud invariant). `.result()`
= identity alias. `EntitySelection.to_label`/`.to_physical` per
R-v2-3 + ADR; `.to_dataframe()`. Legacy `Selection` classes
**byte-unchanged** here (removal is P3). **Gate:** new idiom
daisy-chains at all levels (smoke); 23+ `(id,payload)` callers green
unchanged; ADR committed; tripwire green every commit. Closes HT8,
R3-A, R3-4.

### P2-G — geometry predicate completion (additive; own PR)

Add `crossing_plane(point,normal,*,tol)` + a `Line`/2-point verb path
to `EntitySelection` (entity family) and to `_chain.py
REQUIRED_VERBS`; port `core/_selection.py:166-194` on/crossing/not_*
(8-BB-corner) semantics into the entity hooks. **Gate:** parity test —
`EntitySelection` crossing/line/not_* == legacy `queries.select`
across the ~60 existing crossing/line sites; `__init_subclass__`
enforces the new verbs. Closes HT9. (Must land **before** P3 removes
legacy `queries.select`.)

### P3 — full removal + spatial dedup (BREAKING; gated; own PR)

Gated precondition: P2-I/P2-G shipped; **zero internal callers** of
each removed symbol (re-run the blast-radius census). Delete: both
`Selection` classes, `queries.select`+`select_all*`,
`g.model.selection.*`, `fem.*.get/get_ids/resolve`,
`g.mesh_selection.add_*/from_*`, the 5 legacy chains; drop `Selection`/
`SelectionComposite` from `apeGmsh.__init__.__all__`; repoint the
viewer `.selection` property + public export to `EntitySelection`.
Migrate the ~35 load-bearing src callers (opensees/viewers/cuts/
results). Unify the 6 spatial copies → `_kernel/spatial.py` as
**pinned reviewed flips** (each flip = a P0-C pin diff in the same
commit; absorbs ratified v1-S5 item 3). Rewrite ~7 legacy test files +
the two guide docs. **Gate:** full suite green; no legacy symbol
importable; every spatial flip is a reviewed pin diff. Closes HT6,
HT7, R-v2-1.

### P4 — docs / ADR / memory

Supersede `selection-unification.md`; ADR for the label/PG
non-merge + the `_kernel` boundary; tighten the
`MeshSelectionSet.py:741-742` axis-aligned-vs-generalized docstring;
update apegmsh-helper skill selection section; update MEMORY entries.

---

## 7. Invariants that must hold at every commit

- `tests/test_resolution_contract.py` + `tests/test_target_resolution.py`
  (20) green; **byte-unchanged through P1-K** (invisibility proof).
- `tests/test_import_dag_polarity.py` (widened `PKGS` incl.
  `_kernel`+`fem`) green every commit; `BASELINE` changes only as an
  explicit same-commit reviewed diff.
- All P0-C characterization pins green; any change is a deliberate,
  same-commit pin-flip with a regression test (never silent drift).
- Name-resolvers (FEMData node/element, `_group_set._resolve`) **not
  unified** (HT2/HT3). `_constraint_resolver/_geom.py` **not** folded
  into `spatial.py`.
- Verification protocol (non-negotiable):
  ```
  $env:PYTHONPATH = "<worktree>\src"
  $env:LADRUNO_OPENSEES_QUIET = "1"
  C:\Users\nmora\venv\opensees_venv\Scripts\python.exe -c "import apeGmsh; print(apeGmsh.__file__)"   # MUST be the worktree
  C:\Users\nmora\venv\opensees_venv\Scripts\python.exe -m pytest <targets> -q
  ```
  The editable install resolves `apeGmsh` to the **main repo** `src/`
  otherwise — a green without the worktree assertion is a false
  negative.
- Delivery: worktree → PR onto `origin/main`, one PR per phase.

---

## 8. Explicitly OUT of scope

- Collapsing the FEMData node/element/broker **name**-resolvers
  (HT2/HT3 — load-bearing correctness; ratified R-v2-5).
- `_constraint_resolver/_geom.py` (non-matching-mesh tie projection).
- Any backward-compat shim, deprecation-window, or `select→get`
  aliasing beyond the single `.result()` identity alias (R-v2-2).
- Reviving v1's "facade preserved / keep both Selections" — that was
  backward-compat-driven and is explicitly reversed.
