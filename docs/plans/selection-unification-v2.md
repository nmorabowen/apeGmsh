# Selection / Resolution Unification v2 — Hardened Plan

Status: **ALL PHASES (P0-T · P0-C · P1-K · P2-I · P2-G · P3-K · P3-R ·
P3-S · P4) SHIPPED & MERGED to `main` — program COMPLETE** (P3-S #256
`4316aab`, P4 #257 `61b24ee`; 2026-05-19). Supersedes the scope
conclusions of
`docs/plans/selection-unification.md` (v1). The §6 ledger + the note
directly under it are authoritative (refreshed 2026-05-18 to
independently verified reality; the old "all pending" was stale).

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
| **HT6** | the two `Selection` classes are **structurally irreconcilable** (`list` subclass `.tags()` *method* vs frozen `__slots__` `.tags` *property*); `viz.Selection` is **public-exported** + viewer-constructed | `core/_selection.py:436` (HEAD; was `:361`); `viz/Selection.py:99-160`; `apeGmsh/__init__.py:79,187-188`; `model_viewer.py:1669-71`, `mesh_viewer.py:1454-56` | They were **never the same removal target** (RED/BLUE G2/SC-8, source-verified, owner-re-ratified 2026-05-18). Only `viz.Selection`'s **package exports** are dropped (P3-R); the **`viz.Selection` class is retained** as the viewer pick-result type, and **`core/_selection.Selection` is retained** as the `EntitySelection`/`GeometryChain` `.result()` terminal payload (R-v2-8 — entity-side parallel of R-v2-4). Retained *by architecture*, not backward-compat. |
| **HT7** | spatial = **6 copies with real divergence** — axis-aligned `nodes_on_plane(coords,axis,value,atol)` vs `(point,normal,tol)`; silent row-0 centroid bug | `mesh/_mesh_filters.py:23-42`, `:137-162` (`:159` `.get(nid,0)`); `results/_composites.py`; the 4 chain `_spatial_*` hooks | Unify via **pinned reviewed flips** (P3), not relocation. Owns ratified v1-S5 item 3. |
| **HT8** | chain `__iter__` yields **bare atoms**; legacy terminals yield `(id,payload)` | `_chain.py:216-217`; `mesh/FEMData.py:114-116`; `mesh/_element_types.py:211-219` | `MeshSelection.__iter__` override + `.result()` alias (ratified). |
| **HT9** | `GeometryChain` lacks the `crossing` straddle mode **and** the 2-point `Line` primitive that legacy `queries.select` has (parity battery = **122 occurrences / 9 files**, not ~60 — §6.1) | `_chain.py:47-50 REQUIRED_VERBS`; `core/_selection.py:93-125` (`Line`), `:132-194` (`_parse_primitive`: dict→Plane, 2pts→Line, 3pts→Plane; `on`/`crossing`/`not_*`) | P2-G: add `crossing_plane` + `Line`/2-point verbs **before** legacy removal; parity-test vs the ~60 sites. |
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
- **(c) The results-side removal target is the chain `.values()`, NOT
  the typed composite reader** (RED/BLUE G5, source-verified). The
  typed `results.<nodes|elements|gauss|fibers|layers|line_stations|
  springs>.get(component=, ids=, pg=, label=, selection=)` reader
  (`results/_composites.py:700,922,957,991,1031,1077,1113` →
  `Results._reader.read_{nodes,elements,…}` + `_resolve_{node,
  element}_ids` `:277`) is the **RETAINED** documented public reader.
  Only the *chain* terminal (`results.*.select(...).values()` →
  `ResultChain.get` → `host.get` `_result_chain.py:308-315`) is
  removed. Any P3 text saying "`results.*.get(component=)` is removed"
  means **the chain `.values()` path only** — `MeshSelection.values()`
  is repointed onto the retained reader in P3-K (SC-2).

(a)/(b) are P0-C reconciliations; (c) is a RED/BLUE scope
clarification. None changes the ratified architecture — (a)/(b) refine
HT2 / correct a mental-model error and add two characterized behaviors
to the P3 fail-loud review set; (c) corrects a removal-scope conflation.

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
- **R-v2-8** `core/_selection.Selection` is the entity-side `.result()`
  identity-alias terminal payload (`core/_selection.py:1368`/`:1039`) —
  the exact parallel of R-v2-4's element payload — **RETAINED, never
  removed**. The `viz.Selection` *class* is likewise retained as the
  viewer pick-result type; P3-R drops only the **package exports** of
  both. Retained *by architecture*, not backward-compat (RED/BLUE
  G2/SC-8, source-verified; **owner-re-ratified 2026-05-18**).

---

## 6. Phase ledger

Order: **P0-T ∥ P0-C → P1-K → P2-I → P2-G → P3-K → P3-R → P3-S →
P4.** P0 blocking. (`P2-I ∥ P2-G` was REFUTED — sequential, see §6.1
STOP-1. The monolithic "P3" / "two-PR split" was REFUTED — 3-unit
sequence, see §6.2 SC-1..SC-5.)

| Phase | Status | Commit | PR |
|-------|--------|--------|----|
| P0-T  | **SHIPPED** | `732fc50` | #245 |
| P0-C  | **SHIPPED** | `732fc50` | #245 |
| P1-K  | **SHIPPED** | `adc0c72` | #249 |
| P2-I  | **SHIPPED** | `caaea3c` | #251 |
| P2-G  | **SHIPPED** | `b427e5f` | #252 |
| P3-K  | **SHIPPED & MERGED** — invisible delegation-collapse + spatial unify | `a696bb0` | #254 |
| P3-R  | **SHIPPED & MERGED** — BREAKING removal (exports only; classes retained) + caller migration + `_mesh_filters` flip + dependent-test rewrite | `49033ed` | #255 |
| P3-S  | **SHIPPED & MERGED** — additive new-idiom spatial regression pins (4 gaps; M-CORRECTION-P3S) | `44a9087` | #256 |
| P4    | **SHIPPED & MERGED** — docs / ADR 0016 / skill / memory + the two ratified capability gaps (M-CORRECTION-P4) | `81ae4b0` | #257 |

> [!note] Ledger authoritative as of 2026-05-18 (was stale "all pending")
> **All additive phases shipped and LANDED ON `main`.** Each of the 4 v2
> PRs merged into its *parent feature branch* not `main` (only #245/P0
> reached `main` directly); the stranded-but-intact cumulative tip was
> then landed via **PR #253** (`guppi/sel-v2-p2i` → `main`, merge
> `63a4312`), reconciling the independent `origin/main` divergence
> (#247/#248/**#250 `fix(tests): field-based kw_only`**) into history.
> **Head-engineer independent post-#253 re-verification (2026-05-18, on
> merged `63a4312`, env-trap honoured, `apeGmsh.__file__` under the
> worktree):** full suite **5909 passed / 64 skipped / 0 failed**
> (exceeds the 5907/64/0 pre-divergence baseline by +2 additive/#250
> test deltas; zero regressions). The 4 byte-unchanged proof files
> (`test_resolution_contract`, `test_target_resolution`,
> `test_pin_resolution_v2`, `test_pin_spatial_v2`) are **byte-identical
> `git diff 732fc50..63a4312` (empty)** and green; `test_import_dag_polarity`
> green. #250's collision risk is **disproven** — it edited
> `tests/opensees/contract/*` (the OpenSees primitive contract suite),
> **not** the selection `tests/test_resolution_contract.py`. `63a4312`
> is a verified P3 base; P3 branches off it. The earlier owner-ratified
> "split in two" was **REFUTED by the P3 pre-flight** (5 source-proven
> FATALs, head-reverified — §6.2): P2-I shipped `MeshSelection` as a
> delegating shell over the legacy chains, so collapse==spatial-unify
> and 3 proof files call deleted symbols. **Owner re-ratified
> 2026-05-18 a 3-unit sequence P3-K → P3-R → P3-S** (§6.2). The
> adversarial **RED/BLUE pass on the §6.2 plan is COMPLETE** (4 RED +
> 1 BLUE, head-adjudicated, key claims source-verified): 13/14 findings
> conceded, 3-unit structure defended. **Owner re-ratified 2026-05-18
> two architecture-surface corrections:** (i) `core/_selection.Selection`
> RETAINED as the `EntitySelection` `.result()` terminal payload
> (R-v2-8, entity-side parallel of R-v2-4); (ii) SC-8 = the
> `viz.Selection` *class* retained as the viewer pick-result type —
> P3-R drops only the **package exports** of both, not the classes.
> Phase count unchanged (3-unit); P3-S rescoped to additive new-idiom
> regression pins (the behaviour-changing `_mesh_filters` flip moved
> into P3-R as a same-commit production+assertion diff). P3-R gated on
> a committed caller-migration artifact (§6.3, RED/BLUE m6).

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

### 6.1 P2 pre-flight corrections (ratified before any P2 execution)

An exhaustive read-only P2 execution-map + contract pre-flight found
the plan prose wrong a third time. Corrected **before any P2 file
changed**; the locks below are same-commit reviewed-diffs (identical
discipline to the import BASELINE and §2.1/§3.1):

- **STOP-1 — `P2-I ∥ P2-G` is unsafe; mandated order is P2-I → P2-G.**
  `_kernel/chain.py:91-116` `__init_subclass__` enforces
  `REQUIRED_VERBS` at **import time** on every concrete chain. The
  instant P2-G appends `crossing_plane`, every chain lacking it (5
  legacy + 2 new) raises `TypeError` at `import apeGmsh`. So: **P2-I
  first** (no `REQUIRED_VERBS` touch). **P2-G second, internally
  ordered:** (1) add a *base concrete* `SelectionChain.crossing_plane`
  (+ `Line`/2-point verb) delegating to a new hook, **point family
  raising loud** (the `GeometryChain.in_box` `inclusive=`→`TypeError`
  precedent — entity-only, never silent `[]`); (2) override on
  `EntitySelection` (+ legacy `GeometryChain` for through-P3 parity);
  (3) **only then** append to `REQUIRED_VERBS`. `crossing_plane` is a
  `REQUIRED_VERBS` **verb**, **never** a `_REQUIRED_HOOKS` entry.
- **STOP-2 — P2-I is NOT purely additive.** The 5 `.select()` host
  hooks (`core/Model.py:236`, `mesh/FEMData.py:310/964`,
  `results/_composites.py:618/678/891`, `mesh/MeshSelectionSet.py:834`)
  must **return the new types** (legacy 5 `*Chain` left
  defined-but-unwired; P3 deletes them). Same-commit reviewed-diffs the
  plan's P2-I gate omitted: (a) `tests/test_selection_idiom.py`
  `_EXPECTED_CHAINS` is an **equality** lock → 5→7; (b)
  `test_point_family_laws` must unpack `(id,_)` once
  `MeshSelection.__iter__` yields `(id,payload)` (the ratified HT8
  design — set-algebra is unaffected, it works on `_items`; only the
  public `__iter__` presentation changes); (c) `type(x) is
  <LegacyChain>` identity assertions in `test_fem_chain.py` /
  `test_result_chain.py` / `test_result_chain_subcomposites.py` /
  `test_geometry_chain.py` / `test_mesh_selection_chain.py` break at
  **P2-I** (not P3) — update/xfail-with-P3-marker same-commit. The
  23+ `(id,payload)` production callers go through `.get()`
  (`NodeResult`/`GroupResult`, **not** repointed) — untouched.
- **Placement.** `EntitySelection` → `core/_selection.py` (beside
  `GeometryChain`; no new edge). `MeshSelection` → **new leaf
  `mesh/_mesh_selection.py`** (mirrors `_node_chain.py`): eager
  `from .._kernel.chain import SelectionChain` only, payloads deferred.
  Adds **one** declared same-commit BASELINE triple
  `("mesh","_kernel","mesh/_mesh_selection.py")` — the *same downward
  polarity already frozen*; **no `core↔mesh` triple, no deferred edge
  flips eager**. ("P2 additive ⇒ BASELINE unchanged" was imprecise.)
- **`EntitySelection.to_dataframe()` is NEW** (legacy
  `core/_selection.Selection` has none; only `viz/Selection` does) —
  implement **locally**, no `viz` import (R8), session via
  `self._engine`. `.to_label`→`session.labels.add` (Tier-1,
  `_label:`); `.to_physical`→`session.physical.add` (Tier-2 raw) —
  ADR-locked non-merge (R-v2-3).
- **Parity battery = 122 occurrences / 9 files** (lead
  `tests/test_selection.py` 59), not HT9's "~60".

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
**byte-unchanged** here (removal is P3). Repoints the 5 `.select()`
host hooks to return the new types (§6.1 STOP-2). **Gate:** new idiom
daisy-chains at all levels (smoke); 23+ `(id,payload)` callers green
unchanged (they go via `.get()`, untouched); ADR committed; the four
proof tests byte-unchanged; **same-commit** locks per §6.1 STOP-2
(`test_selection_idiom._EXPECTED_CHAINS` 5→7, `test_point_family_laws`
pair-unpack, the 5 identity-assertion files) + the one declared
BASELINE triple for `mesh/_mesh_selection.py`. Closes HT8, R3-A, R3-4.

### P2-G — geometry predicate completion (additive; own PR)

Add `crossing_plane(point,normal,*,tol)` + a `Line`/2-point verb path
to `EntitySelection` (entity family) and to `_kernel/chain.py
REQUIRED_VERBS`; port `core/_selection.py:166-194` on/crossing/not_*
(8-BB-corner) + `Line` semantics into the entity hooks. **Internally
ordered per §6.1 STOP-1** (base concrete verb + point-family loud-raise
→ entity override → append to `REQUIRED_VERBS` last; `import apeGmsh`
green at every intra-PR commit). **Gate:** parity test —
`EntitySelection` crossing/line/not_* == legacy `queries.select`
across the **122-site / 9-file** parity battery (§6.1);
`__init_subclass__` enforces the new verbs on all 7 concrete chains;
`test_selection_idiom.test_identical_public_verb_surface` carves
`crossing_plane` into the family-specific-signature exception
(same-commit, the `in_box` template). Closes HT9. (Must land **before**
P3 removes legacy `queries.select`.)

### 6.2 P3 pre-flight corrections (ratified before any P3 execution)

An exhaustive read-only P3 execution-map + cycle/contract pre-flight
found the plan prose wrong a **fourth** time — and unlike §2.1/§3.1/§6.1
(stale lines / mental-model refinements) this one is **architectural**:
it invalidates the monolithic "P3" *and* the owner's first "two-PR
split". Corrected **before any P3 file changed**; the head-engineer
independently re-verified every FATAL at HEAD `63a4312` source; the
owner re-ratified the 3-unit sequence 2026-05-18.

**Root cause (one error, same class as the prior three):** the plan's
P3 prose was written against an *imagined* post-P2-I architecture, not
the one **P2-I actually shipped**. P2-I's `MeshSelection`
(`mesh/_mesh_selection.py:30-38` docstring; `_delegate()` `:112-144`)
is a thin **engine-polymorphic delegating shell** — it constructs a
fresh legacy `NodeChain`/`ElementChain`/`ResultChain`/
`MeshSelectionChain` and routes *every* per-engine hook to it
(`_coords_of` `:151`, `_spatial_*` `:154-163`, `__iter__` `:211`,
`connectivity` `:256`, `groups` `:277`, `values` `:310-312`,
`_materialize` `:377`). The shipped docstring itself says "*Those
legacy chains stay defined-and-importable through P2-I (P3 deletes
them)*" — P2-I **deliberately deferred the collapse to P3**; the
plan's "the 5 chains are defined-but-unwired, delete them" is false at
source.

STOP-conditions (all head-re-verified at HEAD source):

- **SC-1 FATAL** — the surviving v2 `MeshSelection` is wired to 4 of
  the 5 legacy chains via `_delegate()`
  (`mesh/_mesh_selection.py:112-163`); load-bearing, not unwired. Only
  `GeometryChain` (`core/_selection.py:852`) is genuinely unwired.
- **SC-2 FATAL→ADDRESSED-IN-P3-K** (RED/BLUE, source-verified) —
  `MeshSelection.values()` (`:310-312`) → `ResultChain.get` →
  `host.get(...)` (`results/_result_chain.py:308-315`). `host.get` is
  the **typed sub-composite reader** (`results/_composites.py:700,922,
  957,991,1031,1077,1113` → `Results._reader.read_{nodes,elements,…}`
  + `_resolve_{node,element}_ids` `:277`) — the **RETAINED** documented
  public reader, **not** a removal target. Only the *chain* terminal
  (`results.*.select(...).values()`/`ResultChain.get`) is removed. P3-K
  repoints `MeshSelection.values()` to call the spawning sub-composite
  `host.get(ids=list(self._items),component=,time=,stage=,**extra)`
  **directly** (the identical call `ResultChain.get` already makes), so
  the v2 results terminal survives chain deletion. The §6.2/§6 prose
  "`results.*.get(component=)` is removed" means the **chain
  `.values()` path only** (see §3.1(c)).
- **SC-3 FATAL** — 3 of 4 byte-unchanged proof files exercise removed
  symbols: `test_target_resolution.py:62-64,82-83` +
  `test_pin_resolution_v2.py` (17 sites) call `fem.*.get`;
  `test_pin_spatial_v2.py:63,141` calls `g.mesh_selection.add_nodes`.
  Only `test_resolution_contract.py` is removal-safe (uses
  `g.constraints.resolve`, not `fem.*.resolve`). §7 invariant 1 (byte-
  unchanged through removal) is **unsatisfiable as written** → rescoped
  in §7.
- **SC-4 FATAL** — the "PR-A remove / PR-B spatial-dedup" seam does not
  exist at source: deleting the 4 wired chains *requires* giving
  `MeshSelection` its own spatial+materialise kernel, which **is** the
  spatial unification — same edit at `_mesh_selection.py:112-163`.
- **SC-5 FATAL** — `fem.elements.select(element_type=|dim=|partition=)`
  runs `self.get(...).ids` (`mesh/FEMData.py:962-973`); the v2 element
  host hook calls the deleted `.get`.
- **SC-6 SERIOUS** — `results/_composites.py:139,264` +
  `_result_chain.py:196` (spatial targets) call the `fem.elements.
  resolve` removed in P3-R — confirms the SC-4 entanglement.
- **SC-7 SERIOUS→MINOR (RED/BLUE-resolved, source-verified)** —
  `core/Model.py:260` `viewer()` → `self.selection.picker(**kwargs)`;
  `Model.preview` (`:289`) calls `preview_model`, **not** `.picker`
  (dropped from SC-7's affected list). `SelectionComposite.picker`
  (`viz/Selection.py:645-651`) only constructs+shows
  `ModelViewer(parent=self._parent,model=self._model,…)` — it does not
  use the `Selection` class. **Resolution:** inline `Model.viewer()` →
  `ModelViewer(parent=self._parent,model=self,**kwargs);p.show();return p`
  (mechanical; P3-R).
- **SC-8 SERIOUS→RESOLVED (RED/BLUE, owner-re-ratified 2026-05-18)** —
  both `model_viewer.py:1669-71` and `mesh_viewer.py:1454-56` construct
  **`viz.Selection`** (`Selection(picks,self._parent)`), NOT
  `EntitySelection`/`core.Selection`; `MeshViewer` has no `_Queries`
  reachable at all (built `MeshViewer(self._parent)`, `mesh/Mesh.py:207`).
  **Resolution: the `viz.Selection` class is RETAINED** as the viewer
  pick-result type (structurally distinct from `core.Selection`, HT6);
  P3-R drops ONLY its package exports (`viz/__init__.py:2,4`,
  `apeGmsh/__init__.py:79,187-188`). **No viewer API change, no repoint.**
- **SC-9 SERIOUS (P3-K-blocking, same-commit)** —
  `tests/test_import_dag_polarity.py:196-200` spike-imports
  `apeGmsh.mesh._node_chain` + asserts `NodeChain.FAMILY=="point"`;
  BASELINE carries the **4 deleted-chain triples** `:84`
  `("mesh","_kernel","mesh/_elem_chain.py")`, `:87`
  `(…,"mesh/_mesh_selection_chain.py")`, `:88`
  `(…,"mesh/_node_chain.py")`, `:89`
  `("results","_kernel","results/_result_chain.py")` (one is
  `results→_kernel`). P3-K deletes those modules → the §7-invariant
  tripwire breaks unless the 4 BASELINE rows + the `_node_chain`
  spike-import are rewritten **in the P3-K commit** (reviewed diff).
- **SC-10 SERIOUS** — census scope: removed symbols referenced by
  **~75 test files**, not "~7"; ~40+ are `results.*.get(component=)`
  consumers. P3-R test-rewrite is an order of magnitude beyond the old
  estimate. Census re-runs must use ripgrep-backed Grep, **not
  `git grep -E`** (silently under-reports on Git-for-Windows).
- **SC-11 MINOR (flip bound to P3-R)** — `mesh/_mesh_filters.py` has
  **two** silent-row-0 sites (`:159` `element_centroids` *and* `:215`
  `elements_on_plane`). `_mesh_filters` is imported **only** by
  `mesh/MeshSelectionSet.py:35` (used `:239,308,314,317,636,643,885,
  935`) — the removal-target `g.mesh_selection.add_*` surface; **no
  chain imports it** (verified). The `:159`+`:215` silent→fail-loud
  flip is therefore a **production+assertion reviewed diff bound to
  P3-R** (inseparable from the `MeshSelectionSet`/`add_*` removal),
  **both sites in the same phase** — NOT a P3-S assertion-only flip.
- **SC-12 MINOR** — `g.mesh_selection.add_*/from_*` have **zero PROD
  src callers** (gate passes) but `from_geometric` consumes
  `viz.Selection.to_mesh_*`; P3-R removes both ends and the
  "geometric-selection → named mesh-selection" capability has no v2
  replacement (`.save_as` is live-engine-only) — a user-facing gap to
  document in P4, not an internal blocker.

**Spatial inventory (HT7 corrections, re-derived at HEAD; RED/BLUE):**
the 4 point-chain `_spatial_*` kernels are **byte-identical** at source
(de-dup is observably free); the only genuinely-divergent copy is
`mesh/_mesh_filters.py` (axis-aligned `nodes_on_plane` `:23-43`; silent
row-0 `:159` *and* `:215`), consumed **solely by the removal-target
`MeshSelectionSet`** (no chain) → P3-K leaves `_mesh_filters.*`
**byte-untouched** as oracle (invisibility holds).
`results/_composites._element_centroids` (`:123-149`) is **already
fail-loud at HEAD** (the `_result_chain.py:25-29` "np.clip silent"
docstrings are stale). `test_pin_spatial_v2.py` has exactly **2 pins**,
**both observing only the removal-target surface**: Pin 1
(`g.mesh_selection.add_nodes` `:63,141-142` + direct
`_flt.nodes_on_plane` `:151`); Pin 2 (direct `_flt.element_centroids`
`:220-222`). Both are byte-unchanged **through P3-K** (oracle untouched)
and **rewritten/retired in P3-R** when their surface is removed (the
`_mesh_filters` flip is the P3-R production+assertion diff, SC-11). The
genuinely-new `_kernel/spatial.py` behaviour is **re-pinned in P3-S via
the NEW idiom** `g.mesh_selection.select(...).on_plane(point,normal,
tol=)` / element-level `.coords` (`MeshSelectionSet.py:715`→`:849`;
`_kernel/chain.py:171`,`:128`).

### P3-K — delegation-collapse + spatial unify (behaviour-INVISIBLE; own PR)

The real keystone P2-I deferred. Make `MeshSelection` self-contained:
relocate the per-engine `_coords_of`/`_spatial_*`/`_materialize` bodies
out of `NodeChain`/`ElementChain`/`ResultChain`/`MeshSelectionChain`
into `MeshSelection` + **one** unified `_kernel/spatial.py`
(point/normal/tol family + fail-loud centroid) — a **NEW kernel for the
relocated chain payloads only**; `_mesh_filters.*` is **byte-untouched**
(no chain consumes it — SC-11). Element payload via the relocated
`GroupResult`/`ElementGroup`, **preserving `_groups` insertion-order**
in materialise (RED/BLUE m4 — else `.resolve()` row-order drifts).
Repoint `MeshSelection.values()` (`mesh/_mesh_selection.py:310-312`) to
call the **retained** sub-composite `host.get(...)` directly (SC-2).
`EntitySelection._materialize` keeps its standalone
`Selection(list(self._items),_queries=self._engine)` line
(`core/_selection.py:1368`) — `core/_selection.Selection` is the
**RETAINED** entity terminal payload (R-v2-8; no entity-side collapse —
it is not a delegation). Remove `MeshSelection._delegate()` + its 4
deferred chain imports. **P3-K deletes NOTHING** (Phase-0 refinement,
`selection-unification-v2-p3k-execmap.md`): the 4 chain modules +
`GeometryChain` + the `_ResultChainEngine`/`_LiveMeshEngine` adapters +
`engine_for` are left **defined-but-dead** (still importable; their own
tests pass byte-unchanged since the chain bodies are untouched). The
chain/adapter deletion + BASELINE 4-triple delta + spike rewrite + the
~10 chain-specific test-file disposition all **move to P3-R** — deleting
them in P3-K would collection-error ~10 test files (all top-level-import
the chain modules), contaminating the invisibility proof; deferral makes
P3-K a strictly invisible **2-file diff** (`mesh/_mesh_selection.py` +
new `_kernel/spatial.py`), mirroring the P1-K precedent (pure
relocation; removals deferred). **P0-C spatial pins stay pinned — NO
flips here.** The legacy *removal-target* surface stays present,
untouched, as the differential **oracle**. **Gate (P1-K-style
invisibility proof):** `git diff --name-only 63a4312` = exactly those
**2 files**; full suite == **5909/64/0** (zero delta); **all 4 proof
files byte-unchanged** (oracle present → differentially proves the
collapse byte-faithful); all P0-C pins green; `test_import_dag_polarity`
green with **BASELINE unchanged** (no deletions → no triple removed);
`apeGmsh.__file__` under the worktree; no legacy removal-target symbol
touched. **Record the P3-K oracle run** (`tests/_p3k_oracle/`: serialize
the legacy-surface outputs P3-R proof-rewrites verify against). P3-K
execution must first re-confirm at source that no chain imports
`_mesh_filters` (verified Phase-0: only `MeshSelectionSet.py:35`).
Rollback: revert the 2-file diff. Severs SC-1/SC-2/SC-4 runtime
entanglement (+ the SC-6 collapse + the structural half of HT7); the
file deletion that *closes* them is P3-R. **SC-5's
`ElementComposite.select`→`self.get` knot (`FEMData.py:962-973`) is
NOT severed by P3-K** — P3-K's invisible 2-file scope cannot/should not
touch `FEMData.py`; SC-5 is **removal-coupled → resolved in P3-R**
(§6.3 M-STOP-1, head-verified at source).

### P3-R — full removal + caller migration (BREAKING; gated; own PR)

Gated precondition: P3-K shipped & green; **FRESH zero-PROD-caller
census re-run** (ripgrep-backed, not `git grep -E` — SC-10); **the
committed P3-R caller-migration contract
`selection-unification-v2-p3r-callers.md` authored & owner-reviewed**
(RED/BLUE m6; supersedes the ephemeral "pre-flight §2 map"). That
contract's **§0 M-STOP-1/2/3 + M-MINOR-a/b + M-CORRECTION are mandatory
same-commit P3-R actions** the prose below under-specified — in
particular **M-STOP-1**: factor a private
`FEMData._filtered_groups(...)` and rewire `ElementComposite.select`'s
aux-branch (`FEMData.py:962-973`) off `self.get` **before/with** the
`get` deletion (else `fem.elements.select(element_type=|dim=|partition=)`
— the migration target — breaks); **M-STOP-2**: delete
`core/Model.py:89-90` (eager `SelectionComposite` instantiation);
**M-STOP-3**: rewire all **3** `fem.elements.resolve` sites
(`_mesh_selection.py:267`, `_composites.py:139,264`) via direct
per-type group iteration. Hard-remove
(no shim/alias — R-v2-1): the **public exports** `viz/__init__.py:2,4`
+ `apeGmsh/__init__.py:79,187-188` (the **classes stay defined** —
`viz.Selection` = retained viewer pick-result type per SC-8;
`core/_selection.Selection` = retained `EntitySelection` terminal
payload per R-v2-8); the `queries.select`/`select_all*`/`queries.line`
**method bodies** (`core/_model_queries.py:603-640,669-794`) and rewrite
its `:15` import to **exactly `from ._selection import Plane`** (the
sole symbol referenced by surviving code — the retained `_Queries.plane`
@:585/:596/:597; `Selection`/`_select_impl`/`Line` are referenced only
by removed methods. The symbol *definitions* in `core/_selection.py`
are untouched — load-bearing for retained `Selection.select`/
`EntitySelection`/P2-G. RED-3/BLUE/head-reverified; see
`selection-unification-v2-p3r-callers.md` §0 M-CORRECTION REVISED);
`SelectionComposite`; `fem.*.get/get_ids/get_coords/resolve`; the chain
`results.*.select(...).values()`/`ResultChain.get` path;
`g.mesh_selection.add_*/from_*`; **the 4 dead chain modules +
`GeometryChain`** (P3-K left them defined-but-dead — Phase-0 deferral).
**Same-commit chain-deletion bundle (moved from P3-K):** relocate
`_ResultChainEngine`/`engine_for` → new pure `results/_result_engine.py`
(repoint the 3 deferred sites `_composites.py:614,686,907`),
`_LiveMeshEngine`/`engine_for` → new pure `mesh/_live_engine.py`
(repoint `MeshSelectionSet.py:814`); remove the 4 BASELINE triples
`test_import_dag_polarity.py:84/87/88/89`
(`_elem_chain`/`_mesh_selection_chain`/`_node_chain`/`_result_chain`) +
rewrite the spike `:197,200` (`_node_chain`/`NodeChain` →
`_mesh_selection`/`MeshSelection`); dispose the ~10 chain-specific test
files (rewrite-to-`MeshSelection` where behaviour-bearing / delete where
chain-identity-only; `test_selection_idiom._EXPECTED_CHAINS` 7→2
`{EntitySelection, MeshSelection}`); rewire the one SC-6
`fem.elements.resolve` centroid site onto a non-removed path. Inline
`Model.viewer()` → `ModelViewer(parent=self._parent,model=self,
**kwargs)` (SC-7; `Model.preview` untouched). Migrate every PROD caller
per the §6.3 artifact (broker `fem.*.get` →
`fem.*.select(...).ids/.groups()/.result()`; results `.values()` →
retained `results.<sub>.get(...)`; **enumerate cuts/(4):
`_drift.py:446`,`_planes.py:214`,`_defs.py:201`,`_polygons.py:183`;
transcoders/(2-3): `_recorder.py:861,870,871`; opensees/(4):
`node.py:314`,`_orientation.py:359`,`build.py:323,345` — keep their
surrounding `try/except (KeyError,ValueError)` around `.select()`,
RED/BLUE m3**). Flip `_mesh_filters.py:159`+`:215` silent-row-0 →
fail-loud **here** (production+assertion same-commit diff — SC-11).
Rewrite `test_target_resolution.py` + `test_pin_resolution_v2.py` +
both `test_pin_spatial_v2.py` pins (SC-3; their surfaces removed here)
**same-PR, each new assertion verified equal to the recorded P3-K
oracle** (or, for the `_mesh_filters` flip, an explicit reviewed
legacy→fail-loud diff); the ~75-file cascade (SC-10) + the two guide
docs. **Gate:** full suite green; **no legacy removal-target symbol
importable** (assert `ImportError`); `test_resolution_contract.py`
still byte-unchanged (SC-3 — removal-safe); rewritten proof tests
behaviour-equal to the recorded P3-K oracle; import-DAG BASELINE +
tripwire green (same-commit reviewed diff). §7 invariant 1 formally
rescoped here. Closes HT6, R-v2-1, and SC-3/SC-7/SC-8/SC-10/SC-11/SC-12.

### P3-S — new-idiom spatial regression pins (own PR)

P3-R already removed both `test_pin_spatial_v2.py` pins' observation
APIs and flipped `_mesh_filters.py:159`+`:215` (the behaviour-changing
diff lives in P3-R, inseparable from the `g.mesh_selection.add_*`
removal — SC-11/RED/BLUE G6-G7). P3-S adds **new** paired regression
pins asserting the unified `_kernel/spatial.py` fail-loud centroid +
`(point,normal,tol)` plane behaviour **via the new idiom**
`g.mesh_selection.select(...).on_plane(point,normal,tol=)` and
element-level `.coords` (`MeshSelectionSet.py:715`→`:849`;
`_kernel/chain.py:171`,`:128`) — never silent drift (§7). Absorbs
ratified v1-S5 item 3. **Gate:** full suite green; every new spatial
behaviour is a reviewed pin with a paired regression test; no
behaviour change beyond P3-R's already-reviewed flipped sites. Closes
HT7 and the spatial half of R-v2-5.

> [!note] P3-S precision reconciliation (2026-05-19 — M-CORRECTION
> class: head-resolvable, owner-INFORMED via this note + the §7 touch
> below + `selection-unification-v2-p3r-callers.md` §0 M-CORRECTION-P3S
> + the P3-S PR body + the memory note; **NOT** owner-re-ratified —
> reverses no R-v2-x, no phase-structure change; same disposition class
> as the §6.3 §0 M-CORRECTION / SC-12 precedent. Owner explicitly
> confirmed the reshaped scope 2026-05-19.) Pre-flight census + 3×RED +
> 1×BLUE + head re-verification at HEAD `fab558f` (P3-R merged):
> 1. **Precision fix to the P3-S prose above.** "the unified
>    `_kernel/spatial.py` fail-loud centroid" is imprecise: at HEAD only
>    the plane/box/sphere **mask math** is unified in
>    `_kernel/spatial.py` (`:74-86` `plane_mask`/`box_mask`/
>    `sphere_mask`; the module docstring `:34-38` explicitly states the
>    element-centroid `_coords_of` source "stays per-engine in
>    `MeshSelection`" and the `_mesh_filters` centroid unification "is a
>    P3-R concern, not P3-K"). The **fail-loud centroid is the
>    per-engine `MeshSelection._centroid_map_live`**
>    (`mesh/_mesh_selection.py:329-356`, `KeyError` at `:346-352`
>    "…not in the live mesh node set — refusing to compute a corrupted
>    centroid (fail loud)."), reached by element-level `.coords`
>    (`:478-486` → `_coords_of` `:385-395` → `_coords_of_live`
>    `:358-382`). Read P3-S as: assert, **via the new idiom only**, that
>    (a) element `.coords` yields the per-engine fail-loud centroid
>    value (never silent row-0) and (b) `.on_plane` runs the unified
>    `_kernel/spatial.py` `(point,normal,tol)` `plane_mask`.
> 2. **Direct `_kernel.spatial` unit pins are rejected** (RED-2): the
>    P3-S mechanism is the new idiom `g.mesh_selection.select(...)`
>    **exclusively** — three committed artifacts name it and never a
>    kernel-unit path (this P3-S text `:719-722`; the superseded-spec
>    `:586-589`; the in-repo charter `tests/test_pin_spatial_v2.py:21-25`).
>    No `from apeGmsh._kernel import spatial` in the new file; no
>    `_kernel/*` math-leaf unit-test precedent exists in the suite.
> 3. **The literal box/sphere/plane + fail-loud-centroid-via-`.in_box`
>    pins are ALREADY green at HEAD** (`tests/test_mesh_selection_chain.py`
>    `:222-336`, `:432-449` — the P3-R rewrites; head-verified zero
>    `.coords` in that file and the only suite-wide selection-`.coords`
>    assertion is `tests/test_fem_chain.py:357-358`, node-level
>    shape/dtype only). P3-S therefore adds **only the four
>    source-proven successor gaps**, never duplicates, in one additive
>    file `tests/test_pin_spatial_kernel_v2.py` (zero `src/` change,
>    zero edits to existing tests; existing green literals re-used as
>    oracles): **PIN 1** element-`.coords` centroid value + a
>    `.coords`-triggered fail-loud regression (Gap D — the *only*
>    element-`.coords` value pin in the suite; the existing `:432-449`
>    fail-loud is via `.in_box`, head-verified, so the `.coords` entry
>    path is genuinely additive); **PIN 2** element-`on_plane` **frozen
>    id-set** replacing the cardinality-only `len()==4` at
>    `test_mesh_selection_chain.py:209` (Gap C — the §7 silent-drift
>    hole); **PIN 3** `<=tol` plane **boundary** via the live idiom
>    (Gap E); **PIN 4** **non-unit normal** normalisation via the live
>    idiom (Gap F).
> 4. **Head-caught corrections to the executor contract (trust-nothing,
>    at source).** PIN 4 must **replicate the existing-oracle chain
>    prefix** `select().in_box((0,0,0),(1,1,1)).on_plane(...)` and vary
>    *only* the normal to non-unit — the `[2,12,17,25]` literal at
>    `test_mesh_selection_chain.py:222-231` is the half-open-box-prefixed
>    z=0 **subset**, NOT the full 9-node z=0 plane that a box-less
>    `select().on_plane(...)` yields (head-verified at source; the full
>    z=0 plane is only `len()==9`-pinned at `:185-195`). PIN 3 freezes
>    its boundary id-set via the plan-endorsed **proof-file/p2g freeze
>    pattern** (empirical capture once under the env trap + a documented
>    structural derivation from the {0,0.5,1}³ lattice), never a
>    self-referential assertion.
> 5. **Stale citations corrected.** P3-S's `MeshSelectionSet.py:715`→
>    `:849` is pre-P3-R numbering; at HEAD `MeshSelectionSet.select` is
>    `:511-645` (symbol unchanged, returns `MeshSelection`).
>    `_kernel/chain.py:171` = `on_plane(self,point,normal,*,tol)`
>    (`tol` keyword-only, **no default** — every call passes `tol=`);
>    `:128` = the abstract `_spatial_plane` stub (impl
>    `mesh/_mesh_selection.py:416`). `_kernel/chain.py` is the surviving
>    `SelectionChain` base, **not** a P3-R-deleted module.

### P4 — docs / ADR / memory

Supersede `selection-unification.md`; ADR for the label/PG
non-merge + the `_kernel` boundary; tighten the
`MeshSelectionSet.py:741-742` axis-aligned-vs-generalized docstring;
update apegmsh-helper skill selection section; update MEMORY entries.
Document the SC-12 capability gap (no v2 "geometric-selection → named
mesh-selection" path replacing `from_geometric`/`viz.Selection.
to_mesh_*`) and the rescoped §7 invariant 1.

> [!note] P4 scope reconciliation (2026-05-19 — M-CORRECTION-P4
> class: head-resolvable, owner-INFORMED via this note +
> `selection-unification-v2-p3r-callers.md` §0 M-CORRECTION-P4 + the P4
> PR body + the memory note; **NOT** owner-re-ratified — every item
> resolves by following the repo's *own documented rules* (ADR
> append-only; fix the CI-deployed published lie); reverses no R-v2-x,
> no phase-structure change). Read-only pre-flight census + head
> re-verification at HEAD (P3-R merged + P3-S):
> 1. **P4-A is NOT "src docstrings only" (FATAL prompt-scope miss).**
>    The worklist also includes the **published mkdocs tree**:
>    `docs/api/selection.md` (in `mkdocs.yml` nav `:196`; `mkdocstrings`
>    `:96`) whose type table `:27-32` carries **build-breaking dead
>    autorefs** into the five P3-R-deleted modules
>    (`[GeometryChain][apeGmsh.core._selection.GeometryChain]`,
>    `…_mesh_selection_chain.MeshSelectionChain`, `…_node_chain.NodeChain`,
>    `…_elem_chain.ElementChain`, `…_result_chain.ResultChain`) and
>    `:14-15` teaches removed `add_nodes`/`queries.select`/
>    `selection.select_*` as live; the other `docs/api/*.md`;
>    `internal_docs/guide_selection.md` / `guide_selection_chain.md` /
>    `guide_queries.md` / `MIGRATION_v1.md`; and the
>    `.claude/skills/apegmsh-helper/SKILL.md` selection sections (§1.3/
>    §1.4/§7/§8/§9 teach the entire removed surface as "unchanged").
>    `.github/workflows/docs.yml` runs `mkdocs gh-deploy` on push to
>    `main` (paths incl. `docs/**`,`internal_docs/**`,`src/**`) → the
>    lie is **republished on merge**; fixing it is correctness-mandatory
>    (deploy-on-merge, non-strict — not a PR-blocking gate, but a P4
>    acceptance goal: mkdocs build clean / no dead autorefs).
> 2. **ADR: add 0016, do NOT edit 0015 (prompt path + verb both wrong).**
>    ADR 0015 exists at
>    `src/apeGmsh/opensees/architecture/decisions/0015-label-pg-separate-registries-kernel-leaf.md`
>    (there is **no** `docs/adr/`), Status "Accepted … Phase P2-I", and
>    is stale (future-tense chain deletion; cites the P3-R-deleted
>    `tests/test_p2i_parity.py`). It already covers the ratified
>    label/PG non-merge + `_kernel` leaf. The ADR
>    `decisions/README.md:7-8` mandates **append-only** ("write a new
>    ADR that supersedes it; do not edit history"). P4 therefore adds
>    **ADR 0016** (v2-completion: chains removed; `core/_selection.
>    Selection` + `viz.Selection` classes RETAINED per R-v2-8/SC-8;
>    `MeshSelection` self-contained on `_kernel/spatial.py`; the two
>    capability gaps) + its README index row; 0015 is left as the
>    historical P2-I record. "Finalize/extend ADR 0015" is reinterpreted
>    per the repo's own append-only governance.
> 3. **The single suite-delta tripwire (STOP — S-1).** Exactly ONE
>    pinned stale token suite-wide: the `from_geometric` substring in
>    `src/apeGmsh/mesh/MeshSelectionSet.py` `_seed_ids_by_name`'s
>    `KeyError` (≈`:498-506`) is asserted by
>    `tests/test_mesh_selection_chain_name_seed.py:227`
>    (`assert "from_physical" in msg and "from_geometric" in msg`).
>    **P4 leaves `MeshSelectionSet.py` byte-untouched** (its stale
>    strings are already classed "P4/optional cosmetic" by §6.3 §0
>    RED-2 MINOR; the message legitimately surfaces the *retained*
>    `from_physical` register-then-select route — the cosmetic gain is
>    not worth the suite-delta blast radius). No doctest collection
>    exists (`pyproject.toml:65` addopts has **no** `--doctest-modules`)
>    → every *other* docstring/string fix is inherently suite-safe; the
>    §7 zero-suite-delta invariant holds iff `MeshSelectionSet.py`
>    `:498-506` is untouched.
> 4. **Re-derive every seed at HEAD; the prompt/§6.2 line refs shifted
>    (P3-R deletions).** e.g. `core/_selection.Selection` terminal
>    `:1368`/`:1039`→`:1127`/`:1139`; `Selection` class still `:436`;
>    `MeshSelectionSet.select` `:511` (docstring `:580-591`),
>    `from_physical` `:337` (RETAINED). LEAVE the naive-sweep false
>    positives: `apeGmsh/__init__.py` `SelectionPicker=ModelViewer`
>    (viewer alias), `viz/__init__.py:3-5` / `core/_model_queries.py:15-21`
>    / `mesh/_live_engine.py` / `results/_result_engine.py` provenance,
>    `viewers/ui/_filter_view_tabs.py` `_select_all` (Qt). `GeometryChain`
>    is **DELETED** → a removed name in docs. The *retained* `Selection`
>    class's own docstrings that reference removed `queries.select` ARE
>    stale-fix (fix the reference, keep the class).
> 5. **The migration table (highest-value artifact — head-owned, RED-checked).**
>    Two non-obvious rows the prompt under-specifies: (i)
>    `g.mesh_selection.add_*` → `.select(...).save_as` is **live-engine
>    only**; the durable explicit-ids successor is the RETAINED
>    `g.mesh_selection.add(dim,ids,name=)`. (ii) "`results.*.get(
>    component=)` removed" is the **chain `.values()` path only** — the
>    typed `results.<sub>.get(component=)` reader is RETAINED and is
>    *itself* the successor (§3.1(c)). The two gaps confirmed at source:
>    SC-12 (`from_geometric`/`viz.Selection.to_mesh_*` — both ends
>    removed, no successor); the `SelectionComposite.select_*` filter
>    grammar (`labels=`/`kinds=`/`*_range=`/`predicate=`/`exclude_tags=`
>    /`physical=`/`at_point=`) — `EntitySelection` has **no** filter-
>    grammar successor (the retained `viz.Selection.filter()` is
>    viewer-pick-only, **not** a `g.model.select` migration path —
>    state precisely to avoid implying a false successor). Migration
>    table + both gaps go in `docs/api/selection.md` (published,
>    user-first) + a `docs/changelog.md` BREAKING entry; the
>    engineering record stays here + in ADR 0016.
> 6. **§7 invariant 1** is already rescoped in §7 — P4 finalizes the
>    wording past-tense (P3-R + P3-S done); it is an internal
>    test-invariant and stays in this plan, **not** in user docs.
> 7. **Memory:** the stale `MEMORY.md` index entry (the P3-resume line)
>    + `project_selection_api_deprecation.md` → final state, both gaps,
>    program-COMPLETE. Exclude `.claude/_p3s_*` / `_p4_*` scratch +
>    `.claude/settings.local.json` from the P4 PR.

---

## 7. Invariants that must hold at every commit

- `tests/test_resolution_contract.py` byte-unchanged **through P3-K
  and P3-R** (exercises no removed symbol — §6.2 SC-3/§4b; the
  through-removal invisibility proof). `tests/test_target_resolution.py`
  + `tests/test_pin_resolution_v2.py` byte-unchanged **through P3-K
  only**; P3-R rewrites them same-PR behaviour-identical, each new
  assertion verified equal to the recorded P3-K oracle.
  `tests/test_pin_spatial_v2.py` byte-unchanged **through P3-K** (both
  pins observe only the removal-target surface); **rewritten/retired in
  P3-R** (its `_mesh_filters`/`add_*` surface removed there), with
  new-idiom regression successors added in P3-S. All four green every
  commit (§6.2 SC-3 — the program-wide "byte-unchanged through P3" was
  unsatisfiable and is formally rescoped here). **FINAL (P4):** the
  invariant held at every commit — `test_resolution_contract.py`
  byte-unchanged through P3-K/P3-R/P3-S (and P4, docs-only); the two
  proof files oracle-equal post-P3-R; `test_pin_spatial_v2.py` is the
  P3-R SC-11 `_mesh_filters` fail-loud reviewed pin; the P3-S
  new-idiom successors are `tests/test_pin_spatial_kernel_v2.py`
  (5 reviewed pins, frozen literals). P4 changed no test
  (docs/skill/ADR/memory only) so the bar is unchanged.
- P3-S's "new-idiom regression successors" (the bullet above) are
  pinned **exclusively through `g.mesh_selection.select(...)`** — NOT a
  direct `_kernel.spatial` import (the §6.2 P3-S 2026-05-19
  M-CORRECTION); each is a reviewed assertion + a paired regression,
  and the element-`on_plane` centroid pin is a **frozen id-set** (not a
  cardinality check) so a cardinality-preserving centroid corruption
  cannot drift silently. The fail-loud centroid is the per-engine
  `MeshSelection._centroid_map_live`, not `_kernel/spatial.py` (which
  unifies only the box/sphere/plane mask math).
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
- Reviving v1's "facade preserved / keep both Selections" *as a
  backward-compat facade* — that was backward-compat-driven and is
  explicitly reversed. (Distinct from R-v2-8: `core/_selection.Selection`
  + the `viz.Selection` class are retained as **terminal payload /
  pick-result types by architecture**, not as a compat facade — only
  their package exports are dropped.)
