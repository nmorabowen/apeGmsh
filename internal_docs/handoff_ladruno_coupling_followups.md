# Handoff — Ladruno fork coupling integration: remaining follow-ups

**Status: 2026-06-12.** The Ladruno plane elements and coupling constraints are
integrated into apeGmsh and merged to `main`. This doc hands off the **deferred
follow-ups** so the next person can pick them up cold. Each item below names the
exact files/functions, the implementation approach, the gotchas, and a test
plan.

> Companion of record: the OpenSees fork's
> `Ladruno_implementation/ladruno_apegmsh_contract.md` (the apeGmsh-facing
> Ladruno feature contract) and the per-element guides
> `LadrunoPlaneElements_guide.md`, `LadrunoKinematicCoupling_guide.md`,
> `LadrunoDistributingCoupling_guide.md` in `nmorabowen/OpenSees@ladruno`.
> Fork class tags are authoritative in `SRC/classTags.h`.

---

## What shipped (done — for orientation)

| Feature | Tag | apeGmsh surface | PR |
|---|---|---|---|
| `LadrunoQuad` (4-node plane) | 33007 | `ops.element.LadrunoQuad` | #605 |
| `LadrunoCST` (3-node CST) | 33008 | `ops.element.LadrunoCST` | #606 |
| `LadrunoKinematicCoupling` (RBE2) | 33012 | `g.constraints.kinematic_coupling` | #609 |
| `LadrunoDistributingCoupling` (RBE3) | 33011 | `g.constraints.distributing_coupling` | #610 |
| Coupling control knobs (`CouplingControl`) | — | `k`/`kr`/`enforce`/`bipenalty_dtcr`/`absolute` on both couplings | #630 |

**How the coupling pipeline works (the map you need before touching it):**
declare → resolve → emit, all by record *type*.

- **Declare** (session): `g.constraints.kinematic_coupling(...)` /
  `distributing_coupling(...)` → `core/ConstraintsComposite.py` builds a
  `KinematicCouplingDef` / `DistributingCouplingDef`
  (`_kernel/defs/constraints.py`), each carrying a
  `CouplingControl` (`_kernel/_coupling_control.py` — a leaf module to dodge the
  `records ↔ defs` import cycle).
- **Resolve** (`get_fem_data`): `_kernel/resolvers/_constraint_resolver/_resolver.py`
  `resolve_kinematic_coupling` → `NodeGroupRecord`;
  `resolve_distributing` → `InterpolationRecord`. Both copy `defn.control` onto
  the record (storing `None` when `control.is_default`). Routing into
  `fem.nodes.constraints` vs `fem.elements.constraints` is by record **type**.
- **Emit** (bridge auto-emit): `opensees/_internal/build.py`
  `_emit_kinematic_couplings` (RBE2) and `_emit_one_interpolation` (RBE3, kind
  branch) write `element LadrunoKinematicCoupling`/`LadrunoDistributingCoupling`
  and append `rec.control.emit_flags()`. Element tag from the canonical
  `TagAllocator`. The token is in `_FORK_ONLY_ELEMENTS`
  (`opensees/emitter/live.py`) → stock OpenSees fails loud at run.
- **H5 round-trip:** `mesh/_record_h5.py` (`cpl_*` columns on the `node_group` +
  `interpolation` payload dtypes) + `mesh/_femdata_h5_io.py`
  (`_encode_control` / `_decode_control`, presence-probed). Neutral schema is
  **2.12.0**.

---

## ~~Deferred item A~~ — host-element auto-scalers (`-k auto` / `-kAlpha` / `-host` / `-wcap`) — **SHIPPED 2026-06-11**

> Implemented as planned below: `CouplingControl` gained `k="auto"` /
> `k_alpha` / `host` (FEM eid) / `bipenalty_wcap`; `fem_eid_to_ops_tag` is
> threaded through `emit_mp_constraints` (+ partitioned + stage variants)
> with fail-loud FEM-eid → ops-tag translation at the emit site; H5 added
> `cpl_k_auto` / `cpl_k_alpha` / `cpl_host` / `cpl_wcap` (neutral schema
> 2.13.0). The 2.12.0-stale schema-pin/parity suites
> (`test_record_h5_dtype` / `test_record_schema_parity` /
> `test_compose_schema_2_9_0`) were repaired in the same PR. See the
> CHANGELOG "coupling host auto-scalers" entry.

**What.** The fork elements support penalty auto-scaling off a representative
host element: `-k auto` (`K_t = kAlpha · max|K_host(i,i)|`), `-kAlpha`, `-host
$eleTag`, and the bipenalty `-wcap β` mode (`m_p = K_t/(β·ω_host)²`). apeGmsh
currently exposes only the **manual** knobs (numeric `k`, `kr`, `bipenalty_dtcr`
via `-dtcr`); `-k auto`/`-host`/`-wcap` are **not** wired.

**Why deferred.** `-host` is an *element* reference, so apeGmsh must resolve a
host FEM element id → its emitted OpenSees element tag **at coupling-emit time**.
That map (`fem_eid_to_ops_tag`) **exists** in `opensees/_internal/build.py`
(threaded through `materialize_*` / `emit_mp_constraints_partitioned` —
grep `fem_eid_to_ops_tag`) but is **not** passed into the serial MP-constraint
pass `emit_mp_constraints(emitter, fem, tags, *, claimed_ids=...)`, so the
coupling emitters can't see it.

**Approach.**
1. Add `k="auto"`, `k_alpha=`, `host=<FEM eid>`, and a `bipenalty_wcap=` field to
   `CouplingControl` (`_kernel/_coupling_control.py`). `k` becomes
   `float | "auto" | None`. Validation: `auto` requires `host`; `wcap` requires
   `host`; `wcap` and `dtcr` are mutually exclusive.
2. Thread `fem_eid_to_ops_tag` into `emit_mp_constraints(...)` and down to
   `_emit_kinematic_couplings` / `_emit_one_interpolation` (it's already in the
   partitioned siblings — mirror that). `CouplingControl.emit_flags()` will need
   the resolved host *ops* tag, so either resolve before calling `emit_flags`
   (pass the tag in) or have the emit site translate `host` (FEM eid) → ops tag
   and append `-host <tag> -k auto -kAlpha ...` itself.
3. H5: add `cpl_k_auto` (uint8), `cpl_k_alpha` (f64), `cpl_host` (int64, FEM eid,
   -1=none), `cpl_wcap` (f64) columns (presence-probed; bump neutral schema
   2.12.0 → 2.13.0). Store `host` as the **FEM eid** (stable across emit), not
   the ops tag.

**Gotchas.** Store/round-trip the FEM eid, not the ops tag (ops tags are
emit-time). The composite validates part labels but `host` is an element — decide
how the user names it (FEM eid vs a label resolved to a representative element).
`-wcap` on RBE3 needs `ω_host`; same host plumbing.

**Tests.** Extend `tests/opensees/unit/test_coupling_control_knobs.py`: `auto`
without `host` raises; `host` resolves to the emitted `-host <opstag>`;
round-trip of `auto`/`host`/`wcap`. Add an emit test that feeds a
`fem_eid_to_ops_tag` map and asserts the translated tag.

**Effort:** medium (the host-tag threading is the real work; the rest mirrors #630).

---

## ~~Deferred item B~~ — RBE3 tributary-area / explicit `-w` weighting — **SHIPPED 2026-06-12**

> Implemented as planned below: `weighting="area"` on
> `g.constraints.distributing_coupling` computes per-independent tributary
> areas in `resolve_distributing` (slave faces threaded via a new
> `_resolve_distributing` composite helper; the `face_map` gate now covers
> area-weighted distributing couplings), sets `InterpolationRecord.weights`
> in the sorted independent order, and fails loud on a node with no
> incident slave face. Emit/H5 needed no change, as predicted. The lumping
> model (face area split equally among face nodes) deliberately matches
> `g.loads` surface-tributary resolution. See the CHANGELOG "RBE3
> tributary-area weighting" entry.

**What.** `g.constraints.distributing_coupling` emits **uniform** weights only
(`-w` omitted ⇒ the fork's equal-weight default). The fork accepts `-w w1..wN`
(per-independent weights). The valuable case is **tributary-area** weighting so
the distributed load matches a uniform surface traction.

**Why deferred.** Per-independent weights need geometry: either the user supplies
an explicit list (but the independent **node ordering is internal** — the
resolver `sorted()`s them, so positional weights are unusable), or apeGmsh
computes each independent node's **tributary area** on the slave surface. The
latter is the right UX and the real work.

**Approach.**
- `composite.distributing_coupling(weighting="area")` (currently raises
  `NotImplementedError`): compute per-node tributary area over the resolved slave
  surface entities (sum of incident face areas / nodes-per-face), in the
  **resolver** where the mesh/connectivity is available
  (`_resolver.py::resolve_distributing`). Set `InterpolationRecord.weights` to the
  area vector (the field already exists and already round-trips through H5 —
  `interpolation_payload_dtype` has `weights`).
- The emit already handles `rec.weights is not None → -w ...`
  (`_emit_one_interpolation`), so **no emit/H5 change needed** once the resolver
  populates `weights`.
- Keep weights aligned to the **same sorted independent order** the record emits
  (`master_nodes`), so `-w[i]` matches `i_i`.

**Gotchas.** The resolver currently only has node sets; tributary area needs the
slave **faces** (entity → elements → areas). `resolve_tie` already projects onto
faces — reuse that face-collection machinery. Weights need not sum to 1 (the fork
normalizes by `W = Σw_i`).

**Tests.** `weighting="area"` on a known flat face → weights ∝ tributary areas;
the emitted `-w` order matches the independent order; H5 round-trip (weights
already covered, just assert non-None survives).

**Effort:** medium (tributary-area computation + face plumbing in the resolver).

---

## ~~Deferred item C~~ — RBE2 partitioned (MPI) single-canonical-rank emit — **SHIPPED 2026-06-12**

> Implemented as planned below: `_plan_rank_constraints` routes
> `KINEMATIC_COUPLING` records through a new `_canonical_coupling_rank`
> (min-of-intersection of the SLAVE owners, mirroring
> `_canonical_host_rank`); the reference node is ghost-declared when
> foreign (like the embedded path's constrained node); a slave set split
> across partitions fails loud with the per-slave owner map. The
> fail-loud guard is gone, the serial path is untouched, and the stage
> partitioned variant inherits the routing (shared planner).
> Integration coverage added to
> `test_emit_partitioned_mp_constraint_replication.py`. See the
> CHANGELOG "RBE2 partitioned emit" entry.

**What.** Under partitioned/OpenSeesMP emit, `kinematic_coupling` currently
**fails loud** (`_plan_rank_constraints` in `opensees/_internal/build.py`, the
`NodeGroupRecord` + `ConstraintKind.KINEMATIC_COUPLING` branch raises
`NotImplementedError`). Reason: a `NodeGroupRecord` historically *replicated*
across every owning rank (fine for an idempotent `equalDOF` command), but RBE2 is
now an **element** — replicating it would allocate a distinct element tag per rank
⇒ an N-fold over-constraint.

**Approach.** Route RBE2 like `ASDEmbeddedNodeElement` / RBE3 already are: a
**single canonical rank** emits the element (the rank owning the reference node,
or `min`-of-intersection of the slave owners), with the other tied nodes declared
as **foreign/ghost** nodes on that rank. The pattern is right there in
`_plan_rank_constraints` — the `surface_constraints` block computes
`_canonical_host_rank(rec, masters, node_owners)` and declares foreign nodes;
replicate that logic for the kinematic `NodeGroupRecord` (use refNode + slaves as
the node set that must co-locate). Then drop the fail-loud guard and let the
partitioned `_emit_kinematic_couplings(..., allowed_ids=ids)` path emit on the
canonical rank only.

**Gotchas.** RBE3 (distributing) already works partitioned because it rides the
InterpolationRecord/embedded canonical-rank path — **mirror that**, don't
reinvent. Fail loud (as today) if the ref + slaves genuinely can't co-locate on
one rank (they split across partitions). Keep the serial path unchanged.

**Tests.** Extend `tests/opensees/integration/test_emit_partitioned_mp_constraint_replication.py`
(it currently has **no** kinematic coverage): a 2-rank model with a kinematic
coupling → the `LadrunoKinematicCoupling` element emitted on exactly one rank,
ghost nodes declared on it; a split-across-ranks set still fails loud.

**Effort:** medium (the canonical-rank logic exists to copy; the test fixture is
the bulk).

---

## ~~Doc debt~~ — `internal_docs/guide_constraints.md` — **PAID 2026-06-12**

> `kinematic_coupling` / `distributing_coupling` sections rewritten (fork
> elements 33012/33011, fork-only failure mode, control knobs incl. the
> host auto-scalers, area weighting, partitioned canonical-rank note,
> RBE2↔RBE3 cross-pointers); the broker table now names the emit targets;
> the `mortar` section now states the fail-loud refusal instead of
> describing a feature that never existed. The fork-side
> `ladruno_apegmsh_contract.md` update ships separately in
> `nmorabowen/OpenSees@ladruno`.

The user guide's `kinematic_coupling` (≈L138) and `distributing_coupling` (≈L229)
sections still describe the **pre-fork** behavior and don't mention:
- they now emit the **fork elements** (`LadrunoKinematicCoupling` 33012 /
  `LadrunoDistributingCoupling` 33011) and are **fork-only** (stock OpenSees
  fails loud at run);
- RBE2's `equalDOF`→element change (offset reference now coupled rigidly via the
  moment-arm transport) and `dofs=None` ⇒ "all the slave has";
- RBE3 ships (was a `NotImplementedError` stub) and is the *flexible* counterpart;
- the **control knobs** (`k`/`kr`/`enforce`/`bipenalty_dtcr`/`absolute`).

Update both sections (and the "lands in the broker" table at ≈L295/L301, which is
still correct on record types but should note the emit target). The
`LadrunoKinematicCoupling_guide.md` / `LadrunoDistributingCoupling_guide.md` in
the fork are the source material.

Also update the fork's `Ladruno_implementation/ladruno_apegmsh_contract.md`:
flip the RBE2/RBE3 rows to *shipped* with the apeGmsh surface, and record the
deferred items above.

---

## Cross-repo workflow & CI lessons (so you don't relearn them)

- **The `## Unreleased` CHANGELOG heading is one giant single line every PR
  appends to** → every concurrent PR conflicts on it, and `main` merges fast, so
  branches go stale-`CONFLICTING` and **conflicting PRs don't run CI**
  ("no checks reported"). Resolve with a 3-way heading merge (take *theirs*
  heading + your branch's last ` · ` bullet; concatenate the section bodies;
  dedup). A throwaway Python `re` snippet does it reliably.
- **CI is NOT a required merge gate.** PRs merge even when the post-merge `Tests`
  run is red (that's how `main` went red after #606 → fixed by #608). So merge
  **manually the instant** `gh pr view N --json mergeStateStatus` is `MERGEABLE`
  / `UNSTABLE` (not `BLOCKED`); **auto-merge is disabled repo-wide**. Don't wait
  for green CI or `main` will move and re-conflict.
- **static-gates = mypy ratchet on `src/apeGmsh/opensees` (baseline 0).** Run
  `python -m mypy src/apeGmsh/opensees` **before pushing**. Past misses:
  `list[int] += "-flag"` (use `list[int | float | str]`); a helper taking
  `rec: object` then reading record attrs (annotate the concrete record type via
  a `TYPE_CHECKING` import).
- **`suite` semantic-drop:** a two-way auto-merge can silently drop test
  `expected`-set edits when two PRs touch the same literal (the #608
  catalog-coverage incident). After merging, grep all test files for stale
  assertions about the thing you changed.
- **Local test env:** `PYTHONPATH=<worktree>/src python -m pytest ...` (no
  editable install). System py3.11 has **no importable `openseespy`**, so
  `*live*` / `build_live` / `*_real` tests fail locally with
  `RuntimeError: Failed to import openseespy` — these are **env-only** and pass in
  CI (Linux). Filter them out when scanning for real failures.
- **Worktrees off `origin/main`**, base every PR on `main` (never hand-stack —
  see `~/.claude/CLAUDE.md`). Clean up the worktree + branch after merge.

---

## Quick reference

- Control knobs: `src/apeGmsh/_kernel/_coupling_control.py` (`CouplingControl`).
- Couplings: `core/ConstraintsComposite.py` (composite),
  `_kernel/defs/constraints.py` (defs),
  `_kernel/resolvers/_constraint_resolver/_resolver.py` (resolvers),
  `_kernel/records/_constraints.py` (records),
  `opensees/_internal/build.py` (`_emit_kinematic_couplings`,
  `_emit_one_interpolation`, `_plan_rank_constraints`),
  `opensees/emitter/live.py` (`_FORK_ONLY_ELEMENTS`).
- H5: `mesh/_record_h5.py` (`_coupling_control_fields`),
  `mesh/_femdata_h5_io.py` (`_encode_control` / `_decode_control`,
  `NEUTRAL_SCHEMA_VERSION`).
- Tests: `tests/opensees/unit/test_coupling_control_knobs.py`,
  `tests/opensees/unit/test_kinematic_coupling_emit.py`,
  `tests/opensees/unit/test_distributing_coupling_emit.py`,
  `tests/test_constraint_resolver.py`.
- Fork source: `nmorabowen/OpenSees@ladruno` —
  `SRC/element/ladrunoKinematicCoupling/`, `…/ladrunoDistributingCoupling/`,
  `SRC/classTags.h`, and the three `Ladruno_implementation/*guide.md` files.
</content>
