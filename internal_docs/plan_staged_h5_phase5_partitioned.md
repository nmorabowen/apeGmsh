# Plan — ADR 0055 Phase 5: partitioned staged H5 archival + capture

**Goal:** close the last fail-loud persistence gap — make `apeSees.h5()` and
`ops.domain_capture` work for **PARTITIONED staged** builds, so partitioned SSI
runs (the Cerro Lindo class: staged gravity → liner activation → HOLD supports →
absorbing flip, on an MP-partitioned mesh) get a Composed run file carrying
`/opensees/stages` + `/opensees/partitions` + the envelope ndf. That is the
feedstock the stage-aware viewer (ADR 0055 V1, PR #592) and the `model=`
requirement on every `Results.from_*` constructor (ADR 0020 INV-1) need.

**Demand naming.** ADR 0055 §"Open: partitioned demand" gates Phase 5 on "a
named user needing partitioned staged **H5** (not tcl/py)". Named here:
stage-aware results viewing (#592 consumes `/opensees/stages` from the run
file) and Composed-lineage results for partitioned staged runs. `ops.tcl/py`
covers MPI *execution* and keeps doing so; it does **not** produce the
viewer/lineage feedstock.

**Design authority:** `decisions/0055-staged-h5-archival.md` (Phases 1–2
shipped; this doc is the Phase-5 execution breakdown) + ADR 0027 (partition
emission invariants) + ADR 0052 (HOLD).

## Live-fact reconciliation (verified 2026-06-10 against source; ADR 0055 text is stale)

1. **Partitioned HOLD is SHIPPED.** ADR 0055 (§Context ¶3, §Guard lift,
   §Stays deferred, Phasing #5) says `_emit_stages_partitioned` "raises on any
   HOLD support (`apesees.py:2144-2152`)" — **false today**. Per-rank HOLD
   fan-out is implemented (`apesees.py:2447-2458` pre-resolve to
   `(node, dof)`; `2504-2508` rank filter; `2616-2623` per-owning-rank
   `pattern_open`/`sp_hold`/`pattern_close`) and locked by
   `tests/opensees/integration/test_emit_partitioned_stage_support_hold.py`
   (5 invariants). The ADR's "both partitioned legs" reduces to **one leg:
   H5 capture/replay**.
2. **The only fail-louds left:** the `ops.h5` guard
   (`apesees.py:5841-5849`) and the `domain_capture` `bridge=None` degrade
   (`apesees.py:4853-4855`). `_emit_stages_partitioned` itself never raises —
   the `h5()` guard comment (`apesees.py:5836-5840`) states this explicitly.
3. **Capture machinery (Phase 2, shipped):** in-band capture into
   `_StageEmitBlock` (`h5.py:550-659`; `stage_open` at 1756 creates the
   bucket; fix/mass/MP/pattern/chain calls redirect into it) plus the
   declarative complement attached by `set_stage_records` (`h5.py:2702`) with
   fail-loud drift cross-checks. `_PartitionEmitBlock` (`h5.py:511`)
   accumulates per-rank node/element ids during `partition_open` brackets —
   "purely a bookkeeping mode switch" (`h5.py:1641-1643`). The **stage×rank
   interleaving** (`partition_open` *inside* `stage_open`) is currently
   unreachable (the guard fires first) and therefore undefined and untested.
4. **Latent baseline bug (probe-proven 2026-06-10):** the *global*
   partitioned pass captures rank-REPLICATED records into the flat H5
   buffers. Probe: 2-rank stub with shared boundary node 2 + one
   `ops.fix(nodes=[2], ...)` → `H5Emitter._fixes` holds the record **twice**.
   `fix()` (`h5.py:975-984`) appends with no dedupe; `_emit_partitioned`
   emits fixes inside each owning rank's bracket
   (`_emit_fixes_partitioned`, per-rank ownership). The same applies to MP
   constraints (ADR 0027 INV-1 replication → duplicate
   `/opensees/constraints/*` rows). Consequence: every partitioned non-staged
   archive carries duplicated `/opensees/bcs/fix` and constraint rows; a flat
   `from_h5 → build()` re-emit double-applies them (duplicate SP constraints —
   doubled stiffness under a Penalty handler — and duplicate equalDOFs).
   Untested today: `test_partition_pipeline_e2e.py` step 10 only asserts
   rehydration, never re-emit.
5. **Latent baseline gap: no partition restore on re-write.**
   `opensees_model.py` has **zero** partition references; there is no
   partitions analog of `restore_stage_blocks` (`h5.py:2782`). A
   `from_h5 → to_h5` of a partitioned archive therefore drops
   `/opensees/partitions`, the `element_meta/*/partition_ids` columns, and
   `boundary_node_ids` → `model_hash` drifts and the viewer's partition
   metadata is lost on round-trip. (Write the failing test first — P5.0.)
6. **Flat-degrade replay is the established precedent:** `compose.py` has
   zero partition mentions — a partitioned *non-staged* archive replays as a
   FLAT deck (a legitimate single-process re-run), modulo bug #4.
7. **Re-derivability (the key enabler):** every per-rank decision in
   `_emit_stages_partitioned` is a pure function of (flat stage content) ×
   (neutral `/partitions` zone) × (`/opensees/element_tag_map`):
   `rank_owned = part.node_ids` (`apesees.py:2469`); element ownership via
   fem_eid (`2484-2488`, `2498-2503`); HOLD/fix/mass/remove/pattern filters
   are all `nid in rank_owned`. Every input already round-trips (neutral
   `/partitions` since schema 2.5.0; `element_tag_map` since Phase 2).
8. Schema constant today: **2.18.0** (`h5.py:357`).

## Design decision: declarative-once capture, derived fan-out, flat replay

Store the stage zone **rank-agnostically** — byte-identical to what the same
model captures unpartitioned. The partitioned emit *shape* (per-rank
bracketing, single global `domain_change`, per-rank pattern fan-out) is
**derived state**, not authored state — exactly ADR 0055's store-and-echo
governing rule applied one level up. Replay stays **FLAT** (matches the
non-staged degrade precedent, fact #6). True partitioned re-emit from H5 is
deferred to P5.4 (demand-gated; `ops.tcl/py` from the live bridge already
covers MPI deck generation).

Capture mechanism: make the H5 buffers **idempotent under INV-4
replication** ("partition-aware capture") instead of switching emit paths:

- **Replicated-by-design kinds** (fix / mass / equalDOF / rigidLink /
  rigidDiaphragm / embeddedNode / remove_sp / stage owned-node decls /
  `sp_hold`): identity-dedupe on the full record while a partition bracket is
  open. ADR 0027 INV-1 guarantees the replicas are byte-identical, so
  dedupe-by-identity is exact, not lossy.
- **Per-rank-split kinds** (stage patterns, stage regions, MPCO filter
  regions): each rank emits the SAME tag with a rank-filtered subset of
  lines/members → merge by tag (union of lines, emit order by first
  occurrence, `emit_index` seq stamp from the first open).
- **`_PartitionEmitBlock`s merged by rank at write time** — stage×rank
  brackets re-open the same rank, producing multiple blocks per rank; the
  writer must union them (today it would collide on the
  `partition_NN` group name).

Rejected alternative (keep on file in case implementation falsifies the merge
approach): route the H5 target through `_emit_stages_flat`. Cleaner capture,
but the global pass would still run partitioned while stage-owned topology is
excluded from it — `/opensees/partitions` would not cover stage-owned
elements and their `partition_ids` would stay `-1` → viewer Partition mode
regression on exactly the staged SSI models this phase exists for.

**INVARIANT P5-1 (the load-bearing test) — REVISED at P5.1 code time:**
for the same model, the `/opensees/stages` zone of a partitioned build is
**content-equal** (same owned-topology sets, fixes, HOLD lines, pattern
lines, region member unions, chain, analyze) to the unpartitioned
build's, masking the INV-5 `*_runtime_fallback` chain keys. Strict
byte-equality is NOT achievable: partitioned capture order is rank-major
(rank 0's owned subset first) while the flat emit order is target-major —
canonicalizing would change the flat 2.18.0 bytes. The byte-level gates
are instead (a) `from_h5 → to_h5` hash stability of the partitioned
archive itself and (b) two-fresh-builds hash determinism. Capture order
is the replay order; rank-major node/fix order within a stage block is
semantically equivalent (all between the same `domainChange` barriers).

## Slices (each = own PR, `--base main`)

### P5.0 — Baseline fixes (bugfix PR; no staging involved; no schema bump expected)

- **(a) Partition-aware dedupe of GLOBAL captures.** `fix` / `mass` /
  `equalDOF` / `rigidLink` / `rigidDiaphragm` / `embeddedNode` dedupe on full
  record identity while `_partition_current is not None`. Probe-shaped
  regression test first (shared-node stub → exactly one `/opensees/bcs/fix`
  row). Scope to partition-bracketed captures only — duplicate records in a
  flat build (user error) keep today's behavior.
- **(b) Partitions restore on re-write.** `restore_partition_blocks`
  symmetric to `restore_stage_blocks` + read-side plumbing on
  `OpenSeesModel` (feedstock exists: `H5Model.partitions()` /
  `PartitionEmittedRecord` in `h5_reader.py`). Failing test first:
  `from_h5 → to_h5` of a partitioned non-staged archive preserves
  `/opensees/partitions`, `partition_ids` columns, `boundary_node_ids`, and
  `model_hash`.
- **(c) Hash note.** Dedupe changes the bytes newly-written partitioned
  archives produce → `model_hash` differs vs. pre-P5.0 files. Lineage chain
  is warn-not-raise (three-broker contract); accepted bugfix consequence.
  Layout is unchanged, so no version bump — but re-check the ADR 0023 stance
  at code time and bump here if warranted.
- **Verify:** probe case single row; partitioned non-staged
  `from_h5 → to_h5` hash-stable; flat re-emit of a partitioned archive emits
  exactly one `fix` per (node, dofs) and one constraint per record; full
  `tests/opensees/h5` + partition integration tests green.

### P5.1 — Stage×rank capture + `ops.h5` guard lift (schema 2.18.0 → 2.19.0)

- Define `partition_open` semantics inside a stage bracket (today undefined):
  the rank bracket sets capture context; the stage bucket dedupes/merges per
  the Design section — `owned_node_ids` dedupe (a stage node on a rank
  boundary emits on both ranks), fixes/masses/MP/remove_sp identity-dedupe,
  pattern/HOLD/region merge-by-tag. `remove_element` and owned elements are
  single-owner (no dup possible) — assert, don't merge.
- Per-stage initial-stress and `activate_absorbing` ride the declarative
  side-channel already (their per-rank `addToParameter` / `flip_element_stage`
  calls are no-ops on H5, `h5.py:1730-1754`) — no change needed; pin with a
  test.
- Phantom-claim degrade (gate-2, #591) behavior unchanged: stage-claimed
  embedded constraints still degrade sidecar-less.
- Lift the `apesees.py:5841` guard entirely; rewrite the `Raises` docstring;
  **invert** the partitioned case in `tests/opensees/h5/test_h5_staged_fail_loud.py`.
- Schema **2.19.0** History bullet ("partitioned staged archives now exist;
  the stage zone is rank-agnostic by construction; hard-floor sentence per
  ADR 0023"). Dependent edits (the P2.0 list): `tests/fixtures/schema.py`
  `OPENSEES_CURRENT`/`OPENSEES_PRIOR_MINOR`, any hardcoded version asserts.
- **Verify:** INVARIANT P5-1 byte-equality (4-quad/2-partition staged fixture
  from `test_emit_partitioned_stage_bound_bcs.py`, incl. a HOLD stage and a
  stage pattern, vs. the same model unpartitioned); `from_h5 → to_h5`
  byte-stable; two-fresh-builds hash test; the recording-emitter oracle
  extended to the partitioned staged fixture; all existing partitioned-staged
  emit tests untouched and green.

### P5.2 — Flat replay acceptance — SHIPPED with P5.3 (zero source changes; tests only)

- Post-P5.1 the stage buckets are flat-equivalent, so `_replay_staged_into`
  (`compose.py:588`) should accept partitioned staged archives **as-is**.
  Confirm and close residual gaps — in particular the INV-5
  `*_runtime_fallback` chain attrs must re-emit the runtime conditional on
  replay (they round-trip through per-stage `chain_attrs`; verify the replay
  chain writer consumes them).
- **Verify:** `OpenSeesModel.from_h5(<partitioned staged>).build('tcl')`
  produces the FLAT staged deck, line-equal (INV-5 carve-out) to the
  unpartitioned build's tcl; runnable single-process smoke — the INV-5
  conditional makes the deck portable by construction.

### P5.3 — `domain_capture` gate lift (the user-facing payoff) — SHIPPED with P5.2

- Remove the `bridge=None` degrade (`apesees.py:4853-4855`); partitioned
  staged builds forward the bridge → the Composed run file carries
  `/opensees/stages` + `/opensees/partitions` + the envelope ndf
  (ADR 0048 sidecar — this also closes the ADR 0055 Phase-4 concern for the
  partitioned leg; check overlap at code time).
- **Verify:** tests mirroring `test_h5_staged_domain_capture.py` — the
  partitioned-staged case now forwards the bridge; the run file contains
  `/opensees/stages`; stage-aware viewer V1 (#592) loads stage masks from a
  partitioned staged run file (offscreen panel test, per the no-GPU
  verification convention); ndf sidecar round-trips.

### P5.4 — DEFERRED (demand-gated): true partitioned re-emit from H5

`_replay_staged_into_partitioned`: derive `element_owner`/`node_owners` from
the neutral `/partitions` zone + `element_tag_map`; reproduce per-rank
bracketing, the single global `domain_change`, per-rank pattern/HOLD/
initial-stress/absorbing fan-out (the ADR's Phase-5 verify bullet). Build
only if a user needs H5 → MPI-deck regeneration *without* the live bridge.
Until then the answer to "regenerate the MPI deck" remains `ops.tcl/py` from
the live bridge.

## ADR / doc follow-through (fold into the P5.1 PR or a sibling docs PR)

- **Amend ADR 0055:** correct the stale partitioned-HOLD claims (§Context ¶3,
  §Guard lift, §Stays deferred, Phasing #5); record the named demand; add an
  amendment note pointing here.
- `_DEFERRED.md` staged entries; `guide_partitioning.md` §5 (H5 round-trip);
  `staged-analysis.md` pointers; the `h5.py` deferred-zone comments.

## Risks / open items

- **Pattern/region merge ordering across ranks** is the fiddliest mechanic
  (emit-order by first occurrence must reproduce the flat order). The
  recording oracle + INVARIANT P5-1 catch drift loudly.
- **INV-5 chain attrs** legitimately differ partitioned vs. not — P5-1 needs
  the stated carve-out, not blanket byte-equality.
- **Pre-P5.0 archives** re-written after the dedupe fix change hash — accepted
  (bugfix; lineage warns).
- **Stage-bound MPCO recorders with filters** stay deferred
  (`_DEFERRED.md:224-242`) — orthogonal, restate in docs.
- **Live staged execution** (`LiveOpsEmitter.stage_open` raises) is untouched —
  ADR 0034 deferral, not this phase.
