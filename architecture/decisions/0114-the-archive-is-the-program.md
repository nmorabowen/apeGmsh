# ADR 0114 — The archive is the program: `VERBS`, `command()`, tag law, `will_solve`

**Status:** Accepted, pending the maintainer's merge (2026-10-04; the K0
design was ratified by the maintainer on #1341)

**Owner:** nmora

**Evidence:** two independent architect briefs (`prog-architect-opus`,
`prog-architect-fable`) written on `1691d389`, reconciled on #1341
(comment 5981664535, "K0 reconciliation") and ratified on #1341 (comment
5981885789): "ratified, follow your recommendations on Q2–Q7". The
briefs and the reconciliation stay on the issue; this ADR cites them and
does not copy them.

**Supersedes** [ADR 0019](0019-opensees-model-read-side-broker.md) INV-5
(replay may re-allocate tags). The rest of ADR 0019 stands.

**Program slice:** #1359 (K1-1, the first of the eight K1 PRs; chain
issue #1201).

## Context

Chain K's goal is that the resolved `model.h5` is the program: a deck
rebuilt from the archive equals the deck the bridge emits
(`tcl() == from_h5(h5()).build('tcl')`, K2's oracle). Three facts stood
in the way on `1691d389`:

1. **The archive was incomplete without saying so.** Several Emitter
   verbs reach `H5Emitter` and are dropped silently (global `rayleigh`,
   `modal_damping`, the `eigen` family, `profiler`, the contact, rebar
   and embed passes, `equationConstraint`). Nothing listed them, so the
   gap could only grow.
2. **The Protocol grew without a bound.** Its module docstring records
   a dozen "architecture events", each one a new method on five
   emitters.
3. **Tags were not archive facts.** ADR 0019 INV-5 let
   `OpenSeesModel.build()` re-allocate tags, so any deck comparison had
   to mask them.

## Decision

The ratified K0 design is the reconciliation's §1 (A1–A9) and §2
(picks R1–R6), with the ratification's answers to Q2–Q7. In short:

### D1. `VERBS` is the registry (A2, R3, Q3)

`src/apeGmsh/opensees/emitter/verbs.py` holds one frozen `Verb` row per
Emitter verb. It is data only and imports nothing from apeGmsh. The
columns are `verb`, `via` (`protocol` or `command`), `family`, `scope`
(`global`, `stage` or `both`), `h5`, `store` (the archive path
template), `requires` (a set of capability tokens), `returns`,
`seq_op` / `seq_what` (the `/sequence.changes` row, K0-1), and `decl`.
The module docstring defines each one.

`h5` takes three values:

| Value | Meaning today |
|---|---|
| `archive` | the call's information reaches `model.h5` |
| `refuse` | the call raises, so `apeSees.h5()` fails loud |
| `ledger` | the call is dropped silently, in at least one scope |

The `ledger` rows are K2's xfail ledger. Their count can only shrink:
`tests/opensees/contract/verbs_ledger.txt` lists them under `N_LEDGER`.
K1-4 and K1-8 move rows out of it. Its ratified end state is the contact,
rebar and embed rows plus `equationConstraint` (Q3); refusing those
instead would break ADR 0112 D1 (a file on every run).

Rows record today's truth. Later PRs move them, and the lock pins every
move.

### D2. The Protocol's method count is frozen (A1, Q6)

`EMITTER_METHOD_COUNT` is 74 at K1-1. K1-2 adds `command(verb, *args)`
as method 75, and it is the last: a new verb is a `command()` token
with a `VERBS` row, never a new Protocol method. Existing typed fork
verbs stay typed until K4's flip. The freeze covers methods only.
Capabilities live in one place, E2 `TargetCaps` (P0.3 item 3, ratified
as Q6 and landing in K1-5); `archival` is one of its fields.

Public emitter methods outside the Protocol are side channels, listed
per emitter in `verbs.py::SIDE_CHANNELS` (for example
`H5Emitter.set_stage_records`).

A stage-only verb (`scope == "stage"`) called outside a stage bracket
**raises** (reconciliation §2, "smaller points"). Today it is a silent
no-op in `H5Emitter`; K1-2 implements the raise.

### D3. Who calls `command()` (R1, Q7)

Only a registered primitive's `_emit` with a literal verb, plus K2's
replay. An unknown verb raises on all five emitters (a fail-closed
allow-list). There is no user-facing `ops.command(...)` in K1; adding
one later needs its own ADR. `apeSees._register` stays the single
capture site (K0-4).

### D4. Tag law: tags are archive facts (A3, R6)

A tag is written once, by the bridge's build, and the archive keeps it.
Replay never allocates a tag. **This supersedes ADR 0019 INV-5.**
K2's oracle compares bytes with no tag masking. K1-3 enforces the law
with `TagAllocator.freeze()` at the end of `build()`, an audit of
emit-time minting, an AST lock against allocation in the emitter,
compose and replay paths, and a determinism check (two builds agree).

### D5. Order and declarations (R2, R5, A6, A7, Q2, Q4, Q5)

- Emit order lives in one run-length `/opensees/program` table. A
  record's `emit_index` is derived from it by a reader accessor; there
  are no per-record columns (Q2).
- Declarations live in `/opensees/decls` and `/opensees/decl_params`,
  written through a side channel joined on `(kind, tag)`. Parameters are
  stored by field name from `dataclasses.fields` of the primitive (A6,
  K0-8). Automatic `params_names` apply where the argv equals the
  fields, under a ratchet that can only shrink; argv flags without a
  field stay unnamed for now (Q4).
- `fix` and `mass` gain `name=`; recorders forward theirs; names stay out
  of `model_hash` (A7).
- `program` and the K1-4 `commands` store are hashed, so `model_hash`
  changes once for identical models at the K1-4 minor (Q5).

### D6. `will_solve` (A5, R4)

`/opensees@will_solve` is stamped from `staged or any(Analysis)`. The
archive also carries `@solve_refusals` (the gate ids collected at emit)
and `@requires` (the union of archived rows' `requires`). Replay to a
deck fails closed on the stored ids; `build('live')` refuses on stock
when `@requires` names the fork. The two emitter sniffs
(`type(emitter).__name__ == "H5Emitter"`, `isinstance(emitter,
H5Emitter)`) are replaced by `TargetCaps.archival` and banned by a quirk
rule. `/opensees/analysis` stays the vanilla chain's home (K0-6).

### D7. Schema and scope (A4, A8, A9)

Every schema change in K1 is an additive opensees minor, one per PR,
with one corpus file each (ADR 0113 D5/D8); no major and no floor raise.
Renaming `{type}_{tag}` groups is rejected, because it would be an
opensees major (ADR 0113 D4); the declaration key
`<zone>/<family>/<name|#k>` is the name-based key. Out of scope: a
DeckProgram IR, the fork loader, the `/sequence` writer, provenance
capture, `**kwargs` on `command()`, and a lossy escape.

## Consequences

- K1-1 lands D1 and D2's lock (`tests/opensees/contract/test_verbs_lock.py`
  in `lock-tests`). It changes no emitted byte and no schema.
- What the lock fails, exactly:
  - a 75th Protocol method, or a name bound by assignment on the
    Protocol;
  - a missing row, or a row whose `returns` differs from the method's
    annotation;
  - a K0-5 stage verb whose **row** stops being `archive`;
  - a new ledger row, or a raised `N_LEDGER`;
  - a public emitter method or class attribute missing from
    `SIDE_CHANNELS`;
  - a `refuse` row whose `H5Emitter` method (or any `self._helper` it
    reaches) does not raise `NotImplementedError`, or any other row
    whose method does;
  - an `archive` row whose `H5Emitter` method only discards its
    arguments (every statement is `del`, `_ = …`, `pass`, or `return`
    of a constant).

  It reads the source, not behaviour: an archive body that still
  stores, but stores the wrong thing or only part of it, passes the
  lock. Those regressions are K2's round-trip oracle.
- The stress-control trio (`addToParameter`, `step_hook_ramp`,
  `flip_element_stage`) is `archive` although the calls themselves
  write nothing: their information reaches the archive declaratively
  through the `set_initial_stress_records` / `set_stage_records` side
  channels (ADR 0055). The lock allow-lists exactly these three for the
  empty-body check, and fails if one of them grows a body.
- `constraints("LadrunoContact")` is skipped silently inside an
  `archive` row; that value-dependent skip is outside the row model and
  is K1-4's to ledger by a test.
- The architect-side open questions (verdicts per partition mode,
  ADR 0082 at replay, a warning-only freeze fallback) are
  stop-and-report conditions for the K1 builders, not decisions here.

## References

- #1341: the K0 card, reconciliation (comment 5981664535) and
  ratification (comment 5981885789).
- #1201: chain K. #1359: this slice.
- [ADR 0019](0019-opensees-model-read-side-broker.md) INV-5, superseded
  by D4.
- [ADR 0055](0055-staged-h5-archival.md): the staged archive
  and its side channels.
- [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) D1 and
  [ADR 0113](0113-compatibility-is-a-floor-per-zone.md) D4, D5, D8.

## Amendment — 2026-10-05 — D4: tags come from a build-time tag plan (K1-3d, #1445)

**Evidence:** two independent architect briefs (`prog-architect-opus`,
`prog-architect-fable`) written on `295b4f1d`, reconciled on #1445
(comment 5988261377) and ratified by the maintainer on #1445 (comment
5988371537). The briefs stay on the issue. **Program slice:** #1452
(K1-3d P0, the first of the K1-3d PRs; chain issue #1201).

### What D4 said, and what K1-3 landed

D4 placed the enforcement in K1-3 as `TagAllocator.freeze()` at the end
of `build()`. K1-3 was re-scoped (#1361, PR #1447): it pinned today's
behaviour (the cross-interpreter determinism pin, the `(verb, tag)`
multiset across the four emitters, the AST lock with its locked list of
minting helpers, and the shrink-only replay ledger
`tests/opensees/contract/tag_law_ledger.txt`) and shipped no `freeze()`.
Today, emit-time minting is spread across the emit helpers: `build()`
seeds a fresh allocator from the registered primitives, and elements,
transforms, MP constraints, interfaces, contacts, regions and parameters
each take their tags as they are emitted, with the split, partitioned
and staged decks each in their own order. A freeze at the end of
`build()` would close the allocator after the last mint, so it cannot
separate planning from emission. This amendment replaces that sentence
of D4; the law itself (a tag is written once, by the bridge's build, and
the archive keeps it; replay never allocates one) stands.

### D4, amended

1. **The plan.** Tags are minted by a build-time tag plan:
   `plan_tags` in `src/apeGmsh/opensees/_internal/tag_plan.py` (new, S1).
   Each existing allocation loop moves out of its emit helper into the
   planner (moved, not copied, so there is one source of order), and the
   emit helpers read the plan. Rejected: a dry-run or tape emit (it
   doubles build cost against `emit-cost-gate`) and a second, simulated
   planner beside the emit.
2. **The freeze.** `TagAllocator.freeze()` is called at the end of
   `plan_tags`, not in `build()`. A mint after the freeze raises
   `TagLawError`. `apeSees._tags`, the registration-time allocator, is
   never frozen: registering a primitive after a build is legal, and the
   next build plans again.
3. **Mode-keyed during the migration.** The plan is keyed by emit mode
   (`split`, `partitioned`, `staged`) and memoised lazily per mode, so
   every migration slice (S1–S6) leaves the flat, split, partitioned and
   staged decks, and the 86 golden cells, byte-identical. The
   migration-time safety net is the frozen planner allocator plus a
   per-kind `fork()`: a kind that has migrated is frozen in the plan, so
   a minting site the migration missed raises where it is.
4. **Canonical numbering, in the last slice (S7).** The final slice
   makes partitioned numbering canonical: flat order, so a tag is
   rank-invariant and one owner has one tag whatever the emit mode ("a
   tag is an archive fact" implies it). This is the one deliberate deck
   change of K1-3d: partitioned decks renumber once (the region,
   MP/interface and contact order), the partitioned golden cells are
   regenerated in that PR, and
   `tests/opensees/integration/test_interface_partitioned_emit.py::test_interface_plus_embedded_exactly_once_with_documented_drift`
   flips to equality, retiring the conditional in
   [ADR 0093](0093-zerolength-interface-constraint.md) INV-5 (its S8
   amendment).
5. **The archive.** Tags with no archived row today (the initial-stress
   and absorbing parameters, `update_parameter`) are archived per row in
   K1-8 (opensees 2.27.0). There is no `/opensees/tag_blocks` table: a
   third copy of each tag would need a hashing rule, and the ratified
   K1-4…K1-8 version plan does not change. The ledgered reinforce-tie
   and contact tags get their home when K2 flips those rows.

### Slices and oracles

P0 (this amendment) → S1 (the scaffold: `tag_plan.py`, `freeze()` /
`TagLawError`, the plan == tapped-stream oracle; no hub lock) → S2
(`apesees.py`: elements) ∥ S3 (`_internal/build.py`: transforms, MP,
interfaces, contacts) → S4 (`apesees.py`: regions, which fixes #1446)
∥ S5 (`build.py`: parameters) → S6 (drop `tags` from the emit
signatures, extend the AST lock) → S7 (canonical partitioned
numbering). A new label, `lock:src/apeGmsh/opensees/_internal/build.py`,
serialises S3 and S5 against other `build.py` work. Then K1-4…K1-8,
then V2d-4b (#1307), unless chain T interleaves.

Every slice runs: the 86 golden cells and their `.h5dump` files
byte-identical (S7 excepted, on purpose, and only its partitioned
cells); K1-3's pins (`test_tag_law_lock.py`, `test_tag_streams.py`,
`test_tag_law_replay_pins.py`, and `pytest -m subprocess
tests/opensees/subprocess/test_tag_determinism.py`); from S1, the plan
== tapped-stream oracle (the plan's `(kind, tag)` sequence equals the
stream the K1-3 tap records from the emit); and `emit-cost-gate` within
noise, with no re-baseline. The gate cell has only pre-planned element
specs, so the cost change is expected to be within noise.

### Consequences

- The five replay waivers on the tag-law ledger stay until the archive
  carries those tags: per row from K1-8 for the initial-stress and
  staged parameters, and with K2's flip for the reinforce ties. The
  plan does not touch replay.
- A correction to the #1361 inventory: the partitioned path plans
  elements once per emit; its two `allocate_element_tags` calls are
  exclusive branches.
- Flagged by both briefs for V2d-4b (#1307): the overwrite policy's
  dirty key `(primitive count, fem_hash)` misses edits that only add
  records (region, recorder, fix, mass, stage), so `model.h5` can go
  stale. It is not this design's to fix.
- #1446 (one region tag wasted per stage-claimed filtered recorder on
  the partitioned staged path) is fixed by S4, not in a fast lane.
