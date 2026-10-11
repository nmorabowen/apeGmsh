# ADR 0121 — Charter v2: B with amendments, one engine policy, a mechanized freeze

**Status:** Accepted (2026-10-11; accepted by the maintainer on #1193,
comment 6112930400)

**Owner:** nmora

**Evidence:** two architect briefs written independently on one brief,
both grounded on `origin/main` `63f4a158`: `prog-architect-opus` (Opus at
max effort) and `prog-architect-fable` (Fable at xhigh, an exception the
maintainer stated on 2026-10-11 because an irreversible charter design
benefits from a second model family). The orchestrator reconciled them on
#1193 (comment 6112293477); the briefs themselves are comment 6112306488.
This ADR cites them and does not copy them.

**Supersedes in part:** [ADR 0102](0102-warrant-the-snapshot.md)
D2 (the freeze) and D5 (the bucket table, including its Warranted row).
0102 carries the amendment and a pointer here, both added when this ADR
was accepted. 0102's D1,
D3, D4, D6 and D7 are not touched at all, and stay Proposed (maintainer
ruling 4, 2026-10-11).

**Program slice:** #1678 (link P0.2; chain issue #1193).

Each decision is marked *(ratified)* when the maintainer has already ruled
on it and this ADR only records the ruling, or *(proposed)* when accepting
this ADR is the ruling. *(proposed; direction ruled 2026-10-11)* marks a
proposed decision whose open question the maintainer has already answered
(see "Maintainer rulings" below). The maintainer accepted the ADR on
2026-10-11; the markers are kept as written.

## Context

Three documents govern apeGmsh today, and they disagree.

The first is [`architecture/charter.md`](../charter.md), the bridge's
charter. Its mission promises "three OpenSees emit targets" and a bridge
that writes the deck and stops. Two facts on `main` contradict it: HDF5 is
a fourth target ([ADR 0011](0011-h5-as-fourth-emit-target.md),
`emitter/h5.py`), and the live target cannot run stages
(`LiveOpsEmitter.caps.supports_stages` is False). AGENTS.md still sends
every agent to it.

The second is ADR 0102, Proposed since 2026-08-30. It froze new families in
prose (D2) and defined "Warranted" in a hand-filled table (D5). The prose
freeze did not hold: ADRs 0117 (assembly), 0119 (ground-spring bed) and
0120 (multi-rank routing) all landed after it. Its own acceptance clause is
unmet, too. `CHANGELOG.md` is 869 kB under one Unreleased heading, and
`architecture/README.md` still reports Phase 8.6, 81 primitives and schema
2.2.0. So 0102 cannot be accepted as written without a release cut, and
accepting it would quietly ratify D6 (a named human on every row) and D7
(drop "any solver") along with the freeze.

The third is the expert panel. It decided the fork question 9–0 as "B with
amendments" (`plan_expert_panel_2026-09.md` §0 K1, §1), and its first open
decision (§9.1) asked for exactly this ADR: ratify B, and amend 0102 D2/D5
into a mechanized freeze and the rule "fork green in `live-fork`; stock
honest or refused". On #1192, panelists P3 (R6) and P8 (R7) added that the
freeze may lift only as fast as `live-fork` can verify. The P0.3 rulings
that followed live only as comments on #1193. This ADR is the one place
they are written down together.

What `main` holds, probed on `origin/main` `f65d55dd` (2026-10-11):

| Claim | State on `main` |
|---|---|
| One fork signal | Shipped. `opensees/_target.py::BackendInfo` reads only `ladrunoBuild()`, through `backend_info_of`, which never raises. `OpenSeesTarget.mode` defaults to `"auto"`, and `resolve_kind` returns `info.kind` for it |
| A pinned fork build and a nightly lane | Shipped. `FORK_PIN` names `ladruno-v20261005` (`011b0258`). `.github/workflows/live-fork.yml` runs nightly as a non-required check and asserts that `ladrunoBuild()` equals the pin. The last five scheduled nights (2026-10-07 to 10-11) were green |
| `live-stock` required, on Python 3.12 | Shipped. `scripts/land_pr.py::REQUIRED_CHECKS` names it, and `tests.yml` runs it on 3.12 |
| Capabilities as data | Shipped. `emitter/caps.py::TargetCaps` (one frozen value per emitter) and `SolveStamp` (`will_solve`, `requires`) |
| Stock refusals | Shipped. `emitter/live.py::_FORK_ONLY_ELEMENTS` and `_FORK_ONLY_INTEGRATORS` gate the live path, and `opensees_model.py::refuse_live_requires` refuses a stamp that requires the fork on a stock build. Pinned by `tests/opensees/unit/test_bezier_run_gate.py` (the element gate), `tests/test_capability_map_drift.py` (both sets equal the documented map) and `tests/opensees/h5/test_h5_solve_stamp.py` (`refuse_live_requires`) |
| Deck provenance | Partial. `emitter/tcl.py::provenance_stamp` stamps decks built for a pinned target. `mode="auto"` decks stay unstamped by ruling (#1511; follow-up #1603, open). `results.h5` carries `opensees_backend` and `opensees_build` from DomainCapture and footfall |
| Two-sided divergence roster | Shipped after the briefs were written. [`architecture/fork-divergence-roster.md`](../fork-divergence-roster.md) holds rows D1–D9 and is locked by `tests/opensees/unit/test_fork_divergence_roster.py` (F3-a, #1666, merged as #1672 on 2026-10-11). Both briefs recorded it as not shipped |
| `ladrunoSchema` | Not shipped (F3-b, #1667, open) |
| Family completeness | Shipped as completeness, not as a freeze. `tests/test_family_completeness.py` walks `tests/families.py::FAMILIES` (19 families, ratchet `EXCEPTIONS_BASELINE = 55`). The Protocol's method count is locked by `tests/opensees/contract/test_verbs_lock.py` ([ADR 0114](0114-the-archive-is-the-program.md) D2) |
| A freeze | Absent. 0102 D2 is prose. `land_pr.py`'s `freeze:<file>` labels are file-keyed split windows, and a new family can land inside a file that already exists |

Whether `auto` decks should be stamped is #1603's question. This ADR does
not decide it.

## Decision

### D1 (ratified) — Fork strategy: B with amendments

Stock OpenSees stays a first-class, executable profile. `live-stock` stays a
required check on Python 3.12; it is the only required lane that executes
OpenSees at all. The Ladruno fork is where the investment goes. The default
target is AUTO, resolved from the imported binary through `BackendInfo`.
Going fork-only is a review trigger (K3), fired by the conditions in panel
§1, and never an automatic flip. Provenance is as shipped (table above).

Sources: panel K1 (9–0) and K3 (8–1); P0.3 item 5, ratified on 2026-10-05.

### D2 (ratified) — Run layer: emission is the official path

Emission (Tcl, py and `model.h5`) is the official and recommended path
(P0.3 item 1: K24 is stage-only), and the archive is the program (ADR
0114). Live execution stays supported at its current scope, for unstaged
models. No staged-live features, tests or docs are added. `will_solve`
marks a deck as solvable, not a live session.

The other P0.3 rulings are recorded here so that one document holds them
all:

- **K19.** `ModelData` is cut only where nothing is lost. The inventory
  (`internal_docs/program/x1_modeldata_inventory.md`) found capabilities
  with no replacement, so `ModelData` stays (#1506).
- **K8.** E2 (`TargetCaps`) was built inside K1, as slice K1-5. E1 waits
  for its trigger.
- **`strict: false`** stays on `main`, with the red-main trip-wire that T
  reads weekly.
- **Fork rules (a)–(e)** of P0.3 item 5 stand, with (a) as the maintainer
  restated it on 2026-10-11 (#1198): fork work runs in worktrees cut from
  `origin/ladruno`; the base checkout is never edited, and
  `OpenSees_Compile` is never touched.

### D3 (proposed; direction ruled 2026-10-11) — One engine policy: fork green, stock honest or refused

This replaces 0102 D5's bucket table, Warranted row included. Every member
that behaves differently on the two engines is in exactly one of four
states.

- **Refused.** On a stock build, apeGmsh raises at build, emit or
  live-emit time, *before* `analyze`, with an error that names the
  fork, and a `pytest.raises` test pins the refusal. The run is refused;
  the domain is not necessarily untouched. The element gate
  (`LiveOpsEmitter._element_fork_gated`) issues `ops.element` first and
  raises when the build rejects or silently drops it, so a stock domain may
  be left partly built. The integrator gate raises before its call. Emission
  is never gated: a Tcl or py deck for a fork class is written on any build.
- **Honest.** The stock result meets a closed-form or cross-engine oracle in
  `live-stock`. A known divergence has a row in the divergence roster that
  names the correct engine, and the live path either refuses it or warns
  with a named category.
- **Fork green.** The member's own test *passed*, not skipped, in
  `live-fork` on the build `FORK_PIN` names.
- **Silent.** Anything else. A silent member is never Warranted.

A member is Warranted when its warrant row is green: fork green on the fork
side, refused or honest on the stock side. Every other member is Inventory.
Promoting a member is work (a refusal, an oracle, a test that passes),
never a status edit. Warrant is derived from gates, and nobody fills in a
table.

Two kinds of green are kept apart (ruling 3). The *lane* is green when
`live-fork` passes with its `ladruno_mkl` skips-with-reason counted, the
panel's definition and the one the K3 30-day trigger reads. A *member* is
fork green only when its own test passed; a skip never counts for it. A
warrant test that skips in its own lane fails that lane. How the lane
enforces this, an environment switch or a marker, is left to PR-4.

### D4 (proposed; direction ruled 2026-10-11) — The freeze is a lock

This replaces 0102 D2. A census lock in the `lock-tests` job holds four
registries, the ones that map onto 0102 D2's list:

| Registry | Source on `main` | What 0102 D2 froze |
|---|---|---|
| Primitive families | `tests/families.py::FAMILIES` (the family names) | new families |
| Constraint kinds | `src/apeGmsh/_kernel/records/_kinds.py::ConstraintKind` | a new constraint kind |
| Studio MCP tools and habitat verbs | the `@mcp.tool` functions in `studio/mcp.py`; `studio/_profile.py::HABITAT_TOOLS` | a new MCP tool or habitat verb |
| Emitter verb families | the `family` values of `emitter/verbs.py::VERBS`, drawn from `verbs.py::FAMILIES` | new families, on the command side |

The remaining bullet of 0102 D2, viewer ontology, is already decided by
ADRs 0112 and 0116 and is not locked here. Its last bullet, the typed
wrapper for the next Ladruno class tag, is replaced by the member rule
below.

Growing a registry needs an ADR whose Status, as
`scripts/adr_index.py::parse_adr` reads it, is Accepted, and the new census
entry cites that ADR. Two checks in `scripts/land_pr.py` keep the exception
a maintainer act. It refuses (a) a diff that sets any ADR's Status to
Accepted, whether it changes an existing ADR or adds a new ADR file born
with Status Accepted, and (b) a diff that edits the census baseline on a PR
without the `human-gate` label. The two close different holes: (a) stops
an agent from accepting the ADR that admits its own exception, and (b)
stops it from editing the baseline around one.

Like every `land_pr.py` check, these bind only PRs landed through the
script. The maintainer's own merge in the GitHub web UI bypasses them, and
that is by design: the web merge is the exception path, and it is a human
act. An agent lands through `land_pr.py` (AGENTS.md, "How work lands").

New *members* of existing families are not frozen. As the maintainer ruled
(ruling 1), a new fork member (a material, an element, an integrator)
merges with a stock refusal whose error names the fork, and with a
warrant-row test. The refusal is the member's own: a `_FORK_ONLY_*` entry or
a raise at build time. A `requires` stamp does not serve, because
`requires` is set per `VERBS` row (`verbs.py::_FORK`), not per member. The
member is Warranted only once that test *passes*, not skips, in the nightly
`live-fork`; until then it is Inventory. That is how warrant moves only as
fast as `live-fork` verifies, without making a merge wait on a non-required
lane. The fork members that predate the freeze, roughly 48 by the
architects' count (PR-2's census fixes the exact number), stay Inventory
with no deadline, tracked in X4 (ruling 5). Fixes, fail-loud guards,
deletions and stock primitives are not frozen either.

A full stock/fork classification of every primitive is deliberately not
part of this lock. Classified by hand before `ladrunoSchema` exists, it
would be a second roster; F3-b and F4 derive it from the binary instead.

### D5 (proposed) — How ADRs are amended

A decision is never rewritten. Errata go in a dated `## Amendment` section,
as ADRs 0113 and 0114 already do. A changed decision goes in a new ADR that
names the clauses it supersedes. Pointers and Status edits on the
superseded ADR land in the PR that *accepts* the new one, because a
Proposed ADR supersedes nothing. This ADR followed its own rule: its
0102 amendment and `charter.md` edits landed in the PR that accepted it.

### D6 (proposed) — Charter principles, each with its gate

The strains report (§5) proposed eight principles for the next phase. Five
of them are adopted here: its principles 1, 2, 5, 7 and 8. Principle 6 is
already ADR 0113. Principles 3 and 4 are excluded (below). Warrant, from D3,
is new. Each of the six charter principles names the gate that enforces
it; the strains number is in parentheses.

1. **Local-change safety** (§5.1). Adding a member is one registration,
   and a completeness gate fails closed when something does not
   participate. Gates: C1 (`tests/test_family_completeness.py`) and the
   verbs lock.
2. **Fail closed at seams** (§5.2). An unknown class, capability or stream
   raises or warns loudly. Gates: the `scripts/check_quirks.py` rules, and
   fork rule (e), which gives every divergence a roster row.
3. **Variants are data** (§5.5). One capability model, read from the
   binary. Gates: `TargetCaps` and `BackendInfo`, locked by
   `tests/opensees/contract/test_target_caps.py` and
   `tests/opensees/unit/test_backend_info.py`.
4. **State is derived, not narrated** (§5.7). Gates: the `adr-number` and
   `doc-path` quirk rules, and the generated ADR index
   (`scripts/adr_index.py --check`).
5. **Retire by deletion** (§5.8). A retirement gets a ledger entry and an
   exit criterion. Gate: X4's ledger, still pending.
6. **Warrant** (new). D3 of this ADR.

Compatibility (§5.6) is already decided by
[ADR 0113](0113-compatibility-is-a-floor-per-zone.md), whose D5 rejects
"additive changes do not bump", so it is cited and not restated. So are the
archive (ADR 0114) and the viewers
([ADR 0112](0112-files-are-the-model-and-a-read-only-app.md), with
[ADR 0116](0116-render-technologies.md)), which §5 did not list. The two
excluded principles are not charter material. Semantics surviving lowering
through explicit IRs (§5.3) was rejected by the panel (S13: no DeckProgram,
ColumnSpec or picture IR). Machine-checked boundaries (§5.4) stay a
measurement, because PROGRAM.md §8 keeps structure KPIs out of the hard
gates.

### D7 (proposed; direction ruled 2026-10-11) — `charter.md` stays the bridge charter

`architecture/charter.md` keeps its job: it binds the bridge, and CI cites
its principles by number ("charter P12" in `tests.yml`). As the maintainer
ruled (ruling 2), the accepting PR adds a scope header that points here for
library-level policy, and fixes its two stale facts: HDF5 is a fourth emit
target, and live runs flat (unstaged) models only. There is no full
rewrite: P1–P14 stay as they are. Where one conflicts with this ADR, this
ADR wins.

## Invariants

Each one is testable. "Holds" means a test on `main` already pins it; the
others become true in the slice named. Invariants 7 and 9 share one
cut-off, *the freeze point*: the merge commit of PR-2, which commits the
census baseline. Entries and members already on `main` at that commit are
grandfathered.

1. `OpenSeesTarget().mode == "auto"`, and for an auto target
   `resolve_kind(info) == info.kind`. Holds
   (`tests/opensees/unit/test_opensees_target.py`).
2. `backend_info_of(ops).kind == "fork"` if and only if
   `ops.ladrunoBuild()` returns a 40-hex sha, and the probe never raises.
   Holds (`tests/opensees/unit/test_backend_info.py`).
3. `"live-stock"` is in `land_pr.REQUIRED_CHECKS` and `"live-fork"` is not;
   the `live-stock` job runs Python 3.12. `tests/test_land_pr.py` already
   replays a failed and a missing `live-stock`; PR-3 adds the explicit
   membership and version assertions.
4. `LiveOpsEmitter.caps.supports_stages` is False. Holds
   (`tests/opensees/contract/test_target_caps.py::test_caps_values_per_emitter`).
5. A `SolveStamp.requires` holding `"fork"` makes `build("live")` raise on
   a stock `BackendInfo` before the live emitter is constructed. Holds
   (`tests/opensees/h5/test_h5_solve_stamp.py::test_build_live_refuses_before_touching_the_domain`).
6. Each frozen registry equals its census. An extra registry member, or a
   census entry with no member behind it, fails `lock-tests`. PR-2.
7. Every census entry added after the freeze point cites an ADR file whose
   parsed Status is Accepted; an entry that cites a Proposed ADR fails.
   PR-2.
8. `land_pr.py` refuses a diff that sets an ADR's Status to Accepted,
   whether by changing an existing ADR or by adding a new ADR file born
   Accepted, and a diff that touches the census baseline on a PR without
   `human-gate`. PR-3.
9. A fork member added after the freeze point ships, in the same PR, a
   stock refusal of its own (a `_FORK_ONLY_*` entry or a build-time raise)
   and a warrant-row test. PR-4.
10. A member counts as Warranted only when its own warrant test passed in
    its lane, and a warrant test that skips in its lane fails the lane.
    Every *refused* row's test contains `pytest.raises`, checked on the
    AST. PR-4.
11. Every divergence-roster row names `stock`, `fork`, `policy` or
    `pending #<n>` as the correct engine, and its pinning tests exist.
    Holds (`tests/opensees/unit/test_fork_divergence_roster.py`).
12. `python scripts/adr_index.py --check` passes; ADR 0102 parses as
    Proposed (ruling 4) and 0121 as Accepted. No test pins the second
    half: it is checked at review until PR-3's Accepted refusal (invariant 8) makes it
    mechanical for every PR landed through `land_pr.py`.
13. `charter.md` claims neither three emit targets nor a stage-capable
    live target. Holds since this ADR's acceptance; checked at review.

## Slices

**PR-1** is #1678 (PR #1681): this ADR, the regenerated index, one
changelog fragment and, at acceptance, the 0102 amendment and the
`charter.md` scope header and two fixes. It is labelled `semantic` and
`human-gate`. The rest are follow-up
slices on #1193, each of at most about six files, started when the
maintainer clears them.

- **PR-2, the freeze lock.** The census, the lock test for invariants 6
  and 7 wired into `lock-tests`, a `gates.md` row and a fragment. Its
  oracle is mutation: a new `ConstraintKind` constant, a dummy
  `@mcp.tool`, a family added to `FAMILIES` and a census entry that cites
  a Proposed ADR must each turn it red, and reverting each must turn it
  green.
- **PR-3, the `land_pr.py` checks.** Invariant 8, with replay cases in
  `tests/test_land_pr.py`: a recorded Proposed-to-Accepted diff, a new ADR
  file born with Status Accepted, and a baseline edit without
  `human-gate`.
- **PR-4, warrant rows and lane wiring.** One refused exemplar and one
  honest exemplar, and the skip-fails-the-lane rule in both live lanes. It
  builds on the F3-a roster, which has now merged.
- **PR-5, pointers.** AGENTS.md, `architecture/README.md` and
  `architecture/agent-onboarding.md` point here.

Owned elsewhere and not decided here: `ladrunoSchema` (F3-b, #1667), the
census derived from it (F4), stamping `auto` decks (#1603), and X4's
retirement ledger.

## Alternatives rejected

The briefs agreed on the substance and differed on eight points; each
resolution is part of the decision above.

**Changing `charter.md`.** The Opus brief rewrote it in the acceptance PR,
with a v2 mission, the principles and a P1–P14 status column. The Fable
brief kept it, with a scope header and two corrected sentences, edited in
PR-1. A rewrite re-decides live rules, breaks CI's citations of principles
by number, and contradicts the Opus brief's own "no P1–P14 rewrites"; an
edit in PR-1 changes a norm while this ADR is still Proposed. The minimal
edit, made at acceptance, keeps the best of each (D7). The maintainer
ruled for it (ruling 2).

**Admitting a new fork member.** The Opus brief required a warrant row
plus a green `live-fork` run on the PR head, which forces a `FORK_PIN` bump
before every new class and needs a head-run script. It also makes a merge
depend on a lane that fork rule (d) keeps non-required. The Fable brief
froze nothing, provided stock refused and an effect test shipped, which
drops the panel's rule that warrant moves only as fast as `live-fork`. D4
takes the middle: refusal and warrant row at merge, Warranted only once
green. The maintainer ruled for it (ruling 1).

**Which registries are frozen.** The Opus brief classified every primitive
as stock or fork; the Fable brief froze `FAMILIES`, the constraint verbs,
MCP tools and `VERBS` families. D4 takes the union that maps onto 0102 D2's
list and defers the full classification to F3 and F4, since a hand
classification before `ladrunoSchema` is a second roster.

**One exception check or two.** The Opus brief refused Accepted flips in
`land_pr.py`; the Fable brief refused baseline edits without `human-gate`.
Each leaves the other's hole open, so D4 keeps both.

**Whether `ladruno_mkl` skips count.** The Fable brief counted them; the
Opus brief did not. They count for the lane and not for a member (D3),
as the maintainer ruled (ruling 3).

**When the 0102 amendment lands.** The Fable brief added a dated amendment
section while this ADR was still Proposed. That would amend 0102 on the
authority of a Proposed ADR, which D5 forbids; it landed at acceptance.

**"Boundaries machine-checked" as a principle.** The Fable brief included
it. PROGRAM.md §8 says structure KPIs are never a hard gate, so it stays a
KPI (D6).

**A named strict-mode switch.** The Opus brief specified
`APEGMSH_WARRANT_STRICT=fork|stock`. The mechanism belongs to PR-4; this
ADR states only the invariant (invariant 10).

Four options that the briefs rejected outright are worth recording. A
**prose or label freeze** is what 0102 already tried. The prose did not
hold, and `freeze:` labels are file-keyed: a new family lands inside a file
that already exists without touching a frozen path. Like D4's checks, the
labels bind only PRs landed through `land_pr.py`, and the maintainer's web
merge bypasses both; that shared property is by design, since the web merge
is the maintainer's exception path. The labels fail on being file-keyed,
not on the bypass. A **quirk rule as the freeze** is AST-only,
cannot see C1's family walk, and skips when there is no base to diff.
**Accepting 0102 as amended** needs a CHANGELOG cut that CI refuses outside
`--assemble`, and would ratify D6 and D7 without anyone deciding them.
**CODEOWNERS** does not work here: agents act through the maintainer's
account, so a required owner review could never be given.

## Maintainer rulings (2026-10-11)

On each of the design's five questions the maintainer chose the
recommended option (#1193, comment 6112664614); the decisions above state
them as decided. The maintainer then accepted the ADR the same day.

1. **Admitting a fork member** (D4). It merges with a stock refusal naming
   the fork plus a warrant-row test, and is Warranted only when that test
   passes, not skips, in the nightly `live-fork`. Rejected: a head-SHA
   `live-fork` run at merge, or freezing new fork wrappers until the
   lane's 30-day green.
2. **`charter.md`** (D7) stays the bridge charter, with a scope header and
   the two factual fixes at acceptance. Rejected: a v2 rewrite.
3. **`ladruno_mkl` skips** (D3). They count for the lane's green, never for
   a member's warrant.
4. **The rest of ADR 0102** (D1, D3, D4, D6, D7) stays Proposed. Rejected:
   accepting it after the release cut, or rejecting it.
5. **The grandfathered fork members** (D4) stay Inventory with no
   deadline, tracked in X4. Rejected: a deadline for their warrant rows.

## References

- Chain issue #1193: the reconciled design (comment 6112293477), the two
  briefs (6112306488), and the P0.3 rulings (5897056544, 5981886601,
  6004282361), and the maintainer's rulings on this ADR's five questions
  (6112664614). Fork rule (a) as restated: #1198. Panel rules P3 R6 and P8
  R7: #1192. This slice: #1678.
- `internal_docs/plan_expert_panel_2026-09.md` §0 (decisions at a glance),
  §1 (the fork: "B with amendments") and §9.1 (the Charter v2 ask).
- `internal_docs/plan_architectural_strains_2026-09.md` §5 (design
  principles for the next phase).
- `internal_docs/program/PROGRAM.md` §4 (human gates), §8 (KPIs, never
  hard gates) and §10 (baseline decisions).
- [ADR 0102](0102-warrant-the-snapshot.md), superseded in part on
  acceptance; [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md);
  [ADR 0113](0113-compatibility-is-a-floor-per-zone.md);
  [ADR 0114](0114-the-archive-is-the-program.md);
  [ADR 0116](0116-render-technologies.md).
- [`architecture/charter.md`](../charter.md) and
  [`architecture/fork-divergence-roster.md`](../fork-divergence-roster.md).

## Implementation log

- 2026-10-11: PR #1681 (slice #1678) opened the ADR as Proposed; the
  maintainer's rulings on its five questions were folded in (#1193,
  comment 6112664614).
- 2026-10-11: **Accepted** by the maintainer (#1193, comment 6112930400)
  in PR-1. As D5 requires, the same PR appended the dated amendment and
  pointer to ADR 0102 (Status unchanged, Proposed) and gave `charter.md`
  its scope header and two fixes; P1–P14 are unchanged. The `land_pr.py`
  Accepted refusal (PR-3) did not exist at this landing, so the
  maintainer's ruling is the explicit exception.
