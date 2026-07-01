# Handoff — contact leftovers + EQ_Constraint H5 + docs sweep COMPLETE

Written 2026-06-29. Successor to `handoff_contact_stack_followups_complete.md`
(which landed the contact stack #741 + follow-ups). This records a session that
closed the **contact leftovers** (ADR 0073), the **EQ_Constraint H5** gap (ADR
0068 item 4), and a **docs sweep** — plus the one **fork-side** contract update.
Everything below is **on `main` / `ladruno` and green** — a state record + the
remaining queue, not a runbook.

## Snapshot

- apeGmsh `main` @ `be3e4909` (#764) — green (static-gates: ruff
  `src/apeGmsh/opensees` hard + mypy baseline 0; suite `-m "not live and not
  subprocess and not bench"`; lock-tests). No open apeGmsh PRs.
- Neutral H5 schema **2.24.0** (was 2.22.0 at session start): 2.23.0 (contact
  `-cell` column) → 2.24.0 (`/contact_planes` group). *(Bumped again to **2.25.0**
  by the follow-up edge-edge lane — apeGmsh #766, see queue item #1.)*
- Fork `nmorabowen/OpenSees` `ladruno` @ `9223d06f1` (#441).

## What shipped this session

### apeGmsh (`main`)
1. **#753** — `enforce="equation"` ties round-trip via the neutral lane; the H5
   deck emitter drops the `H5EquationConstraintDeviationWarning` (now **dormant**)
   — **ADR 0068 Open item 4 RESOLVED**. The "needs a schema bump" premise was
   stale: the equation tie is a resolved `InterpolationRecord` and the neutral
   lane already persists `enforce` + `weights` (schema 2.14.0), so it recovers via
   `FEMData.from_h5` → re-emit like embed/contact/reinforce. **No fork-feature
   carries an H5 deviation warning anymore.**
2. **#754** — unbreak `Deploy docs`. Root cause: apeGmsh #747 deleted `MortarDef`
   but left a `docs/api/constraints.md` mkdocstrings `:::` ref → mkdocs
   `BuildError`. Red since #747; the deploy uses non-strict `mkdocs gh-deploy
   --force`.
3. **#758** — api-flows mortar signature refresh (`flows.json` is HAND-CURATED;
   edit it then run `docs/api-flows/_embed.py`) + 2 broken TOC anchors
   (`--` slug bug). The 8 `internal_docs → src/ADR` strict-mode warnings are
   **deliberate** (`mkdocs.yml:10-13`) — do NOT "fix" them.
4. **#760** — `g.constraints.contact(..., cell=)` broad-phase `-cell` knob
   (neutral 2.23.0).
5. **#761** — `g.constraints.contact_plane(slave, normal=, point=, kn=, visc=,
   soft=)` rigid analytical-plane contact: full sibling stack of the contact
   generator; `/contact_planes` H5 group (neutral 2.24.0). Adversarial review
   caught a **CRITICAL** (now fixed): the partitioned serial-only guard
   (`apesees.py` ~2209) checked only `contacts`, not `contact_planes` → a
   plane-only MPI model silently dropped the plane AND auto-emitted a spurious
   `LadrunoContact` handler (unenforcing cross-partition MP constraints). Guard
   now fires on `contacts` OR `contact_planes`; regression test added.
6. **#764** — fixed stale `FEMData` comments claiming `contacts`/`embed_ties`
   aren't H5-persisted (both ARE: `/contacts` 2.21.0, `/embed_ties` 2.22.0).

### Fork (`nmorabowen/OpenSees`, base `ladruno`)
- **#440** — contract status reconcile (the four generators reinforce/embed/
  absorbing/monitor flipped "TO IMPLEMENT" → shipped). *(Not authored this
  session — a co-authored-by-Claude commit dated the same day; predates this
  thread's visible work.)*
- **#441** — added the missing **Contact** row to
  `Ladruno_implementation/ladruno_apegmsh_contract.md` (the whole contact
  subsystem was absent from the table). Docs-only.

## Key findings (carry forward — these reframed tasks)

- **Face-to-face contact FRICTION is already fully exposed** (`mu`/`eps_t`/
  `cohesion`/`tau_max`/`consistent_tan`/`geom_tan`). A request to "expose contact
  friction" is a no-op — the real remaining contact gap is the **edge-edge lane**.
- **`-epsTie` is intentionally NOT exposed** — a redundant alias for the `-epsN`
  penalty slot that `eps_n` already emits. Resolved, no action.
- The **contact subsystem is serial-only**: any contact verb (`contact` or
  `contact_plane`) under partitioned/MPI emit **fails loud** (`apesees.py` ~2209).
  The staged-partitioned path is covered by the same unconditional guard (it runs
  before the staged dispatch).
- Adversarial-review pattern (4 lenses → refute-by-default verify → synthesize)
  caught the only material defect of the session (the partitioned guard). Worth
  running on any H5-schema / multi-emitter change.

## Remaining queue (NOT done — for the next session)

### Contact
1. ~~**Edge-edge contact lane**~~ — **SHIPPED (apeGmsh #766, 2026-06-29).** The
   11 mortar-only flags (`-edgeedge` + the `-edge*` knobs, ADR-57 E2–E7) now ride
   `g.constraints.contact(edge_edge=, edge_kn=, edge_band=, edge_mu=, edge_kt=,
   edge_cohesion=, edge_tau_max=, edge_consistent_tan=, edge_soft=, edge_alm=,
   edge_aug_tol=)`. `ContactDef` mirrors the fork's two fail-loud gates
   (`edge_edge` requires `formulation="mortar"`; the `edge_*` params require
   `edge_edge=True` — the fork warns-and-ignores them otherwise). H5 round-trips
   via additive edge columns on `contact_payload_dtype` (neutral **2.24.0 →
   2.25.0**, presence-probed). Grammar verified char-by-char against
   `OPS_LadrunoContact`; adversarial review (4 lenses, refute-by-default) raised
   9 findings, 0 confirmed. Fork contract row flipped via OpenSees #442 (base
   `ladruno`). **With this the whole contact subsystem is exposed apeGmsh-side —
   no substantial contact gap vs the fork remains.**
2. ~~*(defense-in-depth only)* a **staged-partitioned** `contact_plane` fail-loud
   test~~ — **DONE.** Added
   `test_contact_plane_under_partitioned_staged_emit_fails_loud` to
   `tests/opensees/integration/test_contact_partition_fail_loud.py`: a STAGED +
   partitioned plane-only model still raises `BridgeError` (the serial-only guard
   in `_emit_partitioned` sits before the staged dispatch, so the staged path
   can't let a plane slip through). Confirms the review's claim with a regression.

### EQ_Constraint / DRM (ADR 0068, ADR 0066)
3. **DRM integration test (P5 capstone)** — a non-matching soil/structure
   interface under explicit DRM (the real `enforce="equation"` use case). Needs an
   openseespy **built from current `ladruno`** (the local build is stale) + a full
   SSI model.

### Ladruno catch-up — small/deferred (from
`~/.claude/.../memory/project_ladruno_fork_catchup_2026_06.md`)
4. `/opensees` **deck-zone replay** for reinforce/contact/embed/equation —
   **reinforce leg SHIPPED (apeGmsh #771).** `OpenSeesModel.build("tcl"/"py"/
   "live")` now re-emits `LadrunoEmbeddedRebar` ties: `_replay_into` gained a
   step **8b** that re-emits them from the neutral-zone `fem`
   (`fem.elements.reinforce_ties`, already passed in for connectivity
   rehydration) — **no dedicated `/opensees` deck record, no schema bump** (the
   cleaner re-emit-from-neutral approach, superseding the plan's
   dedicated-record design). Tie element tags are seeded past the max replayed
   element tag; the bond name→tag map threads from `OpenSeesModel._names`. The
   `h5` re-emit caller passes no `fem` so it is skipped (ties persist via the
   neutral zone there). **Still a documented follow-on:** `_replay_into` replays
   NO other MP-constraint family (equalDOF/rigidLink/rigidDiaphragm/embeddedNode/
   contact/embed/equation) — reinforce is the first/only; canonical recovery for
   the rest stays `FEMData.from_h5` → forward re-emit. Plan = `plan_rebar_p5.md`
   §A4-full.
5. ~~`s.mortar` / `s.tied_contact` **stage-bound claims**~~ — **RESOLVED (nothing
   to build).** `s.tied_contact` is already SHIPPED (commit `ae9c2131`, ADR 0034
   follow-on) + tested (`test_stage_tied_contact_claim.py`), part of the full
   `s.*` claim family. `s.mortar` is **intentionally out of scope** (explicit
   code comment, `apesees.py` ~7911): as of ADR 0073 `g.constraints.mortar`
   delegates to the fork contact-tie → a `ContactRecord` on
   `fem.elements.contacts` (serial-only), NOT a claimable `SurfaceCouplingRecord`
   — there is no stage-claimable record. The queue item was stale.
6. standalone **EnergyBalance text recorder** — energy already lands via
   `.ladruno -G energy`.
7. ~~**g.reinforce R3c** `-corot -shapeB` large-rotation leg.~~ — **SHIPPED
   (apeGmsh #769, 2026-06-29).** `g.reinforce(corot=True)` co-rotates the bar
   axis (ADR 20 §10.5); the resolver computes the host-element-tag-free
   `-shapeB` point-B weights (step `0.05·host_radius` along the bar, inverse-map
   into the same host) and round-trips them via additive `corot`/`shape_b`
   columns (neutral schema **2.25.0 → 2.26.0**, presence-probed). The R3
   reinforcement runway is complete; only R4 docs/skill/recipe remain.
8. **partitioned/MPI emit** of reinforce + embed ties (single-process only today).
9. **LogStrain2D** (ND_TAG 33016) — RESERVED, not built fork-side; nothing to
   expose yet.

## Env / conventions (carry forward)

- Interpreter for tests/scripts: `C:\Users\nmora\venv\opensees_venv\Scripts\
  python.exe` with `LADRUNO_OPENSEES_QUIET=1`; run pytest with
  `PYTHONPATH=<worktree>/src` (the editable install resolves to the MAIN repo
  `src/`, not the worktree).
- Fork ground truth for contact grammar: `OPS_LadrunoContact` (~380) /
  `OPS_LadrunoContactPlane` (~862) in `SRC/interpreter/OpenSeesOutputCommands.cpp`
  on the `ladruno` worktree
  `C:\Users\nmora\Github\OpenSees_Compile\OpenSees\.claude\worktrees\infallible-johnson-a2210c`.
  Verify any flag against the parser AND (when a build has it) run it live.
- CI gates: static-gates (ruff `src/apeGmsh/opensees` HARD + mypy baseline 0) +
  suite + lock-tests. The hard ruff gate is opensees-only; `src/apeGmsh/core`
  carries 4 pre-existing `HEX8_TO_6_TETS`-style unused-import warnings (not
  CI-gated, untouched).
- **`git add -A` HAZARD:** the suite regenerates `docs/benchmarks/
  cross_rank_constraint_cost.md` (timestamps/timings) and the worktree carries an
  untracked `.cache/`. Stage explicitly (`git add src tests CHANGELOG.md` + named
  files); never `-A`. Both leaked into a commit this session and had to be
  stripped.
- Neutral-schema bump recipe (used twice this session — cell column, contact_plane
  group): `*_payload_dtype()` in `mesh/_record_h5.py`; `_write_*`/`_encode_*`/
  `_read_*`/`_decode_*` in `mesh/_femdata_h5_io.py`; wire write/read entry points +
  `ElementComposite(...=)`; bump `NEUTRAL_SCHEMA_VERSION` + a history comment; bump
  `tests/fixtures/schema.py` (`NEUTRAL_CURRENT` / `NEUTRAL_PRIOR_MINOR`).
- **Process wart to avoid:** the `guppi/docs-housekeeping` branch got opened as TWO
  PRs (#756 + #758), both merged (idempotent, so `main` is fine). One branch → one
  PR.
