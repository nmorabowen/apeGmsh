# Handoff — contact-stack backlog + follow-ups COMPLETE

Written 2026-06-25. Successor to `handoff_pr_merge_sequence.md`. That doc was
the merge runbook for landing the contact stack (#741) + making `main` green;
this one records the **follow-up backlog** that ran on top of it. Everything
below is **landed on `main` and fully green** — this is a state record + the
remaining open items, not a runbook.

## Snapshot

- `main` @ `9b3750e1` (#748) — **GREEN on both axes** (static-gates: ruff
  `src/apeGmsh/opensees` hard + mypy baseline 0; suite: pytest
  `-m "not live and not subprocess and not bench"`; lock-tests). No open PRs.
- Neutral H5 schema is now at **2.22.0** (was 2.20.0 at the start of the
  session). Two bumps this session: 2.21.0 (contact), 2.22.0 (g.embed).
- ADR 0073 (`src/apeGmsh/opensees/architecture/decisions/0073-contact-generator.md`)
  is the single source of truth for every decision below — its **Scope /
  deferred** and **Consequences** sections were amended as each item landed.

## What shipped this session (6 PRs, in order)

The original 4-item backlog from the prior handoff, then 2 follow-ups it surfaced:

1. **#742** `fix/730-live-ops-cache-leak` — fixed the lone #730 suite flake
   (`test_emitter_partition_open_close.py::test_live_emitter_restores_real_ops_after_close`).
   Root cause: the process-wide `_OPS_CACHE` in `emitter/live.py` leaked a
   backend across tests, defeating the `fake_ops` fixture order-dependently.
   Fix: the fixture now monkeypatches `live._get_ops` directly (matches the
   sibling `test_parallel_runtime_fallback.py`). **Restored fully-green main.**

2. **#744** `feat/contact-extensions-report` — re-ported #723's contact
   extension modifiers (`-soft`/`-visc`/`-consistanttan`/`-geomtan`) onto the
   landed #741 stack and exposed them via `g.constraints.contact(..., soft=,
   visc=, consistent_tan=, geom_tan=)`. **Supersedes #723** (now CLOSED — it
   pre-dated the adversarial-review hardening and was based on the old
   contact-generator branch). Each flag verified against the fork
   `OPS_LadrunoContact` option loop + run live on the fork. Added the two
   fail-loud gates the fork enforces that #723 missed: `-soft` needs a base
   penalty (kn/eps_n) + excludes `-tie`; `-visc` excludes `-tie`. Plus
   `-geomtan` NTS-only and SOFSCL coupled-stability warnings.
   - Put through its **own 17-agent adversarial-review workflow** (6 dimensions,
     double-verified). 8 confirmed findings → all fixed; 3 refuted. The
     critical one: a numeric `kn` + trailing extension flags emitted a stream
     the fork aborts (the bare-`kn` parser-abort class, reintroduced). Fixed by
     **always padding the `kn kt mu` triple for a numeric kn** — immune to which
     trailing tokens follow. Run-verified live on the fork.

3. **#745** `feat/mortar-delegate-contact-tie` — `g.constraints.mortar()` is
   now a **deprecated convenience alias** (curated wrapper) delegating to
   `contact(formulation="mortar", tie=True, ...)`. Breaking change (was a
   `NotImplementedError` stub): returns `ContactDef`; semantics flip
   (Lagrange-multiplier tie → ALM penalty contact-tie, fork-only at run time);
   `dofs`/`integration_order` dropped (TypeError if passed); `outward` now
   required. Design chosen with the maintainer (curated wrapper over full
   pass-through). Own focused review → 0 code defects; fixed the docstrings the
   change falsified.

4. **#746** `feat/contact-h5-persistence` — native H5 round-trip of contact
   records via the **neutral** FEMData zone (`/contacts` group,
   `contact_payload_dtype`, schema **2.21.0**), mirroring reinforce/embed.
   Tri-state penalties (kn/eps_n/eps_t = None|"auto"|numeric) and soft
   (None/off|bare-True|numeric) use a `*_mode` flag; faces are flat-int +
   stride. Deck-zone no-op silenced. Proven by a fork-gated
   `to_h5 → from_h5 → re-emit → runs on the fork` end-to-end test.

5. **#747** `chore/remove-dead-mortardef` — removed the now-dead `MortarDef`
   dataclass + `ConstraintResolver.resolve_mortar` (the never-built
   ∫ψ·N dΓ Lagrange path, unreachable once `mortar()` started delegating) and
   all re-exports / dispatch entries. `_resolve_face_both` kept (TiedContactDef
   uses it). `ConstraintKind.MORTAR` / `SurfaceCouplingRecord` (the separate
   **record-level** concept, exercised by H5 round-trip tests) untouched.

6. **#748** `feat/embed-h5-persistence` — native H5 round-trip of `g.embed`
   ties (`LadrunoEmbeddedNode`) via the neutral zone (`/embed_ties` group,
   `embed_tie_payload_dtype`, schema **2.22.0** — the isotropic sibling of
   `/reinforce_ties`). Deck-zone no-op silenced. Fork-gated end-to-end test.
   **With this, no emitter raises `H5FeatureDeferredWarning` any more**
   (g.reinforce / g.constraints.contact / g.embed all persist via the neutral
   zone). The class + its back-compat alias `H5ReinforceDeviationWarning` are
   retained for future deferrals.

## Final state of the ADR 0073 program

| Feature | Emit (tcl/py/live) | Neutral H5 round-trip | OpenSees deck-zone replay |
| --- | --- | --- | --- |
| `g.constraints.contact` (NTS / mortar) | ✅ | ✅ (2.21.0) | ❌ deck-replay follow-on |
| contact extensions (soft/visc/…) | ✅ | ✅ (rides ContactRecord) | ❌ (same) |
| `g.constraints.mortar()` → contact-tie | ✅ (deprecated alias) | ✅ (it IS a ContactRecord) | ❌ (same) |
| `g.embed` (LadrunoEmbeddedNode) | ✅ | ✅ (2.22.0) | ❌ deck-replay follow-on |
| `g.reinforce` (LadrunoEmbeddedRebar) | ✅ | ✅ (2.15.0, prior work) | ❌ deck-replay follow-on |

The neutral-zone round-trip is the supported, complete path for all of these
(`FEMData.from_h5` → `apeSees(fem).tcl()/py()/run()` re-runs the forward
emit). The `/opensees` **deck** zone deliberately carries no record for these
fork features — that is the only remaining H5 gap (see below).

## Remaining open items (NOT part of this program; surfaced for the next session)

1. **EQ_Constraint (`enforce="equation"`) H5 persistence** — ADR 0068, Open
   item 4. Still genuinely deferred: the H5 emitter no-ops the equation route
   and raises its **own** `H5EquationConstraintDeviationWarning` (distinct from
   the now-dormant `H5FeatureDeferredWarning`). This is the only fork-feature H5
   gap left. See `internal_docs/handoff_equation_tie_adr0068.md`. Would need an
   `equation_constraint` neutral group + schema bump (the same additive pattern
   used 3× this session) OR the deck-replay route.

2. **OpenSees deck-zone replay for reinforce / contact / embed** — the
   `/opensees/...` deck zone has no `reinforceTie` / `contactSurface` / contact /
   `LadrunoEmbeddedNode` record + `OpenSeesModel.from_h5` reconstruction +
   `OpenSeesModel.build` deck-replay. Documented as a follow-on in the h5.py
   emitter comments and `internal_docs/plan_rebar_p5.md` ("A4 full"). Low
   priority — the neutral round-trip already makes these features fully
   recoverable; this would only matter for a consumer that replays the
   `/opensees` deck zone directly instead of going through `FEMData.from_h5`.

3. **`s.mortar` / `s.tied_contact` stage-bound claim** — still deferred
   (`_DEFERRED.md`). `s.mortar`'s reason was updated this session: now that
   `g.constraints.mortar` resolves to a serial-only `ContactRecord` (not a
   claimable MP `SurfaceCouplingRecord`), there is no stage-claimable contact
   record. `s.tied_contact` is the SurfaceCouplingRecord-nesting double-emit
   issue.

## Env / conventions (unchanged, carry forward)

- Interpreter `C:\Users\nmb\venv\opensees_env\Scripts\python.exe` with
  `LADRUNO_OPENSEES_QUIET=1`. The fork build (contact + rigid body + embed) is
  at `C:\Users\nmb\Documents\Github\OpenSees` branch `ladruno`; point live runs
  at it with `APEGMSH_OPENSEES_BIN=C:/Users/nmb/Documents/Github/OpenSees/dist/bin`.
- Fork ground truth for contact: `SRC/interpreter/OpenSeesOutputCommands.cpp`
  (`OPS_LadrunoContact` ~380–859 — the option loop is the authority on valid
  token streams). Verify any new contact flag against it AND run it live.
- CI gates: static-gates (ruff `src/apeGmsh/opensees` HARD + mypy baseline 0)
  + suite + lock-tests. The hard ruff gate is opensees-only; `src/apeGmsh/core`
  carries 4 pre-existing `HEX8_TO_6_TETS`-style unused-import warnings (not
  CI-gated, untouched).
- Neutral-schema bump recipe (used 3× this session — reinforce/contact/embed
  are the templates): add a `*_payload_dtype()` in `mesh/_record_h5.py`; add
  `_write_*` + `_encode_*` + `_read_*` + `_decode_*` in `mesh/_femdata_h5_io.py`;
  wire the write/read entry points + `ElementComposite(...=)`; bump
  `NEUTRAL_SCHEMA_VERSION` + add a history comment; bump `tests/fixtures/schema.py`
  (`NEUTRAL_CURRENT` / `NEUTRAL_PRIOR_MINOR`). The two-version reader window
  auto-derives from `NEUTRAL_SCHEMA_VERSION` — no separate list to edit.

## Branch housekeeping (safe to prune)

- **Merged PR branches** were squash-merged with `--delete-branch`, so they're
  gone on the remote; clear stale local remote-tracking refs with
  `git fetch --prune`.
- **Stale UNMERGED remote branches** from the superseded stacked PRs — safe to
  delete on GitHub now that #722/#723 are closed and the work is landed:
  `feat/contact-generator` (#722 base), `feat/contact-extensions` (#723 head),
  `feat/g-embed-generator` (#721 base), `fix/contact-merge` (the old
  consolidation branch). `#703` (Linux viewer) is unrelated — leave it.
- Stale **local** branches with `[gone]` upstreams (`docs/concrete-guide-*`,
  `feat/ladruno-fork-backend`, the merged `feat/contact-*`) can be pruned with
  `git branch -d` once `git fetch --prune` has run.
