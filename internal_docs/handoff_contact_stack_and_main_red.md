# Handoff — contact stack (#722/#723) + `main`-red cleanup

Written 2026-06-25. Two independent workstreams. **Do B first (quick, unblocks
CI on `main`), then A.**

## TL;DR

- **B — `main` is RED** from my own #734/#735: 12 real stale-assertion test
  failures (I added H5 dtype fields + bumped schema to 2.20.0 but didn't update
  the assertion tests; the PRs merged before CI caught it). + 2 known flakes.
  Quick fix on a fresh branch off `main`.
- **A — the contact stack** (#722 NTS/mortar contact + #723 flag extensions) is
  *merged onto main* on local branch `fix/contact-merge` (conflicts resolved,
  static-gates green, 8/8 fork-live tests pass incl. contact on the updated
  runtime) **but the adversarial review found blocking bugs** in the contact
  code. Fix bugs → write ADR → re-review → land. Do NOT merge as-is.

## Branch / PR / commit state

- `main` @ `e38cad1e` (#735). The constraints-coverage program (B3 equalDOF_Mixed
  / A1 EmbeddedNode pressure / B2 LadrunoRigidBody+omega; ADRs 0069–0071; neutral
  H5 schema **2.20.0**) is fully merged.
- Local branch **`fix/contact-merge`** (NOT pushed) — this is the resolved #722
  content + `main` + fixups. 4 commits on top of `main`:
  - `75fbc63f` g.embed generator → LadrunoEmbeddedNode (from closed #721)
  - `552fa911` g.constraints.contact generator → fork NTS/mortar (#722)
  - `14f0f8ea` Merge `main` into the stack (10 conflict files resolved)
  - `50dae7b7` post-merge fixups (see below)
- Open PRs: **#722** `feat/contact-generator → feat/g-embed-generator` (stale
  non-main base!), **#723** `feat/contact-extensions → feat/contact-generator`
  (stacked on #722). #720 merged; **#721 CLOSED unmerged** (its g.embed work
  lives inside #722). #703 (Linux viewer) is unrelated.
- **Fork runtime updated**: `opensees_env` now has `LadrunoContact` +
  `LadrunoRigidBody` (build `bab6995c`). `contactSurface`/`contact` run live.

## Workstream B — fix `main` red (my #734/#735 collateral)

Proven pre-existing on a clean `main` worktree. **12 to fix + 2 ignore:**

- `tests/test_record_h5_dtype.py` (6) — assert OLD field tuples. Update for the
  fields I added: `master_dofs` (node_pair), `as_element`/`mass`/`omega`
  (node_group), `cpl_pressure`/`cpl_kp` (the `_coupling_control_fields` mirror in
  interpolation + node_group). Tests: `test_node_pair_payload_fields`,
  `test_node_group_payload_fields`, `test_interpolation_payload_fields`,
  `test_surface_coupling_payload_fields`, `test_node_group_vlen_offsets_packed_flat`,
  `test_node_pair_record_roundtrips_through_h5`.
- `tests/mesh/test_rebar_element_h5_roundtrip.py` (2) +
  `tests/mesh/test_reinforce_tie_h5_roundtrip.py` (2) — hard-code schema `2.16.0`
  / "current neutral version" / "prior minor". Schema is now 2.20.0
  (`tests/fixtures/schema.py` already has `NEUTRAL_CURRENT=2.20.0`,
  `NEUTRAL_PRIOR_MINOR=2.19.0`). Fix the hard-coded `2.16.0` expectations + the
  "reads pre-2.16.0 within window" boundary.
- `tests/test_results_element_response_phase11b.py::TestCustomRuleCatalog::test_no_unexpected_entries`
  + `phase11c.py::TestFiberCatalog::test_size_two_tokens_per_class` (2) — results
  catalog drift. `git blame` to confirm whether mine or someone else's; update
  the catalog expectation.
- **IGNORE (known flakes):**
  `tests/opensees/unit/test_emitter_partition_open_close.py::{test_live_emitter_rank_0_passes_through,
  test_live_emitter_restores_real_ops_after_close}` — order-dependent live-ops
  flakes, documented in #730. Confirm on base before blaming any diff.
- `tests/viewers/test_results_dim_hide_render.py` (2) — offscreen-Qt pixel tests;
  pre-existing locally. Verify against CI (`QT_QPA_PLATFORM=offscreen`) — likely
  env, not a code bug; leave unless CI is red on them.

**Do:** branch off `main`, update the stale assertions, run the suite, PR,
squash-merge. `main` green.

## Workstream A — contact stack bugs (adversarial review)

Three skeptical reviewers (arg-grammar / resolver-geometry / integration) on
`git diff main` of `fix/contact-merge`. Mergeability blockers, by severity.
Most are pre-existing in the contact code (#722/#723), not the merge.

| # | Sev | Bug | Fix sketch |
|---|---|---|---|
| 1 | 🔴 Crit | Contacts & embed ties **silently dropped under partitioned (OpenSeesMP) emit** — `apesees.py::_emit_partitioned` never calls `emit_contacts`/`emit_embed_ties` and has NO fail-loud guard (reinforce/rebar do, ~`apesees.py:2101-2129`). | Add `contacts`/`embed_ties` to that fail-loud guard block. |
| 2 | 🔴 Hard-fail | **NTS bare-`kn` + `-outward`** emits an unparseable token stream; fork reads 3 doubles, hits `-outward`, aborts `contact`. `contact.py:97-133`. | Always emit the full `kn kt mu` triple (pad `kt=mu=0.0`) whenever `outward` present (mirror the friction path). |
| 3 | 🟠 High | **Higher-order faces emit wrong `nps`** (tri6→6, quad8→8 vs 3/4) → fork misreads facets. `_parts_registry.py:1641-1649` → `contact.py:57`. No drop-to-corners (embed path does it at `EmbedmentsComposite.py:248`). | Drop to corner nodes; enforce `nps ∈ {3,4}`. |
| 4 | 🟠 High | **Auto `-outward`** = single param-midpoint normal + whole-surface centroid sign-flip → wrong/invalid on curved/closed masters. `ConstraintsComposite.py:422-460`. | Design call: per-facet outward, or fail-loud on non-planar/closed masters, or require explicit `outward=`. |
| 5 | 🟠 High | Contact-handler `return` short-circuits the **equation-tie handler upgrade** → contact + `enforce="equation"` tie silently drops EQ_Constraint. `apesees.py:4803-4824` before `:4846`. | Check `_fem_has_equation_ties` before the contact `return`; fail-loud or correct-warn. |
| 6 | 🟡 Med | **No range validation**: `kn<0`, `mu<0`, `eps_n<0`, `ngp<=0`, zero `outward`, embed `k<=0` all reach the emitter. `defs/constraints.py:939-977` (contact), `:837-878` (embed). | Add `__post_init__` range checks (mirror ReinforceDef). |
| 7 | 🟡 Med | Handler forcing overrides user `Transformation`/`Lagrange`; LadrunoContact is Plain-style for MP (fork P1a) → MP silently inert with contact. `apesees.py:4803-4814`. | Design: fail-loud on contact+MP instead of warn-and-emit-broken. |
| 8 | 🟡 Med | resolve-before-orphan-filter → dangling tie tags. `_fem_factory.py:516-525` vs `:578`. | Reconcile tie node sets against post-filter `node_ids`. |
| 9 | 🟡 Med | Embed corner-only geometry + nearest-centroid prefilter → mislocation on curved hosts. `EmbedmentsComposite.py:244-252`, `_inverse_map.py:335-353`. | Design: verify host straightness / widen candidate set. |
| 10 | 🟡 Med | Solid-part master → silent `None` outward (no dim-2 entity). `ConstraintsComposite.py:433-448`. | Derive outward from the boundary face actually used. |
| 11 | ⚪ Low | H5 `contact_surface` doesn't consume the latched MP-comment name → mislabels next MP. `h5.py:1382-1411`. | Add `self._consume_pending_mp_name()`. |
| 12 | ⚪ Low | `H5ReinforceDeviationWarning` reused for embed/contact (category error). `h5.py:1375-1411`. | Rename to `H5FeatureDeferredWarning` or split classes. |
| 13 | ⚪ Low | **Duplicate `LadrunoEmbeddedNode`** in `_FORK_ONLY_ELEMENTS` (merge dup). `live.py:74` & `:77`. | Remove the dup at `:74`. |
| 14 | ⚪ Low | Mixed tri/quad on one surface hard-fails. `_parts_registry.py:1644-1648`. | Coverage gap; document or support mixed stride. |

Plus the **1 contact-merge regression**: `tests/test_import_dag_polarity.py::test_eager_cross_package_edges_frozen` — the new `g.embed`/contact cross-package imports break the frozen import DAG. Update the frozen edge set (or refactor the import) — verify the new edges are legitimate first.

Disproven (don't chase): no contact *duplication* under partitioning (it's
omission); snapshot-hash exclusion correct; no MP double-counting; emit
order/tag-allocation clean; Protocol uniform across emitters.

## How the merge was resolved (trust/extend)

- Conflict files: `base.py` Protocol + `tcl/py/live/recording.py` resolved by
  HAND (union of `equationConstraint` [main] + `embedded_node`/`contact_surface`/
  `contact` [branch]; order matters). `build.py`/`FEMData.py`/`_fem_factory.py`/
  `apesees.py`/`_core.py` via additive keep-both (every conflict was
  `embed/contacts` [branch] vs `rebar_elements`/equation-tie [main]).
- `h5.py` auto-merged but dropped `H5ReinforceDeviationWarning`'s class def →
  fixed in `50dae7b7`.
- `50dae7b7` also: gated `test_contact_live` to skip when the build lacks
  `contactSurface` (autouse fixture probing `live._get_ops()`); narrowed the
  `-omega` emit for mypy.

## Dev env / conventions

- Interpreter `C:\Users\nmb\venv\opensees_env\Scripts\python.exe` with
  `LADRUNO_OPENSEES_QUIET=1`. Now has contact + rigid body (fork `bab6995c`).
- CI (`.github/workflows/tests.yml`): **static-gates** = `ruff check
  src/apeGmsh/opensees` (hard) + `mypy src/apeGmsh/opensees` (baseline **0**);
  **suite** = `pytest tests -m "not live and not subprocess and not bench"` (no
  openseespy installed → fork tests skip); **lock-tests**. Local `mypy` shows ~16
  openseespy-dep artifacts identical on `main` — CI's clean env doesn't see them;
  **don't chase them**.
- Emitter Protocol (`emitter/base.py`) is FROZEN — widening it is an architecture
  event (all 6 emitters + ADR note). Schema bumps additive, two-version reader
  window (ADR 0023).
- Fork parser ground truth: `C:\Users\nmb\Documents\Github\OpenSees` branch
  `ladruno` — `SRC/interpreter/OpenSeesOutputCommands.cpp` (`OPS_LadrunoContactSurface`
  ~322, `OPS_LadrunoContact` ~380, `OPS_LadrunoContactPlane` ~818),
  `SRC/element/ladrunoEmbeddedNode/OPS_LadrunoEmbeddedNode.cpp`.

## Recommended plan

1. **B** (quick): new branch off `main`, fix the 12 stale assertions, PR, merge.
2. **A**: on `fix/contact-merge` rebased onto the now-green `main`, fix the
   contained bugs (#1, #2, #3, #6, #11, #13, import_dag), decide the design-level
   ones (#4, #7, #9, #10) with the user, write contact ADR 0072, re-run the
   adversarial review, then land #722 (retarget base → `main`) and rebase/land
   #723. Replace the `g.constraints.mortar()` NotImplementedError if #722 didn't.
