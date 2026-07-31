# ADR 0084 — Results viewer cascade freeze: one reconciler before catalog growth

**Status:** Accepted (2026-07-31) — both forks ratified by the user the
same day: **F-A(a)** id-coalesced RENDER-lane replay on batch exit and
**F-B(a)** qt CI lane now. Evidence base: a three-document external
assessment (`internal_docs/results_viewer_assessment.md`,
`results_viewer_adversarial_review.md`, `results_viewer_gui_perf_assessment.md`)
cross-checked by a four-agent adversarial verification sweep; the verified
findings and the PR ladder live in
[`internal_docs/plan_results_viewer_phase0.md`](../../../../../internal_docs/plan_results_viewer_phase0.md).
Finishes the adoption of ADR 0056 (viewer state & event contract) in the
results viewer; constrained by ADR 0058 (concurrent geometries).

## Context

The results viewer is the subsystem with the most fix–break churn: fixing
scrub breaks pins, fixing pins breaks export, hiding a contour makes the mesh
vanish. The verification sweep confirmed this is **structural, not
incidental** — the shell runs two reconcilers in series and half its safety
net is switched off:

1. **Dual STEP/RENDER.** `Director.set_step`/`set_stage` still run a direct
   pin-blind `registry.update_to_step` + `_render()`
   ([`_director.py:1083,1152,1202-1213`](../../../viewers/diagrams/_director.py)),
   then fire observers that run the pin-aware dispatcher pumps and render
   again. Every scrub tick pays ~2× STEP and 2× paint, with a pin-blind
   intermediate frame. In combined-stage mode the two paths disagree the
   *other* way: the direct pump pushes the translated local step, the
   dispatcher pump pushes raw global `step_index` last — potentially out of
   range ([`results_viewer.py:1467-1480`](../../../viewers/results_viewer.py)
   vs [`_director.py:628-629,1191-1208`](../../../viewers/diagrams/_director.py)).
   **Amended 2026-07-31 (PR 8 evidence):** the dual path is not merely
   redundant work plus flicker — it is a **crash path**. `set_step`
   pushes the raw global index to every attached diagram
   ([`_registry.py:400-409`](../../../viewers/diagrams/_registry.py) is
   pin-blind), so a geometry pinned to a *shorter* stage reads a step
   that does not exist and
   [`results/_time.py:68`](../../../../../results/_time.py) raises
   `IndexError: List time_slice contains out-of-range indices`. Stage
   pin + a short stage + scrubbing past its end is a live user-facing
   crash today, which raises D2's priority from performance to
   correctness.
2. **Batches drop the RENDER lane.** `Dispatcher.fire` under suppression
   enqueues UI-lane subscribers and returns before the RENDER-lane loop
   ([`_dispatch.py:421-436`](../../../viewers/diagrams/_dispatch.py)); batch
   exits replay primitives + UI lane only (the gap is acknowledged in-code at
   `_dispatch.py:713-716`). Ten RENDER-lane subscribers can be skipped.
   Session restore hand-patches one
   ([`results_viewer.py:3247-3256`](../../../viewers/results_viewer.py));
   the composition-eye batch ([`_outline_tree.py:574`](../../../viewers/ui/_outline_tree.py)),
   the reference-ghost batch (`_outline_tree.py:976`), and
   `duplicate_geometry` ([`_director.py:960`](../../../viewers/diagrams/_director.py))
   do not — producing the "hid the contour, mesh disappeared" ghost state via
   stale `fill_v = 0` (`results_viewer.py:1198`).
3. **Silent pumps.** Five `except Exception: pass/continue` sites inside the
   STEP/DEFORM/GATE pump bodies (`results_viewer.py:1493,1501,1529,1573,2774`)
   convert real failures into "the toggle did nothing."
4. **The guard stops at the shell's door.**
   [`test_viewer_state_contract.py:55-56`](../../../../../tests/viewers/test_viewer_state_contract.py)
   AST-guards `ui/`, `overlays/`, `mesh_viewer.py`, `model_viewer.py` —
   `results_viewer.py`, where the dual paths live, is unguarded.
5. **Session restore can shuffle layers.** `deserialize_session` compacts
   failed specs out of the list
   ([`_session.py:449-454`](../../../viewers/diagrams/_session.py)) while
   `CompositionSnapshot.layer_indices` are positional; `_apply_session`
   pads `None` for construction failures but downstream of the compaction,
   so a mid-list retired kind silently reassigns surviving layers to wrong
   compositions.
6. **The interaction net is out of CI.** The 15 `@pytest.mark.qt` tests are
   deselected by default addopts (`pyproject.toml:106`); the sole skip cause
   is pytest-qt not being installed. Auto-apply's flush fires N unbatched
   `DIAGRAM_MODIFIED` events → N renders
   ([`_diagram_settings_tab.py:672-682`](../../../viewers/ui/_diagram_settings_tab.py));
   its "dispatcher coalesces renders" comment is wrong for unbatched fires.

Two claims from the source assessment were **refuted** and shape the design:
removing the dual path does *not* blank export (both export surfaces bind
real pumps; only web/trame, bare-director scripts, and headless director
tests rely on the direct pump), and the cited `APEGMSH_VIEWER_STRICT` /
Gauss-escalation precedent does not exist in the tree — a bare re-raise
would vanish into the Qt event loop (`QTimer` callbacks, `safe_slot`).

## Decision

**D1 — Freeze before features.** No new diagram kinds or region tools
(threshold, isosurface, scope, interactive slice) land in the results viewer
until the freeze criteria hold: a render-count spy asserts exactly one
STEP + one DEFORM + one RENDER per scrub gesture, and the interaction tests
of D7 are green in default CI.

**D2 — One reconciler when pumps are bound.** `Dispatcher.bind()` records a
*pumps-bound* flag (existence checks against the always-present dispatcher
are forbidden — 0056 doctrine, restated at `_director.py:182-184`). When the
flag is set, Director's STEP/STAGE paths become intent + observer only: no
direct `update_to_step`, no direct `_render()`. The single surviving STEP
implementation is pin-aware and absorbs the combined-mode global→local
translation. When no pumps are bound (WebViewer/trame, bare-director export
scripts, headless director tests), the direct path remains and becomes the
*explicit* unbound-branch contract — the two Director contract tests that
encode today's dual behaviour (`test_results_director.py:263-271,301-324`)
are rewritten against that branch.

**D3 — Batches must leave no stale RENDER-lane state.** The hand-patch
pattern (each call site remembering which sync to re-run) is retired.
*Fork F-A — ratified (a):* batch exit replays RENDER-lane subscribers
whose events fired inside the batch, id-coalesced per handler — uniform,
kills all three known unpatched sites and future ones. (The rejected (b)
was a declared compensation registry per event kind.) The composition-eye
ghost-mesh repro becomes a permanent integration test.

**Known consequence of replay-once (recorded 2026-07-31, PR 9).** Two
RENDER-lane subscribers consume their event payload, so a batch in
which they fire for *several* subjects replays them once with the LAST
payload only: `_on_geometry_removed` (would leak the earlier
geometries' actors) and `_on_geometry_offset_changed` (would leave
stale pick KD-trees, self-healing on the next per-geometry event). No
call site batches multi-remove or multi-offset today, and option (b)
would not have fixed it either — the tension is inherent to coalescing
per handler. Coalescing per *payload* is not a free fix: the eye
cascade's N layers fire N distinct payloads, so it would restore
exactly the N× storm the batch exists to prevent. A real fix means
letting a subscription declare its coalescing mode, which is a widening
of `subscribe` and needs its own decision. **Do not add a
multi-geometry batching call site without resolving this first.**

**D4 — Pumps are loud.** The five silent pump sites route through
`log_error` + the [`_failures.py`](../../../viewers/_failures.py)
`register_error_handler` registry (the `_pump_restack` pattern,
`results_viewer.py:1600-1605`, is the in-file template). A pytest fixture
registers a collecting handler and fails at teardown — escalate-in-tests
without fighting the Qt event loop. Env-var-triggered bare re-raise is
rejected.

**D5 — The shell joins the guard.** `results_viewer.py` enters
`_GUARDED_FILES` with measured budgets (G-RENDER 9, G-ARTIFACT 15,
G-IMPORT 7) under the existing two-way ratchet. The `mesh_viewer.py`
precedent already ratifies guarding reconciler-containing files; allowlist
comments must distinguish reconciler-legitimate sites from burn-down debt.

**D6 — Session holes are padded, not re-keyed.** `deserialize_session`
appends a `None` sentinel for skipped specs so `layer_indices` stay aligned;
the two non-`_apply_session` consumers gain `is not None` guards. Stable
spec IDs are rejected: the restore path already distrusts persisted IDs
(`results_viewer.py:3166-3167`), `_apply_session` already chose
index-with-holes as its alignment contract, and an ID scheme would force a
schema bump plus dual-path loading for ten prior versions to fix what the
sentinel fixes in ~10 lines. No schema bump (stays v10).

**D7 — The interaction graph gets a CI-visible net, and the pumps get a
seam.** The gate and eye-cascade tests land headless now (real diagram
classes, not stubs; render/gate counters). The pump closures and the
session-apply path are extracted from `_show_impl` into a `PumpSet` with
frozen inputs — the front half of the assessment's "Phase 2 shell carve,"
**promoted into the freeze** because the scrub-values and session
round-trip tests cannot run deterministically headless without it. This is
a mechanical extraction under D4's loud pumps and D5's ratchet, not a
redesign. *Fork F-B — ratified (a):* the qt CI lane lands now
(pytest-qt + per-file fresh-process runs + xvfb; mesa stack already in
CI). (The rejected (b) was deferring it past the freeze.)

### Execution order

Dependency-ordered PR ladder (details, effort, and done-criteria in
[`plan_results_viewer_phase0.md`](../../../../../internal_docs/plan_results_viewer_phase0.md)):

| # | PR | Decision |
|---|---|---|
| 1 | Guard `results_viewer.py` (budgets 9/15/7) | D5 |
| 2 | Loud pumps + strict fixture | D4 |
| 3 | Session `None`-sentinel pad + alignment test | D6 |
| 4 | `gesture_batch` around auto-apply flush | D3 (independent slice) |
| 5 | Headless gate + eye-cascade tests | D7 |
| 6 | Qt CI lane (if F-B(a)) | D7 |
| 7 | `PumpSet` / session-apply extraction | D7 |
| 8 | Scrub-values + session round-trip tests | D7 |
| 9 | Batch-exit protocol (per F-A) + ghost-mesh test | D3 |
| 10 | Single STEP path + render-count spy | D2, closes D1 |

PRs 1–5 are independent and may proceed in parallel; 7 requires 1+2+5 as
its net; 9 requires 7's test surface; 10 lands last, after 9, because
session restore currently relies on the direct pump for mid-batch work
while fires are suppressed. PR 10's regression surface: web/trame
scrubbing, bare-director export, `Results.export_animation`, combined-mode
boundary crossing, `set_stage` end-of-history landing, session-restore
final state, and the scrubber perf gate (#877/#879 family), which must show
the ~2× win rather than a regression. PR 10 must also add the pinned-to-a-
shorter-stage scrub case as a regression test: it is the `IndexError`
above, and the single pin-aware STEP path is what fixes it.

## Alternatives considered

- **Catalog first (original assessment Tier A order).** Rejected: new
  feature nodes on a graph with two reconcilers and a dropped RENDER lane
  inherit every cascade bug and add restack/occlusion surface.
- **Full or shell rebuild.** Rejected for this pain: the verified defects
  are finite and localized (one dual path, one batch gap, five silent
  sites, one compaction bug); the domain layer (diagrams, IR, Results
  pairing) is the asset. A shell-only rewrite remains the documented
  fallback if the carve fails twice under the freeze tests.
- **Unguarded removal of the Director direct path.** Rejected: blanks
  web/trame and bare-director consumers. Hence the pumps-bound flag.
- **Stable spec IDs for sessions.** Rejected (see D6).
- **`APEGMSH_VIEWER_STRICT` bare re-raise.** Rejected: exceptions raised in
  `QTimer`/`safe_slot` contexts never reach the test; the `_failures`
  registry is the working precedent.
- **Making ActiveObjects the single owner.** Out of scope; 0056's
  focus-projection ruling stands. This ADR finishes 0056 adoption rather
  than amending it.

## Consequences

**Positive.** Scrub does one STEP/DEFORM/RENDER (structural ~2× frame-path
win, locked by a spy test and the bench gate); pin flicker and the
combined-mode wrong-index class die with the dual path; ghost-mesh and
layer-shuffle bugs get permanent regression tests; pump failures surface in
CI instead of as empty viewports; the god shell is inside the AST ratchet,
so multi-agent PRs cannot silently reintroduce direct renders; the `PumpSet`
seam is the tested foundation the later shell carve (B1/B2) builds on.

**Negative / accepted.** The unbound direct path survives as a second,
now-explicit branch (web parity work later may retire it); two Director
contract tests must be rewritten, and any test relying on double renders
will surface; the guard allowlist adds maintenance (budgets must ratchet
down as the shell shrinks); PR 10 carries real regression risk and is
deliberately last — the freeze's cost is that the most-wanted fix ships
after its net, not first.
