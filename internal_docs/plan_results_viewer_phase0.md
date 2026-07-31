# Results Viewer — Verified Phase 0 Plan (Cascade Freeze)

> **Date:** 2026-07-31
> **Supersedes for sequencing:** `results_viewer_adversarial_review.md` §5 (Phase 0),
> which itself superseded `results_viewer_assessment.md` §5 Tier A.
> **Method:** Four independent adversarial verification agents cross-checked every
> load-bearing claim in the Grok docs against source. This file records what
> survived, what was corrected, and the resulting PR-sized plan.
> **Freshness:** counts below measured 2026-07-31; re-verify before acting if stale.

---

## 1. Claim verification scoreboard

| Grok claim | Verdict | Correction / addition |
|---|---|---|
| Director dual STEP/RENDER on scrub | **CONFIRMED** | `_director.py:1083,1152,1202-1213` direct pump+render; dispatcher pumps repeat both |
| Batches skip RENDER-lane subscribers | **CONFIRMED** | `_dispatch.py:421-436` early return; exits replay primitives + UI lane only (comment at 713-716 admits it). 10 RENDER-lane subscribers enumerated, all in `results_viewer.py` |
| Composition-eye path unpatched (ghost empty mesh) | **CONFIRMED** | Repro traced end-to-end via `_outline_tree.py:574` → stale `fill_v=0` at `results_viewer.py:1198`. **Two more unpatched sites found:** reference-ghost batch (`_outline_tree.py:976`) and `duplicate_geometry` (`_director.py:960`) |
| Auto-apply flush fires N unbatched DIAGRAM_MODIFIED | **CONFIRMED** | 150 ms debounce is input coalescing only; flush runs every card's commit unbatched → N renders. In-code comment "dispatcher coalesces renders" is **wrong** for this path (`_diagram_settings_tab.py:678-679`) |
| Session deserialize index hazard (CRITICAL) | **CONFIRMED** | Schema is **v10 not v8**. `_apply_session` already pads with `None` — but downstream of the compaction in `deserialize_session` (`_session.py:449-454`), so alignment is lost upstream. Existing skip tests use payloads with no `geometries` block → gap untested |
| "Remove dual path → export/web/headless blank" | **PARTIAL (overbroad)** | Interactive export AND `Results.export_animation` bind real pumps → safe. Only web/trame, bare-director scripts, and ~25 headless director tests rely on the direct pump. F1's "when dispatcher bound" guard is sound — but no pumps-bound flag exists today |
| Pin-blind vs pin-aware STEP | **CONFIRMED** | Plus a **new bug Grok missed**: combined-stage mode disagrees in the *opposite* direction — direct pump pushes translated local step, dispatcher pump pushes raw global `step_index` last (potentially out of range). F1 must absorb combined translation |
| G-RENDER guard excludes `results_viewer.py` | **CONFIRMED** | Extension is mechanical; measured budgets: G-RENDER 9, G-ARTIFACT 15, G-IMPORT 7. `mesh_viewer.py` precedent already ratifies guarding reconciler-containing files |
| Silent pumps | **CONFIRMED, but smaller** | True pump-silence set is **5 sites** (1493, 1501, 1529, 1573, 2774), not ~110-in-file. `_pump_restack` already logs — it is the in-file template |
| `APEGMSH_VIEWER_STRICT` / Gauss escalation precedent | **DOES NOT EXIST** | Grok docs cite a precedent that isn't in the tree. Real mechanism: `viewers/_failures.py` `register_error_handler` registry (tests already capture-and-assert via it). Bare re-raise dies in Qt event loop (`QTimer` callbacks, `safe_slot`) — use collect-at-teardown fixture instead |
| 15 `@pytest.mark.qt` tests out of CI | **CONFIRMED** | 15 tests across 9 files; sole skip reason is pytest-qt not installed. CI already has mesa/EGL stack; lane = pytest-qt + per-file fresh process + xvfb (S–M) |
| ~650 `except Exception` | **STALE** | 694 across 107 files today |

**Bottom line:** the Grok diagnosis (cascade coupling, not missing features) survives
adversarial verification intact. Its plan needs four revisions:

1. **F0 is smaller and differently shaped** — 5 sites, `_failures` registry fixture, not an env-var re-raise.
2. **F1 is riskier than scoped** — needs a pumps-bound flag, combined-mode step translation, rewrite of 2 contract tests (`test_results_director.py:263-271, 301-324`), and it *depends on F2* (session restore currently relies on the direct pump mid-batch while fires are suppressed).
3. **F4 is trivial** — ~10-line `None`-sentinel in `deserialize_session` + one alignment test. Pad; do NOT switch to stable IDs (restore path already distrusts persisted IDs; `_apply_session` already chose index-with-holes as the contract).
4. **F5 splits** — tests 2 & 6 feasible headless today; tests 1 & 7 need pump/session-apply extraction from `_show_impl` first. That pulls the front half of B1′ (PumpSet extraction) INTO Phase 0 as an enabler.

---

## 2. PR-sized plan (dependency order)

| # | PR | Content | Effort | Risk | Done when |
|---|---|---|---|---|---|
| 1 | **Guard the god file** (F3) | Add `results_viewer.py` to `_GUARDED_FILES` with budgets 9/15/7 + honest per-entry comments (reconciler-legitimate vs burn-down debt) | S | None (test-only) | New direct `render`/`SetVisibility` in the shell fails CI |
| 2 | **Loud pumps** (F0) | 5 pump catch sites → `log_error` + `_failures.report`; pytest fixture registers collect-and-fail-at-teardown handler | S | Low (0–5 at-risk tests, all outside default CI) | A failing pump fails the suite, not the viewport |
| 3 | **Session index pad** (F4) | `deserialize_session` appends `None` instead of `continue`; guard 2 consumers (`_prompt_restore`, `session_restorable`); add mid-list retired-kind + two-compositions alignment test | S | Low | Partial restore keeps every surviving layer in its original composition |
| 4 | **Batch auto-apply flush** (F2a) | Wrap `_flush_auto_commits` in `gesture_batch`; fix the wrong "dispatcher coalesces" comment; only run commits for dirty cards if cheap | S | Low | Multi-card commit → 1 render (spy test) |
| 5 | **Interaction tests, feasible half** (F5a) | Test 2: composition gate with real diagram classes (upgrade `test_gate_effective_visibility.py` from stubs). Test 6: end-to-end eye cascade through outline delegate, assert gate==1 render==1 | S+S | Low | Both green headless in default CI |
| 6 | **Qt CI lane** (parallel, optional) | pytest-qt install + 9 per-file fresh-process invocations + xvfb | S–M | Runner GL flakiness | 15 qt tests run somewhere on every PR |
| 7 | **Pump extraction** (B1′ front half, promoted) | Extract `_pump_step`/`_pump_deform`/`_pump_gate` + session-apply from `_show_impl` into a testable `PumpSet` with frozen I/O (closures capture ~6 things: director, scene, render-geometries, deform reader, sync helpers). No behavior change; PR 1's guard + PR 2's loud pumps + PR 5's tests are the net | M | Medium | Pumps invocable headless without `show()`; guard budgets ratchet down |
| 8 | **Interaction tests, hard half** (F5b) | Test 1: contour + line_force + deform scrub asserting values+points via extracted pumps. Test 7: full session round-trip via extracted session-apply | M | Low (after 7) | Both green headless in default CI |
| 9 | **Batch-exit protocol** (F2) | Design decision: replay RENDER-lane (id-coalesced) on exit, or per-event compensation registry. Must cover composition-eye, reference-ghost, `duplicate_geometry`. Integration test: eye-hide occluding contour → substrate fill returns | M | Medium | Ghost-empty-mesh repro test green; no per-call-site hand patches remain |
| 10 | **Single STEP path** (F1) | Pumps-bound flag in `Dispatcher.bind()`; Director skips direct pump+render when flag set; single pin-aware STEP absorbs combined-mode global→local translation; rewrite `test_results_director.py:263-271, 301-324` as unbound-branch contract; render-count spy asserts exactly 1 render/scrub tick | M–L | **Highest** — lands last, under the full net | Scrub = 1 STEP + 1 DEFORM + 1 RENDER; pin flicker gone; web/trame + bare-director tests untouched; bench gate (#877/#879 family) shows the ~2× win |

**Regression surface for PR 10** (from verification): web/trame scrubbing, bare-director
export (`test_animation.py`), `Results.export_animation` end-to-end, combined-mode
boundary crossing, `set_stage` end-of-history landing, session restore final state,
render counts vs the perf bench gate.

## 3. Explicitly deferred (unchanged from Grok, now with evidence)

- Catalog tools (threshold/slice/isosurface/scope) — new nodes on the graph; wait for PR 10 green.
- GUI sprint (G0/G1) — except G0.1-style occlusion/gate feedback may ride PR 9 cheaply.
- Notebook-safe default + time envelope — Phase 1, after PR 10 (envelope touches the time path being unified).
- Full B1/B2 shell carve — PR 7 does only the slice Phase 0 needs.
- Any rebuild option — no evidence emerged that changes the "no rebuild" verdict.
