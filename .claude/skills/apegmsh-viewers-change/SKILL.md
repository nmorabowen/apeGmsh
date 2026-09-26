---
name: apegmsh-viewers-change
description: >
  Checklist for CHANGING apeGmsh's viewer code: src/apeGmsh/viewers/ (results
  session/panes/slots, ResultsViewer, mesh/model viewers, web/trame viewer,
  backends/, diagrams/, scene/, session/, overlays/, ui/), tests/viewers/ and
  viewer_bench/. Use before touching dispatcher/pumps/reconciler, a diagram kind,
  a dock/pane/window, key bindings, a VTK/pyvista call, or a viewer test. Every
  item points to the ADR, test docstring or archive entry that explains it.
  Not for USING the viewers from a model script (that is apegmsh-helper).
---

# Changing viewer code: checklist

Read this before changing anything under `src/apeGmsh/viewers/`, `tests/viewers/` or
`viewer_bench/`.
- Each item names a file and a phrase to grep for. Read that entry when the item applies.
- **[guard]** items are enforced by an AST test in `tests/viewers/`, run by CI's `suite` lane.
- Out of scope: using the viewers (`apegmsh-helper`), and proving a change looks right
  (`.claude/skills/apegmsh-viewers-visual-check/`).

## Before you start

- [ ] Read the ADR that owns the surface (index: `decisions/README.md` under
      `src/apeGmsh/opensees/architecture/`): render seam 0042; pick 0045/0047; state and events 0056; concurrent
      geometries 0058; cascade freeze 0084; design system to legend 0087–0090; stills 0094;
      results session 0098.
- [ ] The results viewer is where fixing one thing breaks another. Before touching pumps, the
      dispatcher, batches or session restore, read
      `internal_docs/results_viewer_adversarial_review.md` › "Fixing X breaks Y".
- [ ] pytest imports the worktree (`pythonpath = ["src"]`), but a script or `python -m` imports
      the editable main checkout. Set `PYTHONPATH=<worktree>/src`; `python -m apeGmsh doctor`
      checks for this.

## State, events, rendering (ADR 0056 / 0084)

- [ ] **[guard]** UI code does not render, flip actor flags or import a backend. It calls
      owner mutators and fires dispatcher events. Enforced by `test_viewer_state_contract.py`
      (G-RENDER / G-ARTIFACT / G-IMPORT). Budgets ratchet both ways; raise one only with an
      ADR 0056 reason.
- [ ] **[guard]** Nothing outside a diagram reads its `_actors`. It has been empty since
      ADR 0042 R-B, so the read is a silent no-op. Enforced by G-ACTORS in the same file; it
      bit in #593 and again in #620.
- [ ] Don't guard the always-present dispatcher with `getattr` or `is not None`. See ADR 0084
      › "existence checks against the always-present dispatcher".
- [ ] Pumps must fail loudly, through `_failures.py` (ADR 0084 › "Pumps are loud"). The
      autouse `pump_failures` fixture fails the test. `except Exception: pass` hides errors;
      see `viewer_bench/README.md` › "Side trap found on the way".
- [ ] Batching several geometries? First read ADR 0084 › "multi-geometry batching call site".
- [ ] A new diagram kind lives in `diagrams/` and registers (`test_diagram_kind_registry.py`),
      overrides `sync_substrate_points` and `set_visible` (`test_deform_follow_contract.py`;
      the base `set_visible` walks the dead `_actors`), and imports no vtk
      (`tests/test_diagrams_pure_no_pyvista.py`).
- [ ] **Fixed one instance of a pattern? Grep for its siblings before pushing.** Every viewer
      guard incident was a sibling site the first fix missed: #593→#620, #1122→#1123,
      #743→#782, #781→#878, #374→184b5734.

## Qt / VTK lifecycle

- [ ] Create the QApplication before any QWidget. This bit twice (b6af330a, then #246); see
      `test_results_viewer_qapplication.py` › "must ensure a QApplication".
- [ ] **[guard]** Every `QApplication(...)` call is preceded, in the same function, by
      `prepare_qt_environment()`. G-QAPP-ENV in `test_viewer_recurrence_guards.py`: #743 guarded
      three sites and #782 found the fourth.
- [ ] Whatever opens a GL context or interactor closes it on dispose or close. See ADR 0098
      › "Amendment 1 caution 1 in a new place". The teardown test must assert that the
      *product* closed it, not the fixture (06f82f9a).
- [ ] The pane host is never the central widget. See `viewer_bench/README.md`
      › "Fixed by ADR 0098 Amendment 3".
- [ ] Navigation docks (outline, browser) are construction-time; never `restoreDockWidget` them.
      See `test_dock_invariant.py` › "returned 3+ times", which scans only the mesh and model
      viewers. Session pane docks call it by design (`session/_host.py`, ADR 0098 A3.3).
- [ ] A key the window documents as global is a `QShortcut` with `ApplicationShortcut`, because
      `add_key_event` only fires while the viewport has focus. Tab needs an `eventFilter`.
      See `internal_docs/viewer_lessons.md` › "Key bindings in a VTK-hosted window". The digit
      keys are guarded by `test_dim_filter_keys.py`.
- [ ] **[guard]** `test_viewer_recurrence_guards.py` holds patterns a fix removed at one site
      while a sibling survived. G-VTK-REMOVED: no method a supported VTK removed (`vtk>=9.2`
      is unbounded; add a `REMOVED_VTK_API` row for the next one). G-GHOST-BIT: the hidden
      bit is 0x20, so import `HIDDENCELL`. G-HASH: no builtin `hash()`; use `zlib.crc32`.
- [ ] **[guard]** Style follows ADR 0087: no literal colours, no shouted labels, no dangling
      QSS. See `test_viewer_style_contract.py`.

## Tests

- [ ] Prefer headless tests. `RecordingBackend` asserts on the emitted layers;
      `frames_match_or_skip` compares pixels (never `array_equal`). Both are in
      `tests/viewers/conftest.py`.
- [ ] A test that opens a real window is `@pytest.mark.qt` and runs one file per process:
      `pytest -m qt tests/viewers/<file>`. A command-line `-m` replaces addopts, so repeat
      `and not qt`; see `.github/workflows/tests.yml` › "`not qt` must be repeated".
- [ ] Test the widget a person uses, not only the session IR (bb0f3388, "gates for the class
      of defect the bench keeps finding"): `test_inspector_picker_law.py` asserts on the Add
      button, and `test_session_capability_parity.py` must list any dropped capability.
- [ ] Added a new open or load door? Add it to `DOORS`; see
      `test_every_door_opens.py` › "A new door must be".
- [ ] Break each new test once: revert the fix and watch it fail. 03666479 is
      "mutation-proven".
- [ ] Viewer test traps (`caplog`, `importorskip`, `timeout=`, GL exhaustion, offscreen at
      import) are in `internal_docs/viewer_lessons.md`.

## Before the PR

- [ ] Run the visual-check guide, and attach its reference and candidate stills to the PR.
- [ ] Add one insert-only `CHANGELOG.md` section (`internal_docs/changelog_workflow.md`).
- [ ] A bench finding is a claim about code, so re-check it against the code. See
      `viewer_bench/README.md` › "This entry was wrong for a while".

Found a new trap? Add it to `internal_docs/viewer_lessons.md` and one pointer line here. If
its shape can be detected mechanically, write an AST guard test in `tests/viewers/` instead.
