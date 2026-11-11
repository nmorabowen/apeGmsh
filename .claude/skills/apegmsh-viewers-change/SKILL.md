---
name: apegmsh-viewers-change
description: >
  Checklist for CHANGING apeGmsh's viewer code: src/apeGmsh/viewers/ (results
  session/panes/slots, mesh/model viewers, web/trame viewer,
  backends/, diagrams/, scene/, session/, overlays/, ui/), tests/viewers/ and
  viewer_bench/, and the Electron app's main process (apeGmshViewer/src/main/,
  apeGmshViewer/test/main/). Use before touching dispatcher/pumps/reconciler, a
  diagram kind, a dock/pane/window, key bindings, a VTK/pyvista call, a viewer
  test, or an apeGmshViewer IPC door. Every
  item points to the ADR, test docstring or archive entry that explains it.
  Not for USING the viewers from a model script (that is apegmsh-helper).
---

# Changing viewer code: checklist

Read this before changing anything under `src/apeGmsh/viewers/`, `tests/viewers/` or
`viewer_bench/`, and the apeGmshViewer section before changing `apeGmshViewer/src/main/`.
- Each item names a file and a phrase to grep for. Read that entry when the item applies.
- **[guard]** items are enforced by an AST test in `tests/viewers/`, run by CI's `suite` lane.
- Out of scope: using the viewers (`apegmsh-helper`), and proving a change looks right
  (`.claude/skills/apegmsh-viewers-visual-check/`).

## Before you start

- [ ] Read the ADR that owns the surface (index: `decisions/README.md` under
      `architecture/`): render seam 0042; pick 0045/0047; state and events 0056; concurrent
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

- [ ] Create the QApplication before any QWidget. This bit twice (b6af330a, then #246); its
      test went with the legacy `ResultsViewer` (X3-d, #1734), so read commit b6af330a.
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
      viewers. Session pane docks call it by design (`viewers/session/_host.py`, ADR 0098 A3.3).
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
      button. A dropped capability is a row in the #1199 parity tables (the parsing guard
      `test_session_capability_parity.py` retired with the legacy viewer, X3-d #1734).
- [ ] Added a new open or load door? Add it to `DOORS`; see
      `test_every_door_opens.py` › "A new door must be".
- [ ] Break each new test once: revert the fix and watch it fail. 03666479 is
      "mutation-proven".
- [ ] Viewer test traps (`caplog`, `importorskip`, `timeout=`, GL exhaustion, offscreen at
      import) are in `internal_docs/viewer_lessons.md`.

## apeGmshViewer doors (the Electron app, `apeGmshViewer/src/main/`)

The page is not trusted to name a path (owner ruling "option A", #2753). Read the
`main.ts` comments at `ungestured`, `windowSet`, `confined` and `fromCurrentPage` first.

- [ ] **A new IPC door that reads or opens a file is confined.** Compare the asked path with
      what main chose, by `samePath` (case-folded where the file system folds), and refuse
      before any read, so the set stays as it is. The doors today come in two kinds:
      - **read the path main holds**, never the asked string: `setFile` (the `results:*` and
        `geometry:open` doors; the window set's file, from `windowSet()`), `confined`
        (`user:read`, `user:readCase`; the set's `.user.h5`, and the model and results its
        staleness is judged against) and `setSources` (`source:goto`; the model's
        `/provenance` as main reads it now);
      - **approve the page's string, then read that string**: `model:open` and
        `viewer:requestOpen` through `ungestured` (the latest drop or a path main sent the
        page), `viewer:offerOpen` through `offerRows` (an offered set's files).

      Prefer the first kind for a new door. A path the model names is tested with
      `isRemotePath` before any `fs` call (`test/main/remoteRows.test.ts` wraps `node:fs` to
      prove it).
- [ ] **Gesture records live in main.** `dropped` is the latest path the preload's `gestured()`
      reported over the synchronous `viewer:dropped` (a File the page made has no path and
      reports nothing); the next drop replaces it, and an open does not spend it. `offerRows`
      mirrors the page's offer list (`PAGE_OFFERS` equals `state/reduce.ts` `MAX_OFFERS`, held
      by `setDoors.test.ts`), and a page load keeps only the rows of batches still waiting
      (`pageLoading`). Record a new gesture in main; never take it from the page.
- [ ] **Check the sender of a page message that arms main state.** Today `viewer:offersReady`
      is the one door that does: it counts only when `fromCurrentPage(e)`, that is, `e.sender`
      is the window's `webContents` and `e.processId` and `e.frameId` are the main frame
      `did-navigate` committed (`pageFrame`). A late message from a page being replaced
      re-arms nothing (VU15, #2776). `viewer:dropped` and `viewer:subscribe` take no such
      check; a new door that arms state uses `fromCurrentPage`.
- [ ] **A reload is a `did-start-navigation` too.** `did-start-loading` fires only when the page
      goes from idle to loading, so a reload while it still loads starts none. Per-page state
      in main is cleared in `pageLoading`, which runs on `did-start-loading` and on a
      `did-start-navigation` with `isMainFrame && !isSameDocument`. New per-page state is
      cleared there.
- [ ] **Error paths log with `shown()`, never `JSON.stringify` of a renderer value.** An IPC
      value can be a BigInt or a cycle; `JSON.stringify` throws, and the door rejects instead
      of answering `ok: false` with the reason. `shown` (bounded `util.inspect`, never throws)
      is in `main.ts` and `source.ts`. A file door resolves with the reason; it never rejects
      (VU11, #2750).
- [ ] **Test a door with the harness in `apeGmshViewer/test/main/`.** `electronStub.ts` stands in
      for `electron`. It is a strict proxy, so a new top-level Electron use fails by name until
      the stub has it; `ready()` makes a recording window, and `commitFrame`, `pageEvent` and
      `pageEvents` deliver its page events. A driver (`setDoorsDriver.ts`, `userDoorsDriver.ts`)
      loads `main.ts` in a child Node process, with `registerHooks` resolving `electron` to the
      stub, because main reads its mode from argv and watches its set; it prints one JSON line
      the test parses. `preload.test.ts` loads the preload against the same stub. Every path a
      refusal test refuses is a readable file, so the door would read it without the check
      (`setDoors.test.ts` header). Run `npm test` in `apeGmshViewer/`.
- [ ] **A door onto a second set reads it as a slot of its own.** VU10's compare doors
      (`variants:compare`, `compare:readStep`, `compare:close`) take a manifest row by index
      and label (`Study.row`, which resolves and confines it again at open time), never touch
      `windowSet`, `dropped` or `followModel`, read through their own `ResultsHost`, and are
      emptied in `pageLoading`. `test/main/variantDoors.test.ts` is the pattern.
- [ ] **The one write door is `workspace:save`** (VU9a, #2803; `main/workspace.ts`). It takes
      one edit with closed keys (`parseEdit`: a `path`, or any key the op does not name, is
      `bad-request` before any read) and writes only `windowSet()`'s `<stem>.workspace.h5`;
      `cannotWrite` refuses every mode but view. The owner's HG9 rules are code there: read
      the target whole and close it (a Node or h5wasm handle on the target blocks the replace
      on Windows, `EPERM`), map a blocked replace to `heldRefusal` (never a retry), and never
      move a held readable file aside (an unreadable one, a truncated HDF5 file included, is
      renamed aside as Python's writer does). Every edit carries `expect`, the `token` of the
      read it was made on (the bytes' sha256, or `absent`); main refuses `stale` when the
      window set's file is not those bytes, so a set switch or another writer never takes an
      edit made on the old read (#2890 review). A new write follows `saveWorkspace`'s order (partial created
      `"x"` through `reader/h5open.ts` `createH5`, verify, fsync, revision check, replace) and
      its tests: `test/main/workspaceSave.test.ts` kills a child mid-save
      (`workspaceKillDriver.ts`) and injects `EPERM` through `WorkspaceDisk`, because a real
      hold blocks only on Windows and CI's `app-tests` runs on Linux.
- [ ] **The second write door is `combinations:save`** (VU9b, #2693; `main/combinations.ts`):
      ADR 0130 A4 into the window set's `.user.h5`, which can be gigabytes. It copies the file
      to its own partial and appends there (`reader/h5open.ts` `appendH5` opens only a
      `.partial` for writing: `a` on the live file corrupts it when killed, the probe on
      #2693). Before the replace it closes the app's own `openH5` handles on the file
      (`closeHandlesOn`; one blocks the rename on Windows as `held`), and it checks free
      space before the copy. The save token is the path and `/meta@revision`, never a
      whole-file digest. `test/main/combinationsSave.test.ts` holds one test per HG9 rule, the
      kill at each step (`combinationsDriver.ts`) and a zone digest of every root group.
- [ ] **h5wasm 0.10.3 write traps.** A dtype's width comes from its letter, not its digits:
      `"<f8"` writes **f4** (use `"<d"`), `"<i4"` is i4 by luck of `i`. An empty array needs
      an explicit dtype (it guesses integer). `"x"` on an existing name returns a negative
      `file_id` instead of throwing (`createH5` throws). Prove a new layout with h5py reading
      it back (`apeGmshViewer/test/fixtures/vu9a/README.md` has the round trip).

## Before the PR

- [ ] Run the visual-check guide, and attach its reference and candidate stills to the PR.
- [ ] Add one `changelog.d/<slug>.md` fragment; never edit `CHANGELOG.md` (`internal_docs/changelog_workflow.md`).
- [ ] A bench finding is a claim about code, so re-check it against the code. See
      `viewer_bench/README.md` › "This entry was wrong for a while".

Found a new trap? Add it to `internal_docs/viewer_lessons.md` and one pointer line here. If
its shape can be detected mechanically, write an AST guard test in `tests/viewers/` instead.
