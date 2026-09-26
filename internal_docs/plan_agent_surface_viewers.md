# Plan: an agent surface for the viewers (task guides + recurrence guards)

Revision 2 — adversarial review (Opus): "ship-with-changes", applied
2026-09-25. Dispositions are in "Revision 2 — review dispositions" below.
Revision 1 was the first draft.

**Status.** Built on branch `claude/agent-surface-viewers`, cut from `main` @
`299eb071` on 2026-09-25. Draft PR. **Merge only after the fix for
`ui/loads_tab.py::pattern_color` lands** (see "Live incidents — merge
order"). Until then the `suite` job is red on `test_g_hash`, on purpose.

**Scope.** Only the viewer subsystem: `src/apeGmsh/viewers/`,
`tests/viewers/`, `viewer_bench/`, the viewer ADRs, and the viewer
internal docs. The rest of apeGmsh is out of scope by the owner's
decision. A concurrent apeGmsh-wide port owns `AGENTS.md` and `CLAUDE.md`
(see "Coordination").

**Method.** The agent-surface playbook (`~/.claude/skills/agent-surface`),
piloted as the Ladruno OpenSees fork's WP-115. The method ports; the
rules below all come from this repo's own incidents.

## Problem

The viewers are the subsystem where fixing one thing breaks another. See
`results_viewer_adversarial_review.md`, ADR 0084, and 128 fix-worded
commits among 445 viewer commits.

The lessons are not lost. There are:
- the ADRs' Context and Rejected sections;
- the adversarial review's scoreboard;
- the viewer_bench "Findings so far";
- CHANGELOG `FIXED` sections;
- guard-test docstrings;
- 47 memory files.

What fails is putting the right lesson in front of the agent at the
moment it acts. The sharpest version of the failure: **a fix repairs the
site that was reported, the lesson gets written down, and the same
pattern survives at a sibling site until it bites on its own.**

### Phase 0 baseline: lessons that recurred (measured 2026-09-25)

| # | Lesson | Occurrences | Evidence | Now enforced by |
|---|---|---|---|---|
| 1 | A diagram's `_actors` is dead after ADR 0042 R-B | #593 674cdceb (06-10), then #620 a345918d (06-11) | git | **G-ACTORS** (new) |
| 2 | VTK 9.7 removed `AddActor2D` / `RemoveActor2D` | #1122 3b820e6e, then #1123 091e5a05 (both 09-07) | git | **G-VTK-REMOVED** (new) |
| 3 | The Qt env guard must run before *every* `QApplication(...)` | #743 bf3c62c6 (06-25), then #782 dc4344e1 (07-07): "despite the guard existing" | git | **G-QAPP-ENV** (new) |
| 4 | The hidden-cell ghost bit is 0x20, not 0x01 | #781 5e23f5fd (07-07), then #878 45db15df (07-28): "two halves disagreed" | git | **G-GHOST-BIT** (new) |
| 5 | Builtin `hash()` gives colours that change per process | #374 3b5cd921 (05-27), then 184b5734 (05-27); a third site is still live | git | **G-HASH** (new, red) |
| 6 | Create the QApplication before any QWidget | b6af330a (04-08), then #246 eb7764fa (05-18) | git | `test_results_viewer_qapplication.py`; guide |
| 7 | Keys don't reach bindings in a VTK-hosted window | Ctrl+Shift+L (v3.x, undated), then Tab (#479 review, 05-31), then digits 0–4 (44f8dd50, 08-13, "same law as ResultsViewer Esc") | memory + git | `test_dim_filter_keys.py`; guide; lessons |
| 8 | Outline dock stuck on the upper edge | 50884941 #235, d70defaa #383, 1da0fc33 #390, then d6e7aabc #484 | git; `test_dock_invariant.py` › "returned 3+ times" | `test_dock_invariant.py` |
| 9 | A pane or interactor outlives its close (orphaned GL context) | 556adef1 #886 (08-03), then 06f82f9a #1022 (08-17): "caution 1's orphaned GL context, moved rather than removed" | git | `test_pane_dispose_releases_context.py`; guide |
| 10 | Diagrams not following deformation | #65, #76, #84, #620 ("again"), #881, #1017 | git (survey) | `test_deform_follow_contract.py` |
| 11 | One artifact, N doors, bug shipped per door | #440, #441, #442 | `test_every_door_opens.py` › "the same bug shipped three times" | `test_every_door_opens.py` |
| 12 | Session IR tested, but the widget a person reaches it through is not | bb0f3388 (08-20): "every viewer defect found this week sat at the same seam" | git | parity + picker-law gates; guide |
| 13 | Swallowed exceptions hide defects | #678, #880, #881, #1122, #1136 | git (survey) | `_failures` + `pump_failures` (pumps only); guide |
| 14 | "The viewers can't be verified visually here" | corrected 05-30 and 06-10; restated in #495, #592, #593 | memory | visual-check guide |

**Baseline: 14 recurring lessons.** 11 are strict: a later fix cites the
earlier lesson, or says "again" / "recurred". 3 are inferred: #6, #12
and #13. Seven of the 14 have a shape that can be detected mechanically,
listed with a documented pre-fix and fix commit: #1–#5, #7 and #8. Two of
those (#7 and #8) were already guarded.

**Gate: pass.** An archive exists and recurrences are counted. Phases 2
and 3 are earned.

## Shape

1. **Task guides**, two of them. They follow the kinds of work in the
   history: 116 viewer commits since July. Most are ADR 0098 session work
   (38) and results-viewer features and UX (32); the rest are dispatcher
   hardening (11), stills and studio (14), standalone fixes (12) and
   test infrastructure (9).
   - `.claude/skills/apegmsh-viewers-change/SKILL.md`: before changing
     viewer code.
   - `.claude/skills/apegmsh-viewers-visual-check/SKILL.md`: before
     claiming a viewer change works. This is the playbook's "a UI →
     visual verification is its own guide", adapted to this repo:
     headless stills from `python -m apeGmsh.viewers render`, the
     conftest pixel band, a live window launched as a subprocess and
     stopped by PID, and the pane-host screenshot template.
   - *Accept:* the guides are 110 and 111 lines, with 11 lines of
     front-matter each, against the playbook's "about 100". Every item
     points into an archive: 20 file › "phrase" pointers. On 2026-09-25 a
     one-off script checked that each phrase greps on a single line of
     its target. Each guide says the end-user skill
     `apegmsh-helper` is out of scope.
2. **Memory lessons moved into the repo**, in `internal_docs/viewer_lessons.md`.
   These are the viewer traps that lived only in agent memory:
   - key bindings and focus;
   - `importorskip` of a parent package;
   - subprocess `timeout=`;
   - the logger that does not propagate, so `caplog` is blind;
   - GL context exhaustion under broad `-k`;
   - offscreen-at-import poisoning;
   - the "can't verify visually" belief.

   The memory files themselves are unchanged. General test-pollution
   lessons (the `sys.modules` purge, the cp1252 cascade) are not
   viewer-specific and are left to the apeGmsh-wide port.
   *Accept:* 7 lessons moved; the guides point at their headings.
3. **Recurrence guards**, as AST guard tests: the repo's own mechanism
   (ADR 0056 Part 5, ADR 0087 D3).
   - **G-ACTORS** joins `tests/viewers/test_viewer_state_contract.py`.
     Its incident is ADR 0056's own context item 1. Unlike G-RENDER /
     G-ARTIFACT / G-IMPORT, it scans all of `viewers/**` with a hard zero.
     Its `self` exemption hides the base `Diagram.set_visible` walk over
     `self._actors`. That hole is closed by an override-completeness check
     in `test_deform_follow_contract.py`: every kind must override
     `set_visible`. See the Revision 2 dispositions.
   - **G-VTK-REMOVED, G-QAPP-ENV, G-GHOST-BIT, G-HASH** go in the new
     `tests/viewers/test_viewer_recurrence_guards.py`. They are pure
     `ast`, over all of `viewers/**`, with a hard zero and no allowlist.
   - Each collector takes the root as an argument, so the plan's
     mutation acceptance runs the same code on `git archive` trees.
   - The tests run in the existing CI `suite` lane
     (`pytest tests -m "not live and not subprocess and not bench and
     not qt"`). No workflow change.
   - *Accept:* each guard flags its incident on the real pre-fix tree and
     passes on the fix commit (table below). The self-tests cover every
     incident shape and every sanctioned alternative shape found.

## Rejected approaches

- **A standalone `scripts/check_viewer_quirks.py` plus a new last step in
  `static-gates`.**
  - This repo already enforces mechanical viewer lessons as AST guard
    tests under `tests/viewers/`. A second mechanism would split where
    the rules live.
  - Those tests already run in `suite`.
  - Touching no workflow also means no collision with the apeGmsh-wide
    session's CI edits.
- **Putting the structural guards in the incidents' behavioural test
  files** (`test_scalar_bar_title_warning.py`, `test_qt_env.py`,
  `test_element_hide_pixels.py`). Those need pyvista, Qt or VTK at
  runtime and skip without them. A structural guard must run in every
  lane.
- **G-ACTORS in `test_deform_follow_contract.py`.** That guard is
  runtime introspection of `Diagram` subclasses. `_actors` is an AST
  rule, and its first incident is ADR 0056's context item 1.
- **Lint `render_points_as_spheres=True`** (#228 45f4163e, then #781
  5e23f5fd; both were "dots invisible on software GL"). HEAD has 5 more
  sites:
  - `mesh_viewer.py:1484`
  - `overlays/measure_overlay.py:160`
  - `session/_realize.py:1384`, `:1506`, `:1562`

  Whether they fail depends on the GL stack, and that cannot be checked
  on this machine. A red rule whose live sites are unproven would be a
  guess. Owner question.
- **A ratchet on broad `except Exception: pass`.** The incidents are real
  (#678, #880, #1122, #1136), but HEAD has 441 such handlers in
  `viewers/`. A per-file budget is its own work package, and ADR 0084 D4's
  runtime fixture already covers the pump sites.
- **Spin boxes that commit on every keystroke** (b7271110 #884, which
  said the fix was "never applied here"). A survey counted 17 candidate
  sites at HEAD. Whether each is a defect is not verified. Owner question.
- **Ban every `plotter.add_key_event`.**
  - The only key incident with a pre-fix and a fix commit (44f8dd50) is
    already guarded by `test_dim_filter_keys.py`.
  - Tab never reached main; review caught it on #479.
  - The remaining letter bindings may be viewport-only on purpose.
- **Lessons into `AGENTS.md` / `CLAUDE.md`.** The concurrent apeGmsh-wide
  port owns those files, so the lessons go to `internal_docs/viewer_lessons.md`.
- **One combined guide.** Changing viewer code and verifying it visually
  are different moments with different readers. The playbook makes visual
  verification its own guide.
- **A pointer-resolution lint for the guides** (Ladruno's L3). There is no
  incident behind it here. The pointers were checked once by a script.
- **Guarding `sections/`** (2 `QApplication` sites, both compliant). Out
  of scope by the owner's decision.
- **Closing the G-ACTORS `self` hole by guarding `diagrams/_base.py`
  directly** (revision 2). The base owns `_actors`, and its `detach` walk
  is legitimate teardown. The real fix, deleting the dead walk from
  `Diagram.set_visible`, is a production-code change, which this work
  package does not make. What turns the dead walk into a bug is a kind
  that inherits it, so the rule is an override-completeness check
  instead. Its mutant (a kind's `set_visible` override deleted at
  runtime) is killed.
- **Lambdas and class bodies as their own G-QAPP-ENV scopes**
  (revision 2). That would flag `class A: app = QApplication([])` inside
  a function that has already called the guard, although a class body
  runs in place, after the guard. Instead, the walk descends into both
  (it used to skip them) and counts them as part of the enclosing
  scope. Both variants are mutants, and both are killed.

## Results

### Mutation acceptance on the real commits

Each tree is `git archive <sha> src/apeGmsh/viewers`, with the guard's own
collector run over it. Sites are listed as `file:line`.

| Guard | Pre-fix tree | Hits | Fix tree | Hits | HEAD |
|---|---|---|---|---|---|
| G-ACTORS | 674cdceb^ | 2 (`results_viewer.py:871`, `:1002`) | 674cdceb (#593) | **1** (`:871`, the #620 site, a day early) | 0 |
| | a345918d^ | 1 (`:878`) | a345918d (#620) | 0 | |
| G-VTK-REMOVED | 3b820e6e^ | 4 (`pyvista_qt.py:864`, `:904`; `_pyvista_pick.py:238`, `:376`) | 3b820e6e (#1122) | **2** (the #1123 sites) | 0 |
| | 091e5a05^ | 2 | 091e5a05 (#1123) | 0 | |
| G-QAPP-ENV | bf3c62c6^ | 4 (`viewer_window.py:198`, `preferences_dialog.py:300`, `theme_editor_dialog.py:486`, `results_viewer.py:72`) | bf3c62c6 (#743) | **1** (`results_viewer.py:72`, the #782 site) | 0 |
| | dc4344e1^ | 1 | dc4344e1 (#782) | 0 | |
| G-GHOST-BIT | 5e23f5fd^ | 3 (`pyvista_qt.py:81`, `element_visibility.py:54`, `results_pick.py:386`) | 5e23f5fd (#781) | **2** (the #878 sites, three weeks early) | 0 |
| | 45db15df^ | 2 | 45db15df (#878) | 0 | |
| G-HASH | 3b5cd921^ | 3 (`color_mode_controller.py:172`, `:258`; `loads_tab.py:58`) | 3b5cd921 (#374) | 2 (the 184b5734 site; `loads_tab.py:58`) | **1** |
| | 184b5734^ | 2 | 184b5734 | 1 (`loads_tab.py:58`, never swept) | |
| | HEAD with `loads_tab.py` patched to `zlib.crc32` (scratch copy, not committed) | | | 0 | |

- The actual test functions were also run with `VIEWERS_DIR` pointed at
  each tree. They FAILED on every pre-fix tree and every intermediate
  tree. They PASSED on the final fix tree and on HEAD, except G-HASH,
  which passes only on the patched copy.
- The existing `test_dim_filter_keys.py` collector was run on 44f8dd50^
  and flagged all three viewers. On 44f8dd50 it reported 0 (see its hole
  under "Open questions").

**Every bold cell is a sibling the rule would have caught before it bit.**

### Self-tests, runtime, gates

- **Cases (revision 2).**
  - G-ACTORS: 6 flagged and 5 sanctioned shapes.
  - Recurrence guards: 27 flagged and 26 sanctioned. One of the
    sanctioned cases is a pinned known hole.
  - Both files carry a scope-sanity test.
  - The state-contract file also carries a ratchet self-test.
- **Runtime.** Scanning all 203 `viewers/**` files takes 3.5–6.7 s for
  the four recurrence guards and 1.6–3.7 s for G-ACTORS. That is the
  range over two runs on a loaded dev box; about 1.6 s of it is parsing.
- **ruff 0.15.9** (the CI pin) is clean on all four touched test files.
  mypy is not applicable: CI runs mypy only on `src/apeGmsh/opensees`.
- **pytest** (revision 1, as first published; the revision 2 rerun is in
  its own section below): `test_viewer_recurrence_guards.py` +
  `test_viewer_state_contract.py` gave 14 passed and 1 failed. The
  failure is `test_g_hash`, the live incident. Revision 1's "17 passed"
  had counted `test_dim_filter_keys.py`'s 3 tests without saying so.
  `tests/test_changelog_structure.py`: 4 passed.
  `scripts/sync_skill.py --check`: in sync.

### The key-binding contradiction: resolved

- **Memory** (`feedback_vtk_keyboard_shortcuts.md`, 2026-05-31) said
  digits are fine for `add_key_event`.
- **`test_dim_filter_keys.py`** and the 2026-08-13 CHANGELOG said VTK
  swallows them.

A probe (pyvistaqt 0.11.4, VTK 9.5.2, Windows) showed:
- `add_key_event("1")` fires while the viewport has focus;
- it does **not** fire with focus on a dock button or a `QTreeWidget`;
- an `ApplicationShortcut` fires in all three cases.

So the 08-13 incident was real, but its mechanism was focus, not
swallowing. The code and the test are authoritative. Revision 2 corrected
the test docstring, which had named the wrong mechanism. The finding is
recorded in `viewer_lessons.md` › "Key bindings in a VTK-hosted window",
and the guide points there.

### The visual-check procedure: run once for real (2026-09-25)

The guide's procedure was run once end to end, with the worktree on
`PYTHONPATH`:
- `Results.demo(path=dir)` wrote a demo pair.
- `python -m apeGmsh.viewers render demo_results.h5 contour.png --json`
  printed `{"ok": true, "written": ["contour.png"]}`, and the PNG was
  opened and inspected.
- `python -m apeGmsh.viewers demo_results.h5 --restore-session no
  --no-save-session` ran as a `Popen`. It was alive after 20 s, then
  `terminate()` took it down (exit 1). `tasklist` confirmed the PID was
  gone, and no `.viewer-session.json` was written.

## Live incidents — merge order

| Site | Rule | Why it is a real bug |
|---|---|---|
| `src/apeGmsh/viewers/ui/loads_tab.py:58` (`pattern_color`) | G-HASH | See below |

**Why `pattern_color` is a real bug.**
- It returns `_PATTERN_PALETTE[abs(hash(name)) % 7]`, and builtin
  `hash()` of a `str` is randomized per process.
- Measured: `pattern_color("dead")` gave `#94e2d5`, `#f38ba8` and
  `#a6e3a1` in three processes. Its docstring says "Stable color".
- It colours the mesh viewer's load arrows (`mesh_viewer.py:1315`) and
  the loads tab (`loads_tab.py:229`). Screenshots of the same model
  therefore differ between sessions, which also defeats reference vs
  candidate still comparison.
- It is the third site of the #374 / 184b5734 class, the one that was
  never swept.

**Merge order.**
1. A fix PR that switches this site to `zlib.crc32`, as
   `color_mode_controller.py` did.
2. Merge `main` into this branch.
3. Merge this PR.

Per the work-package rules, the site is neither fixed nor waived here.

## Coordination

The apeGmsh-wide agent-surface port is **PR #1171** (branch
`claude/agent-surface`). Revision 1 of this doc named a stale branch,
`guppi/agent-surface-port-c45d85`. #1171 owns `AGENTS.md` and
`CLAUDE.md`. This PR creates neither, does not touch
`.claude/skills/apegmsh-helper/`, and does not touch #1171's branch or
files. `.gitignore` already tracks `.claude/skills/`; that was verified
and left unchanged.

**Routing: one row, #1171's.** #1171's `AGENTS.md` already routes
`viewers/` to its own guide:

> `| Changing viewers/ or results/: … | .claude/skills/apegmsh-viewer-results/SKILL.md |`

That stays the single "read first" entry for the viewers. There is **no
second row**. Instead, #1171's `apegmsh-viewer-results` guide links on to
this PR's two guides:
- `apegmsh-viewers-change`, before changing viewer code;
- `apegmsh-viewers-visual-check`, before claiming a viewer change works.

**The key-binding fix #1171 needs.** On #1171's branch,
`.claude/skills/apegmsh-viewer-results/SKILL.md:59-60` says a shortcut
without `Qt.ApplicationShortcut` loses the key because "the interactor
eats the key". The mechanism is focus, not eating, as measured in
"The key-binding contradiction: resolved" above:
- `add_key_event` (and any viewport-level binding) fires only while the
  viewport has keyboard focus;
- an `ApplicationShortcut` fires wherever focus is.

Suggested wording: *"Keys the window documents as global use a
`QShortcut` with `Qt.ApplicationShortcut`: `plotter.add_key_event` fires
only while the viewport has focus (see
`internal_docs/viewer_lessons.md`, 'Key bindings in a VTK-hosted
window')."*

The coordinator has sent that session this note. This PR does not edit
#1171.

**Handed off to #1171:** `src/apeGmsh/results/capture/spec.py:1175`
(`_stable_section_tag`). It is documented as "Deterministic" but uses
`abs(hash(name))`: the G-HASH class, outside the viewer scope.

If #1171 adds a repo-wide lint, the viewer guards stay where they are, as
AST tests in `tests/viewers/`. They are not a second copy of the lint.

## Found along the way (not triaged, not fixed)

- **`results/capture/spec.py:1175`.** `_stable_section_tag` is documented
  as "Deterministic", but it uses `abs(hash(name))`. That is the G-HASH
  class, outside the viewer scope. Handed off to #1171 (see
  Coordination).
- **`scratchpad_shots/render_scalar_bar_sheets.py:117` and `:163`** still
  call `RemoveActor2D` / `AddActor2D`. It is a scratch script, outside
  `viewers/`, so G-VTK-REMOVED does not scan it. On VTK 9.7 it would
  raise. Out of scope, note only.
- **`APEGMSH_EXPECT_GL` is honoured by only three test files**
  (`test_render.py`, `test_render_live.py`, `test_viewers_main.py`).
  Everywhere else a GL skip still reads as green, even in the `suite`
  lane, which sets the variable.
- **`tests/viewers/_render_screenshots.py` and `manual_check.py`.** Both
  open `elasticFrame.mpco` with `model_h5=elasticFrame.model.h5`, and
  `tests/fixtures/results/` has only the `.mpco`. I verified that the
  file is missing; I did not run the scripts.
- **Three qt-marked files under `tests/studio/`** (`test_phase_selector.py`,
  `test_refresh.py`, `test_watch.py`) run in no CI lane. The qt lane
  greps only `tests/viewers`, and every other lane deselects `qt`.
- **`viewer_bench/.../check_slots.py`** prints PASS for a slot whose
  `session.render` skipped for lack of GL. It never looks at pixels.
- **The demo contour still** had legend tick labels drawn dark on the
  dark background, and the top label ran into the title. That is one
  still under this machine's saved theme; not triaged.

## Revision 2 — review dispositions (2026-09-25)

Adversarial review (Opus): "ship-with-changes". Every finding below was
re-verified by running it before acting on it.

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | MAJOR | The plan and PR named a stale branch for the apeGmsh-wide port and proposed a second routing row. #1171's own guide says "the interactor eats the key". | **Fixed.** Coordination now names PR #1171 and keeps #1171's row as the only one; its guide links to these two. The key fix #1171 needs is given with file:line (`apegmsh-viewer-results/SKILL.md:59-60`). `spec.py:1175` is handed off to #1171. |
| 2 | MAJOR | The G-HASH and G-QAPP-ENV walks skipped `Lambda` and `ClassDef`, and G-HASH read calls only. | **Fixed.** G-HASH reads every `Load` use of `hash` outside `__hash__`. G-QAPP-ENV descends into lambdas and class bodies. Self-test cases added for each shape (`lambda`, class-body comprehension, `key=hash`, `h = hash`, `QApplication` in a lambda or class body). |
| 3 | MINOR | "Never call `restoreDockWidget`" was wrong in general: `session/_host.py:494` calls it by design (ADR 0098 A3.3.3). | **Fixed.** The guide item is scoped to navigation docks, notes that `test_dock_invariant.py` scans only the mesh and model viewers, and names the session exception. |
| 4 | MINOR | G-GHOST-BIT missed `ghost[idx] = 1`, `ghosts[ids] \|= 0x01` and `arr & 1`, and flagged `self._n_hidden_cells = 0` and `point_ghosts & 0x02`. | **Fixed where evidence justifies:** subscript writes, augmented masks, UPPER-CASE-only constants, and HIDDENPOINT (0x02) for `point` ghost arrays. **Pinned as a known hole:** `arr & 1` on a name without "ghost" (a `_CLEAN` case says so). |
| 5 | MINOR | 7 of 15 one-line mutants survived. | **Fixed.** A 21-mutant pass (the reviewer's 7, adapted, plus 14 more): **21/21 killed**. Plus 1 runtime mutant for the `set_visible` check, also killed. Table below. |
| 6 | MINOR | G-ACTORS's `self` exemption hides the dead walk in `Diagram.set_visible` (`_base.py:532`). | **Decided: an override-completeness check**, `test_every_rendering_diagram_overrides_set_visible` in `test_deform_follow_contract.py`. It passes today (all 15 kinds override) and fails when an override is deleted. Rationale under "Rejected approaches". |
| 7 | MINOR | Three `_IMPORT_ALLOW` budgets of 1 had 0 hits and passed through `elif hits and …`. | **Fixed.** The ratchet now fails on `len(hits) < budget`. The three stale entries (`glyph_helpers`, `measure_overlay`, `origin_markers_overlay`) are deleted. A ratchet self-test was added. |
| 8 | MINOR | The CHANGELOG said "red on purpose… merges after it", which will be false once merged. | **Fixed.** Reworded to stay true after the merge order is followed. The section is still insert-only against `main`. |
| 9 | NIT | Five items: the dim-keys docstring said "swallows"; the pass count was wrong; two pointer phrases spanned line breaks; `APEGMSH_EXPECT_GL` coverage was not stated; the §6 traps had no pointers. | **Fixed:** the docstring is corrected; the count reads 14/1 (see the gates section); both phrases now grep on one line (all 20 checked); the `EXPECT_GL` coverage (3 files) is in the guide and under "Found along the way"; the traps moved to `viewer_lessons.md` with sources, so the guide is 111 lines. |
| 10 | note | `scratchpad_shots/render_scalar_bar_sheets.py:117,163` still uses `AddActor2D` / `RemoveActor2D`. | **Noted** under "Found along the way". Out of scope. |

### Mutation pass (revision 2)

Each mutant is a one-line change to a collector. It counts as **killed**
when that file's self-tests fail on it.

| Mutant | Result |
|---|---|
| R1 hidden-bit test `not in (bit, ~bit)` → `!= bit` | killed |
| R2 literal-on-left operand order dropped | killed |
| R3 `_BIT_OPS = (BitAnd,)` | killed |
| R4 `~` (Invert) branch dropped | killed |
| R5 `HIDDEN and CELL` → `HIDDEN` | killed |
| R6 guard position `<` → `<=` | killed (pinned by `guard_as_receiver`) |
| R7 `setattr` removed from G-ACTORS | killed |
| M8 `RemoveActor2D` row dropped | killed |
| M9 G-VTK-REMOVED reads nothing | killed |
| M10 no transitive guard helpers (`_lazy_qt`) | killed |
| M11 walk skips lambdas and class bodies again | killed |
| M11b lambdas and class bodies as their own scopes | killed |
| M12 guard position by line only | killed |
| M13 `__hash__` exemption dropped | killed |
| M14 G-HASH reads calls only | killed |
| M15 constant check no longer UPPER-CASE-only | killed |
| M16 point-array bit ignored | killed |
| M17 augmented-assign branch dropped | killed |
| M18 subscript-write branch dropped | killed |
| M19 G-ACTORS owner exemption for every receiver | killed |
| M20 ratchet ignores zero-hit budgets | killed |
| runtime: `ContourDiagram.set_visible` deleted | killed |

### Reruns (revision 2)

- **Acceptance.** All five rows rerun on the same `git archive` trees give
  the same counts as the table above. The test functions FAIL on every
  pre-fix and intermediate tree and PASS on the final fix trees and HEAD.
  G-HASH is the exception: it passes only on the `loads_tab.py`-patched
  scratch copy.
- **pytest.**
  - `test_viewer_recurrence_guards.py`: 6 passed, 1 failed (`test_g_hash`, the live incident).
  - `test_viewer_state_contract.py`: 9 passed.
  - `test_deform_follow_contract.py`: 3 passed.
  - `test_dim_filter_keys.py`: 3 passed.
  - `tests/test_changelog_structure.py`: 4 passed.
  - Total: 25 passed, 1 failed.
- **ruff 0.15.9:** clean on the four touched test files.
- **`scripts/sync_skill.py --check`:** in sync.

## Open questions

- **`test_dim_filter_keys.py` has a hole.** Its collector reads only a
  *literal* first argument, so the pre-fix loop
  `for _key in (...): plotter.add_key_event(_key, ...)` is invisible. It
  caught 44f8dd50^ only through the literal `"4"` line. Should it
  resolve loop-bound keys?
- **Letter shortcuts.** Should the mesh and model viewers' `add_key_event`
  letters (`h`/`i`/`r`/`u`/`y`/`e`/`n`/`b`) work with focus in a dock?
  If yes, they need `ApplicationShortcut` as the digits did.
- **Rules owed a decision:** `render_points_as_spheres=True` (5 sites,
  GL-dependent) and spin-box keyboard tracking (17 candidates).
- **Phase 6.** After about 10 viewer PRs, count review and post-merge
  findings that match a row of the baseline table. If the count does not
  drop, stop investing in the guides; keep the guards.
