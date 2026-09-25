# Viewer lessons that lived only in agent memory

**What this is.** Viewer traps that were written down only in the agent's
private memory (`~/.claude/projects/<apeGmsh>/memory/`), so no other agent
or human could read them. They were moved here on 2026-09-25
([plan_agent_surface_viewers.md](plan_agent_surface_viewers.md)). The
task guides in `.claude/skills/apegmsh-viewers-*/` point at these
headings.

**What this is not.**
- It is not the only viewer archive. Most viewer lessons already live
  in the repo:
  - the ADRs, especially their Context and Rejected sections;
  - `internal_docs/results_viewer_adversarial_review.md`;
  - the "Findings so far" section of `viewer_bench/README.md`;
  - the CHANGELOG `FIXED` sections;
  - the docstrings of the guard tests under `tests/viewers/`.

  Do not copy those here.
- It is not an end-user guide. How to *use* the viewers is in
  `skills/apegmsh/references/results.md`.

**Adding a lesson.** Found a new viewer trap? Add a heading here, then add
one line to the matching guide that points at it. If the trap has a shape
you can detect mechanically, add an AST guard test instead (see
`tests/viewers/test_viewer_recurrence_guards.py` for the shape).

## Key bindings in a VTK-hosted window: `add_key_event` fires only while the viewport has focus

Two sources disagreed:
- **Memory** (2026-05-31) said single-character keys such as digits are
  fine for `plotter.add_key_event`, and that only Tab is special.
- **`tests/viewers/test_dim_filter_keys.py`** (44f8dd50, 2026-08-13) and
  its CHANGELOG entry ("FIXED — dim-filter keys 0/1/2/3/4 swallowed by
  VTK QtInteractor") said VTK swallows digit keys.

A probe on 2026-09-25 (pyvistaqt 0.11.4, VTK 9.5.2, PySide6, Windows)
settled it:

| Where keyboard focus is | `add_key_event("1")` | `QShortcut` + `ApplicationShortcut` | `QShortcut`, default `WindowShortcut` |
|---|---|---|---|
| the `QtInteractor` viewport | fires | fires | fires |
| a button in a dock | **does not fire** | fires | fires |
| a `QTreeWidget` (the outline) | **does not fire** | fires | fires |

A key event sent straight to the viewport fired `add_key_event` for `1`,
`3`, `h`, Tab and Escape alike. pyvistaqt maps digits to the VTK keysyms
`"0"`–`"9"`.

The probe delivered keys through Qt (`QTest.keyClick`,
`QApplication.sendEvent`), which bypasses the native Win32 message path.
Memory also reports that a default-context `QShortcut` (Ctrl+Shift+L,
during the v3.x dock refactor) never fired from the viewport. That was
**not** reproduced at the Qt level and is unverified natively. Use
`ApplicationShortcut` either way.

So the incident was real: the Help → Shortcuts keys did nothing. But the
cause was focus, not swallowing. `add_key_event` is a VTK observer, and
VTK only sees keys while its widget has focus. Click the outline tree and
every `add_key_event` binding goes dead.

The rule that follows:
- A key the window documents as global must be a `QShortcut` with
  `ApplicationShortcut` (`win.add_shortcut(..., application=True)`).
- Keep `add_key_event` for keys that are meant to be viewport-only.

The code and `test_dim_filter_keys.py` are authoritative. The phrase
"swallows those keypresses" in the test docstring names the wrong
mechanism.

Related traps, taken from memory's `feedback_vtk_keyboard_shortcuts.md`:
- **Tab** (ADR 0045 S5 review, PR #479). pyvistaqt sets `WheelFocus`,
  so Qt treats Tab and Backtab as focus traversal and consumes them
  before VTK sees them. Catch them with an `eventFilter` on the
  viewport. The existing HUD filters (`_pick_readout_hud.py`,
  `_shortcut_help_hud.py`) show the pattern.
- **Single-letter shortcuts** need a guard so they do not fire while a
  `QLineEdit`, `QSpinBox` or similar editor has focus. `_on_q_pressed`
  in `ui/_results_window.py` is the canonical example.
- **Keep a reference** to every `QShortcut` (`self._x_shortcut = sc`),
  or garbage collection deletes it.
- **The digit `3`** is also VTK's built-in stereo toggle. Sending `3`
  to the viewport logged "Adjusting stereo mode on a window that does
  not support stereo" in the probe.

Not settled: whether the viewport-only `add_key_event` letters in the
mesh and model viewers (`h`/`i`/`r`/`u`/`y`/`e`/`n`/`b`) are meant to
stop working when focus is in a dock. That is an owner question, noted
in the plan doc.

## `importorskip` on a parent package hides a missing submodule

`pytest.importorskip("trame")` passed while `trame.ui.vuetify3` was
missing. That submodule ships in the separate `trame-vuetify` package, so
the web tests went green without running (PR #438, 2026-05-30). Import the
submodule the test actually uses. `pyvista.trame` has a sibling trap:
since pyvista 0.49 it needs the separate `trame-pyvista` package
(#1122).

## Subprocess and serve tests need `timeout=`

A hung `python -m apeGmsh.viewers` child or `serve_web` server stalled CI
for about 40 minutes (the #444 era, 2026-05-30). Give every
`subprocess.run`, `.communicate()` and `.wait()` in a viewer test an
explicit `timeout=`. The CI lanes cap files for the same reason: `timeout
300` on qt files, and `timeout-minutes: 25` on the suite.

## The viewer logger does not propagate, so `caplog` sees nothing

`apeGmsh.viewers._log` sets `logger.propagate = False` so viewer
diagnostics do not bleed into the root logger. A test that asserts on
`caplog` therefore sees nothing. Monkeypatch
`apeGmsh.viewers._log.log_action` (or `log_error`) and assert on the calls
instead.

## Broad `-k` viewer runs exhaust the offscreen GL context

A single process that builds hundreds of offscreen `pv.Plotter`s runs out
of GL contexts. One broad `-k` selection produced 376 spurious errors,
yet every file passed when run alone (2026-05-30, confirmed a second
time). This is why real-window tests are `qt`-marked and run one file
per process. Judge a viewer change by the files it touches, not by a
whole-tree count.

## Tests that set `QT_QPA_PLATFORM=offscreen` at import poison later real-window tests

Many `tests/viewers` modules call
`os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` at import time.
pytest imports every module during collection, so in a full run any later
in-process test that builds a real `QtInteractor` gets the offscreen
platform. On Windows that is a native access violation.

This was found 2026-07-07 through PR #755's headless export test. Since
then `ViewerWindow.__init__` raises a catchable `RuntimeError` for
offscreen on win32. A new real-window test belongs in the `qt` lane, or
in a subprocess.

## "The viewers can't be verified visually here" is false

Memory held this belief and corrected it twice:
- 2026-05-30: offscreen `pv.Plotter` screenshots work, both locally and
  on CI Mesa.
- 2026-06-10: the full on-screen window works on the dev desktop.

The belief was still repeated after the first correction (PRs #495,
#592, #593), and each time the verification was handed to the owner's
eyes.

Offscreen stills and live windows both work on the dev desktop. The
procedure is in `.claude/skills/apegmsh-viewers-visual-check/SKILL.md`.
