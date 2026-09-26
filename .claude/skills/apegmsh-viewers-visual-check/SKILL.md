---
name: apegmsh-viewers-visual-check
description: >
  Procedure for VISUALLY verifying a change to apeGmsh's viewers (results
  session/panes, legends, docks, mesh/model viewers, stills) before claiming it
  works or opening a viewer PR: reference vs candidate stills from the headless
  render doors, inspecting them for clipping/overlap/stale state/contrast, live
  windows driven programmatically and stopped by PID, and the GL/Qt traps of
  this repo. A green test suite is not a picture. Not for end users rendering
  their own results (that is apegmsh-helper).
---

# Visual check for a viewer change

Read this before you claim a viewer change works, or before you open a viewer PR. A green suite
is not a picture:
- #620's contours stayed undeformed under a green CI;
- a dead Add button sat "under 2333 green tests" (bb0f3388);
- the bench transient stood still while its stills looked fine (`viewer_bench/README.md` ›
  "`loadConst` freezes").

Look at the pixels. For checking the code itself, see `.claude/skills/apegmsh-viewers-change/`.

## 0. Environment

- A venv with the viewer extras (`scripts/make-venv.bat` installs `[all]`);
  `python -m apeGmsh doctor` checks the interpreter, path drift and GL.
- Scripts import the editable install (the main checkout): set `PYTHONPATH=<worktree>/src` for
  the candidate and `<clean origin/main worktree>/src` for the reference.
- One fixture for both: `Results.demo(path="<dir>")` (writes `demo_results.h5` +
  `demo_model.h5`), or a bench case (`viewer_bench/README.md` › "Running it").
- Stills inherit the theme saved in QSettings: pin the palette and window size, and restore any
  theme or density you change.

## 1. Stills: the primary oracle (headless, no window)

```bash
PYTHONPATH=<src> python -m apeGmsh.viewers render <results.h5> ref.png \
    --view contour --component displacement_z --camera iso --window 1280x720 --json
```

- `--view` takes `mesh|contour|deformed|reactions`; `--pack DIR` writes the set. A saved
  session pane: `python -m apeGmsh.results.session render <x.viewer-session.json> s.png`.
- **Skip ≠ pass.** No GL (or `APEGMSH_SKIP_VIEWER=1`) exits 0 with `"written": []`: check the
  list. `check_slots.py` prints PASS on a skipped still. `APEGMSH_EXPECT_GL=1` makes a skip
  fail, but only `test_render.py`, `test_render_live.py` and `test_viewers_main.py` honour it.
- **Compare.** Open both PNGs (Read tool) side by side. To compare in code, use
  `frames_match_or_skip` or `cleared_or_skip` (`tests/viewers/conftest.py`), never
  `array_equal`: Windows GL leaves residue, Mesa is exact.
- **Measuring pixels.** `Plotter.render()` is lazy, so call `plotter.render_window.Render()`
  first (`test_element_hide_pixels.py`).

## 2. Inspect

- **Clipping.** The legend title against the top tick; pane headers at the 240 px floor; the
  viewport must keep at least 50 % of 1280 px (ADR 0088).
- **Stale state.** A layer at the reference configuration while the substrate deforms (#620);
  a hidden thing that comes back; a legend with no slot behind it.
- **Empty frame.** Assert `img.std() > 0`, and watch for the mesh vanishing when a contour is
  hidden (adversarial review › "Contour eye → mesh vanishes").
- **Contrast.** Check `catppuccin_mocha`, `neutral_studio` and `paper` (ADR 0087 › "INV-6").
- **Focus.** A global shortcut must still work with focus in a dock
  (`internal_docs/viewer_lessons.md` › "Key bindings").
- **Invariant review.** Hand the stills to the `viewer-ui-designer` agent (`.claude/agents/`).
  It reviews stills; it cannot capture them.

## 3. Chrome, docks, layout: only a live window shows them

- Start from `scratchpad_shots/take_pane_host_shots.py`: QApplication first, `_layout_settings`
  on a temp ini, theme and density restored, and each pane's `plotter.screenshot()` composited
  into `win.grab()` (`widget.grab()` leaves GL panes blank).
- The screenshot series to capture: ADR 0088 › "D8 — Acceptance criteria", and ADR 0098 S3
  criterion 21 (1/2/4 panes × three palettes × 240 px floor).

## 4. Drive input programmatically

- **Session.** Set `view.contour = Contour(...)` or `session.time = Instant(...)`, or call
  `session.add_view()`. Then call `frame.pane.reconciler.flush_now()`.
- **Widgets and Qt events.** Use `button.click()`, `slider.setValue(...)` and
  `qtpy.QtTest.QTest`. Send keys to `QApplication.focusWidget()`.
- **VTK gestures.** Call `iren.SetEventPosition(x, y)`, then
  `iren.InvokeEvent("LeftButtonPressEvent")` (`tests/viewers/manual_legend_gesture.py`).

## 5. A live window you launched: stop it by PID

```python
env = dict(os.environ); env.pop("QT_QPA_PLATFORM", None)   # never offscreen on Windows
p = subprocess.Popen([sys.executable, "-m", "apeGmsh.viewers", "run.h5",
                      "--restore-session", "no", "--no-save-session"], env=env)
...                                   # drive / screenshot
p.terminate(); p.wait(timeout=15)     # else: taskkill /PID <pid> /T /F — then confirm it is gone
```

- The two flags avoid a modal prompt, and a `.viewer-session.json` written on close.
- `results.viewer(blocking=False)` returns the same `Popen`, but it closes the parent's
  `results`.
- `g.mesh.viewer()` and `g.model.viewer()` always block, so wrap them in your own subprocess.
- Never block a viewer inside a Jupyter kernel.

## 6. Traps

Before capturing, read `internal_docs/viewer_lessons.md`
› "Traps when capturing stills and live windows": the offscreen crash and tofu text, hard
teardown, QSettings writes, plot panes, and `export_animation`.

## 7. Report

- Attach the reference and candidate stills to the PR. Don't commit them: `*.png` is
  gitignored outside `docs/assets`.
- Say what you inspected and what you didn't. "Not verified" is acceptable; "looks fine" with
  no still is not.
