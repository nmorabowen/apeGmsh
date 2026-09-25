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

Read this before you claim a viewer change works, or open a viewer PR. A green suite is not a
picture:
- #620's contours stayed undeformed under a green CI;
- an Add button that could never work sat "under 2333 green tests" (bb0f3388);
- the bench transient stood still while every still looked fine (`viewer_bench/README.md` ›
  "`loadConst` freezes").

Look at the pixels. Checking code is `.claude/skills/apegmsh-viewers-change/`.

## 0. Environment

- **Interpreter.** Use a venv with the viewer extras (`scripts/make-venv.bat` installs
  `[all]`); the bare system `python` may lack pyvista and Qt. `python -m apeGmsh doctor`
  checks the interpreter, path drift and the GL stack.
- **Reference vs candidate.** Scripts import the editable install, which points at the main
  checkout. Set `PYTHONPATH=<worktree>/src` for the candidate and
  `PYTHONPATH=<clean origin/main worktree>/src` for the reference.
- **Fixture.** Use one fixture for both:
  - `Results.demo(path="<dir>")` writes `demo_results.h5` + `demo_model.h5` (a tiny line
    model);
  - or a `viewer_bench` case (`viewer_bench/README.md` › "Running it").
- **Pin the look.** Stills inherit the theme saved in QSettings (`THEME`). Pin the palette
  and the window size, and restore any theme or density you change.

## 1. Stills: the primary oracle (headless, no window)

```bash
PYTHONPATH=<src> python -m apeGmsh.viewers render <results.h5> ref.png \
    --view contour --component displacement_z --camera iso --window 1280x720 --json
```

- `--view` is one of `mesh|contour|deformed|reactions`; `--pack DIR` renders the whole set. A
  saved session pane: `python -m apeGmsh.results.session render <x.viewer-session.json> s.png`.
- **Skip ≠ pass.** Without GL, or with `APEGMSH_SKIP_VIEWER=1`, the command still exits 0 and
  prints `"written": []`, so check that the list is not empty. `check_slots.py` prints PASS
  even when `session.render` skipped, and never looks at pixels. In tests, set
  `APEGMSH_EXPECT_GL=1` so a skip fails.
- Render the reference and candidate the same way, then open both PNGs (Read tool) side by
  side.
- To compare automatically, use `frames_match_or_skip` / `cleared_or_skip`
  (`tests/viewers/conftest.py`). Windows GL leaves residue and Mesa is exact, so never use
  `array_equal`.
- `Plotter.render()` is lazy: call `plotter.render_window.Render()` before measuring pixels
  (`test_element_hide_pixels.py`).

## 2. Inspect

- **Clipping and overlap.** The legend title against the top tick label; pane headers at the
  240 px pane floor; docks eating the viewport (ADR 0088 requires the viewport to keep ≥ 50 %
  of 1280 px).
- **Stale state.** A layer left at the reference configuration while the substrate deforms
  (#620); something hidden that comes back; a legend with no occupied slot behind it.
- **Empty or flat frame.** Assert `img.std() > 0`. Also look for the mesh vanishing when a
  contour is hidden (adversarial review › "Contour eye → mesh vanishes").
- **Contrast.** Check all three palettes: `catppuccin_mocha`, `neutral_studio`, `paper`
  (ADR 0087 › "INV-6").
- **Focus.** A global shortcut must still work with focus in a dock
  (`internal_docs/viewer_lessons.md` › "Key bindings").
- **Design review.** For an invariant review of the stills (ADR 0087/0088), hand them to the
  `viewer-ui-designer` agent (`.claude/agents/`). It only reviews; it cannot capture.

## 3. Chrome, docks, layout: only a live window shows them

- Start from `scratchpad_shots/take_pane_host_shots.py`. It creates the QApplication first,
  patches `_layout_settings` to a temp-ini QSettings, and sets then restores theme and density.
  It composites each pane's `plotter.screenshot()` into `win.grab()`: `widget.grab()` leaves
  GL panes blank, and `grabWindow` races focus.
- Required series: ADR 0088 › "D8 — Acceptance criteria", and ADR 0098 S3 criterion 21
  (1/2/4 panes, three palettes, the 240 px floor).

## 4. Drive input programmatically

- **Session.** Set `view.contour = Contour(...)` or `session.time = Instant(...)`, or call
  `session.add_view()`; then call `frame.pane.reconciler.flush_now()`.
- **Widgets.** `button.click()`, `slider.setValue(...)`.
- **Qt events.** `qtpy.QtTest.QTest` mouse and key events. Send keys to
  `QApplication.focusWidget()` so that focus routing is tested too.
- **VTK gestures.** `iren.SetEventPosition(x, y)`, then `iren.InvokeEvent("LeftButtonPressEvent")`
  (`tests/viewers/manual_legend_gesture.py`).

## 5. A live window you launched: stop it by PID

```python
env = dict(os.environ); env.pop("QT_QPA_PLATFORM", None)   # never offscreen on Windows
p = subprocess.Popen([sys.executable, "-m", "apeGmsh.viewers", "run.h5",
                      "--restore-session", "no", "--no-save-session"], env=env)
...                                   # drive / screenshot
p.terminate(); p.wait(timeout=15)     # else: taskkill /PID <pid> /T /F — then confirm it is gone
```

- The two flags avoid a modal restore prompt and a `.viewer-session.json` written on close.
- `results.viewer(blocking=False)` returns the same `Popen`, but closes the parent's `results`.
- `g.mesh.viewer()` and `g.model.viewer()` always block, so launch them from your own
  subprocess script.
- Never run a blocking viewer inside a Jupyter kernel: it native-crashes the kernel.

## 6. Traps

- **Windows offscreen.** `QT_QPA_PLATFORM=offscreen` with any `QtInteractor` is an access
  violation (`ViewerWindow` raises on purpose). Offscreen Qt text also renders as tofu.
- **One real-window test file per process.** Tear shown panes down hard: hide, `dispose()`,
  `deleteLater()`, then pump events (`manual_legend_gesture.py`).
- **Closing a `SessionWindow`** writes `QSettings('apeGmsh','ResultsSession')`; patch
  `_layout_settings` in scripts.
- **Plot panes have no still.** `session.render` refuses them; only a live window shows them.
- **`export_animation` flashes a real window.** It is not a headless still.

## 7. Report

- Attach the reference and candidate stills to the PR. Don't commit them: `*.png` is
  gitignored outside `docs/assets`.
- Say what you inspected (palettes, sizes, gestures) and what you didn't. "Not verified" is
  acceptable; "looks fine" without a still is not.
