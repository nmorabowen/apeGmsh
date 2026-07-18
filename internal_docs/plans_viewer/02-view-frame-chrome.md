# 02 — View Frame Chrome

**Status:** pending  ·  **Cost:** ~2 days  ·  **Depends on:** none

## Goal

Add a thin toolbar directly above the PyVista interactor in all three viewers, holding
discoverable camera controls (reset, isometric, XY/XZ/YZ presets, ortho/persp toggle,
axis-triad toggle, screenshot). Make the toolbar extensible — diagrams and overlays can
register their own buttons without touching the chrome class.

## Why

Camera presets today are keyboard-only and inconsistently bound across the three viewers
(some support `5` for isometric, some don't, none document it). New users have no way to
discover that `R` resets the view or that screenshots even exist.

ParaView pattern: every render view has a `pqViewFrame` — a `QWidget` wrapping the
viewport with a title bar of registered `QAction`s. Plugins extend it via
`addTitleBarAction`. The standard set (Split, Maximize, Close, view-type switcher) lives
there; plugins add view-specific buttons (annotation tools, axes toggle).

## ParaView reference

- Workflow: [`paraview-flows/index.html` → "Click Split → multi-view re-tiles"](../paraview-flows/index.html) (illustrates the registration pattern, even though we're not building multi-view yet)
- Files:
  - `Qt/Components/pqViewFrame.h` / `.cxx` — the frame widget + button registry.
  - `Qt/Components/Resources/UI/pqViewFrame.ui` — the toolbar layout.
  - `Qt/Components/pqViewFrameActionsInterface.h` — plugin extension hook.

## Files to add / modify

**New:**
- `src/apeGmsh/viewers/ui/_view_frame.py` — `ViewFrame` widget that wraps the existing PyVista `QtInteractor` and adds a top `QToolBar` with the standard buttons.

**Modify:**
- `src/apeGmsh/viewers/ui/viewer_window.py` — replace the bare `QtInteractor` with a `ViewFrame(QtInteractor)`. Expose `view_frame.add_action(QAction)` for diagrams/overlays.
- `src/apeGmsh/viewers/results_viewer.py`, `mesh_viewer.py`, `model_viewer.py` — wire up the keyboard shortcuts that already exist (`r`, `1`/`3`/`5`, etc.) to the new `QAction`s so the chrome and the keyboard stay in sync.

## API sketch

```python
# _view_frame.py
class ViewFrame(QWidget):
    """Container: top toolbar + PyVista interactor below."""

    def __init__(self, plotter: BackgroundPlotter, parent=None): ...

    # Standard actions wired by default:
    #   reset_camera, iso, xy, xz, yz, parallel_toggle, axes_toggle, screenshot
    @property
    def standard_actions(self) -> dict[str, QAction]: ...

    def add_action(self, action: QAction, *, position: str = "right") -> None:
        """Plugins / overlays add their own button. `position` ∈ {'left','right'}."""

    def remove_action(self, action: QAction) -> None: ...
```

Buttons use Qt's built-in standard icons (`QStyle.SP_*`) where possible to avoid shipping
icon files. Tooltips include the keyboard shortcut.

## Risks

- **PyVista's own keyboard handling** may conflict. PyVista binds some keys to its own
  camera operations (`r`, `w`, `s`). Mitigation: use Qt `QShortcut(Qt.ApplicationShortcut)`
  per the existing pattern documented in memory (`feedback_vtk_keyboard_shortcuts.md`),
  not PyVista's `add_key_event`.
- **Screenshot path**: PyVista has `plotter.screenshot()`. Wrap it with a Qt file dialog.
  Defer the multi-option dialog (resolution/transparency) — single-click screenshot now,
  fancy dialog later.
- **Axes triad** is already implemented per-viewer differently. Unify on the PyVista
  `add_axes()` call. If a viewer overrides the placement, allow that via constructor arg.

## Done criteria

- [ ] All three viewers show the toolbar above the viewport.
- [ ] Buttons present: reset camera, isometric, XY, XZ, YZ, parallel/perspective toggle,
      axes triad toggle, screenshot.
- [ ] Hovering each button shows tooltip with keyboard shortcut (when one exists).
- [ ] Clicking each button has the expected effect on the camera/interactor.
- [ ] Keyboard shortcuts still work — actions are bound to the same `QAction`s, so a
      shortcut and a button click route through the same code.
- [ ] At least one overlay or diagram in `results.viewer` adds a custom button via
      `view_frame.add_action(...)` (pick one — e.g. the section-cut overlay adds a
      "Flip section side" button). Proves extensibility.
- [ ] Toolbar can be hidden via View menu (one toggle for the whole chrome).

## Out of scope

- Multi-view layout (splitting the viewport into multiple panes). That's a much larger
  change tracked under `future/`.
- A view-type switcher dropdown (3D / chart / table). Our viewers are 3D-only.
- The Save Screenshot multi-option dialog (resolution, transparency, multi-view stitch).
  One-click screenshot is enough for now.
- Per-viewport active-state indication (border color changes when active). Only matters
  with multi-view.
