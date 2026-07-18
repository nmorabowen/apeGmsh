# 06 — Color Map Editor Dock

**Status:** pending  ·  **Cost:** ~5 days  ·  **Depends on:** 04 (active-objects)

## Goal

Introduce a shared color-map editor dock that automatically follows the active diagram /
representation. The dock contains: array selector, colormap preset picker, range editor
(with auto-rescale and lock-across-stages buttons), color stops, opacity curve, log-scale
toggle, and a "show scalar bar in view" toggle.

## Why

Today each diagram type has its own `_styles.py` knob ladder for colormap, clamp, and
log-scale, partially exposed through the per-diagram settings panel. A user who wants
to change colormaps for three Contour diagrams has to open each one, find the dropdown,
change it. No way to apply a colormap to multiple diagrams at once. Scalar bar settings
are scattered across multiple places.

ParaView pattern: one dedicated dock binds to the active representation. Editing the
colormap there is the canonical workflow; per-rep panels have a small "color by …"
dropdown that just selects the array, then a button to open the editor.

## ParaView reference

- Workflow: [`paraview-flows/index.html` → "Change color array → transfer function rebinds → scalar bar updates"](../paraview-flows/index.html)
- Files:
  - `Qt/ApplicationComponents/pqColorMapEditor.h` (637 lines) — the dock widget.
  - `Qt/ApplicationComponents/pqColorOpacityEditorWidget.h` — the transfer-function curve editor.
  - `Remoting/Views/vtkSMTransferFunctionManager.h` — shared LUTs keyed by array name.
  - `Remoting/Views/vtkPVDiscretizableColorTransferFunction.h` — the LUT itself.

## Files to add / modify

**New:**
- `src/apeGmsh/viewers/ui/_color_map_editor.py` — the dock widget. Contains
  array selector, preset picker, range fields, scalar-bar toggle.
- `src/apeGmsh/viewers/ui/_transfer_function_widget.py` — the actual color-stops +
  opacity curve editor (`QWidget` with a custom paint event; drag stops with the mouse).
- `src/apeGmsh/viewers/core/_lut_manager.py` — shared lookup-table registry keyed by
  array name. Two diagrams that color by "stress_vm" share one LUT by default.
- `src/apeGmsh/viewers/diagrams/_presets.py` — curated preset colormaps (Viridis,
  Coolwarm, Turbo, Rainbow, Plasma, Inferno, Magma, Cividis, Hot, Cool). Use Matplotlib's
  cmap definitions to avoid hand-rolling.

**Modify:**
- `src/apeGmsh/viewers/diagrams/_base.py` — `Diagram` exposes `color_array_name`,
  `lut` (a `LUT` reference into `LUTManager`) as first-class properties. The diagram
  asks the manager for a LUT instead of owning one.
- `src/apeGmsh/viewers/diagrams/_styles.py` — colormap-related fields move *off* the
  style class (they're now on the LUT). Style retains: visibility, opacity, line-width.
- `src/apeGmsh/viewers/results_viewer.py` — dock the editor; bind it to
  `active.activeLayerChanged` so it follows the user's outline selection.

## API sketch

```python
# _lut_manager.py
class LUTManager:
    """One per viewer window. Stores shared lookup tables keyed by array name."""

    def get(self, array_name: str) -> LUT:
        """Returns existing LUT or creates a default one."""
    def list(self) -> list[LUT]: ...

class LUT(QObject):
    """A named lookup table. Multiple diagrams may reference the same instance."""

    changed = Signal()    # any property change → diagrams that reference re-color

    @property
    def array_name(self) -> str: ...
    @property
    def color_stops(self) -> list[tuple[float, tuple[int,int,int]]]: ...
    @property
    def opacity_curve(self) -> list[tuple[float, float]]: ...
    @property
    def range(self) -> tuple[float, float]: ...
    @property
    def log_scale(self) -> bool: ...
    @property
    def preset(self) -> str: ...   # name of last applied preset

    def apply_preset(self, name: str) -> None: ...
    def rescale_to_data_range(self, samples: np.ndarray) -> None: ...
    def lock_range(self, locked: bool) -> None: ...
```

When a `LUT` changes, every diagram observing it re-colors. The Color Map Editor binds to
the LUT of the active diagram (via `active.activeLayerChanged`).

## Risks

- **The transfer-function curve editor is the most custom-paint Qt widget in the
  codebase.** Mitigation: start ultra-simple — just color stops with linear interpolation,
  no opacity curve in v1. Add the opacity curve in a follow-up if needed. Scope discipline.
- **Shared LUTs may surprise users.** "I changed the colormap on Contour A and Contour B
  changed too!" ParaView has the same issue and solved it with a "Use Separate
  Colormap" toggle per representation. We add the same toggle: by default shared,
  toggle to detach.
- **Range lock across stages.** A diagram showing stress at step 0 has range [0, 50];
  step 100 has [0, 800]. Without locking, the colormap rescales constantly. Lock = use a
  fixed range. Auto-compute the locked range across all stages — but that's expensive
  for big results. Mitigation: lazy compute on first lock, cache.
- **Matplotlib dependency for presets.** Already a transitive dep; verify and pin. If
  problematic, hand-pickle the 10 preset arrays as a small `.npz` and ship them.

## Done criteria

- [ ] Color Map Editor dock visible in `results.viewer` (toggle via View menu).
- [ ] Selecting a Contour layer in the outline → editor displays that layer's LUT.
- [ ] Changing the preset, range, or stops updates the diagram and scalar bar in real time.
- [ ] Two Contour diagrams coloring by "stress_vm" share a LUT by default; toggle
      "Use Separate Colormap" on one to detach.
- [ ] Range lock works across stages; rescale-to-data-range computes from active stage.
- [ ] Scalar bar visibility toggle in editor flips the in-view scalar bar.
- [ ] 10 presets ship; cycling through them on a live diagram is smooth.
- [ ] Existing diagram tests still pass (style migration didn't break them).

## Out of scope

- Per-cell or per-point lookup tables (e.g. categorical). Stick to scalar continuous.
- 2D transfer functions (two arrays mapping to color). ParaView has this; we do not need.
- Custom user-defined presets saved to disk. Defer to settings dialog work (plan 07).
- The Color Legend / Scalar Bar UI editor (position, font, ticks). Use defaults for now;
  expose later if asked.
- Vector-field "color by magnitude" auto-computation. Today this is hardcoded per
  diagram; it stays hardcoded in this plan. The magnitude unification is a separate
  problem (see `future/`).
