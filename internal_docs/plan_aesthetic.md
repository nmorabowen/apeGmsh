---
title: Plan — Aesthetic System Implementation
aliases:
  - plan-aesthetic
  - aesthetic-plan
tags:
  - apeGmsh
  - plan
  - aesthetic
  - theme
  - viewers
---

# Plan: Aesthetic System Implementation

> **Status:** Delivered — three theme presets + `theme_editor` dialog shipped (see `src/apeGmsh/viewers/ui/theme_editor_dialog.py`).

Implements the design spec in
[[apeGmsh_aesthetic]] (`architecture/apeGmsh_aesthetic.md`).
Touches only the `viewers/` package; `viz/` and the external
`apeGmshViewer` are out of scope.

> [!note] Scope discipline
> Per `CLAUDE.md` §§2–3: minimum code that solves the problem,
> surgical changes, no speculative abstraction. The aesthetic
> doc is complete; this plan executes it, nothing more. v2
> features (tube rendering, section-extruded beams, shell
> thickness, deformed-shape with undeformed ghost) are
> deferred to a separate plan.

---

## 1. Goal and success criterion

Ship a theme system with three selectable themes (Neutral
Studio, Catppuccin Mocha, Paper) where each theme fully
specifies the viewport palette and rendering behavior of the
three viewers (Model / Mesh / Results). Wire theme selection
into the UI so chrome and viewport switch together.

**Success criterion.** Opening the
`examples/example_frame3D_slab_opensees_manual_results.ipynb`
and `examples/cantilever_solid_frame_gmsh.py` models in each
of the three themes produces the per-theme-per-viewer
rendering specified in aesthetic doc §§2–4. Acceptance is
manual visual review against the aesthetic doc, with
committed reference screenshots under
`tests/viewers/reference/`.

---

## 2. Scope

### v1 — this plan

- Viewport theme dataclass + three instances
- Theme-switching infrastructure (chrome + viewport coupled)
- Model viewer: flat matte shading + black BRep outlines
- Mesh viewer: soft shading + body-relative mesh-line color +
  radial vignette background + contact AO
- Results viewer: enforce theme's colormap defaults; reject
  jet/rainbow with a warning
- Axis scene: corner triad, origin triad, reference grid,
  bounding box — all four toggleable
- Node glyph aesthetic: theme-driven accent color, depth-test
  toggle, "nodes on top" mode
- UI: `View → Theme` menu with three radio actions;
  preference persistence via QSettings

### v2 — deferred (separate plan)

- Line element tube rendering (currently 1D mesh renders as
  raw lines)
- Section-extruded line mode (requires beam-section + geomTransf
  integration)
- Shell thickness-extrusion toggle (§3.3)
- Local-axis RGB mode on tubes and shells
- Node filter categories beyond display (constrained / loaded
  wire-up to broker queries)
- Results deformed-shape with undeformed outline ghost
- Cappuccino (warm-cream) theme as a fourth entry

### Non-goals

- Modifying the existing Catppuccin Mocha stylesheet in
  `viewers/ui/theme.py`. The chrome is preserved byte-for-byte;
  we add a parallel viewport theme module and a selector
  that swaps both.
- Touching `viz/` (matplotlib notebook surface) or the
  external `apeGmshViewer` subprocess.
- A runtime theme-loading / plugin mechanism. Themes are
  hard-coded constants in one file. If users later request
  custom themes, that's a v3 conversation.
- Changing interaction plumbing (`viewers/core/pick_engine`,
  `selection`, `visibility`, `navigation`). Theme is
  orthogonal to picking; leave the plumbing alone.

---

## 3. Architecture

### New files

```
src/apeGmsh/viewers/
├── theme_viewport.py         ViewportTheme dataclass,
│                             three instances,
│                             get_theme() / set_theme()
└── scene/
    ├── axis_scene.py         corner / origin / grid / bbox
    └── background.py         radial vignette + linear gradient
                              helpers (keeps mesh_scene.py lean)
```

### Touched files

```
src/apeGmsh/viewers/
├── ui/
│   ├── theme.py              + apply_theme() that swaps Qt
│   │                           stylesheet AND viewport theme
│   ├── viewer_window.py      + View → Theme menu + signal
│   └── preferences.py        + theme field (QSettings-backed)
├── scene/
│   ├── brep_scene.py         + outline pass + flat matte
│   ├── mesh_scene.py         + body-relative edge color +
│   │                           soft shading + vignette wire
│   └── glyph_points.py       + theme accent + depth toggle
├── core/
│   └── color_manager.py      + body_relative_edge_color()
├── model_viewer.py           wire theme into scene builders;
│                             subscribe to theme-change signal
├── mesh_viewer.py            same as model_viewer
└── results/Results.py        enforce theme colormap defaults
```

### Untouched

- `viewers/core/pick_engine.py`, `selection.py`,
  `visibility.py`, `navigation.py` — interaction plumbing is
  theme-orthogonal.
- `viewers/overlays/*` — overlays already take theme via
  kwargs per [[apeGmsh_visualization]] §2.5; they pick up the
  new theme palette without file-level edits.
- `viewers/ui/mesh_tabs.py`, `model_tabs.py` — tabs render
  data, not styled surfaces.
- `viz/*` — notebook matplotlib; unrelated.

---

## 4. Step-by-step

Each step has an explicit verification. The project's
`CLAUDE.md` §4 is clear that "make it work" is too weak a
criterion; every step below has a check that can fail.

### Step 1 — `theme_viewport.py` with three instances

**File**: `src/apeGmsh/viewers/theme_viewport.py` (new)

Frozen dataclass `ViewportTheme` with fields covering every
entry in aesthetic doc §§4–5 and §3.5:

- `name: Literal["neutral_studio", "catppuccin_mocha", "paper"]`
- `background_mode: Literal["radial", "linear", "flat_corner"]`
- `background_colors: tuple[str, str]` (inner→outer, or
  top→bottom for linear)
- `body_palette: tuple[str, ...]` (cycle used when a body has
  no explicit color)
- `outline_color: str`, `outline_silhouette_px: float`,
  `outline_feature_px: float`
- `mesh_line_mode: Literal["body_relative", "fixed"]`,
  `mesh_line_fixed_color: str | None`,
  `mesh_line_opacity: float`,
  `mesh_line_shift_pct: float` (30% for body_relative)
- `node_accent: str`
- `grid_major: str`, `grid_minor: str`
- `bbox_color: str`, `bbox_line_px: float`
- `cmap_seq: str`, `cmap_div: str`
- `chrome_stylesheet_name: str` (maps to existing
  `viewers/ui/theme.py` stylesheet key)
- `corner_triad_on: dict[Literal["model","mesh","results"],
  bool]`
- `origin_triad_on: dict[Literal["model","mesh","results"],
  bool]`
- `ao_intensity: Literal["light", "moderate", "none"]`
- `shading_style: Literal["flat_matte", "soft", "matched"]`
  — per viewer dict actually: use nested
  `viewer_overrides: dict`

Three module-level instances: `NEUTRAL_STUDIO`,
`CATPPUCCIN_MOCHA`, `PAPER`, with values from aesthetic
doc §4.

Module-level `_active: ViewportTheme = NEUTRAL_STUDIO` plus:

```python
def get_theme() -> ViewportTheme: ...
def set_theme(name: str) -> ViewportTheme: ...
def list_themes() -> list[str]: ...
```

**No import of PyVista or Qt at module load.** Import-time
cost is a dataclass definition and three literals.

**Verify**:

1. `pytest tests/viewers/test_theme_viewport.py` — three
   instances exist, each field populated, `set_theme("paper")`
   → `get_theme().outline_silhouette_px == 2.0`.
2. `python -c "import apeGmsh.viewers.theme_viewport"` in a
   headless env (no Qt, no VTK display) succeeds.

### Step 2 — `apply_theme` coupling layer

**File**: `src/apeGmsh/viewers/ui/theme.py` (touched)

Add `apply_theme(app_or_window, name: str) -> ViewportTheme`.
Responsibilities:

1. Look up Qt stylesheet for `name`; call
   `app.setStyleSheet(...)`. For Catppuccin Mocha this is the
   existing stylesheet; Neutral Studio and Paper need new
   stylesheets written alongside (short — reusing the
   existing `styled_group` / theme-var machinery).
2. `theme_viewport.set_theme(name)`.
3. Emit a Qt signal `theme_changed = pyqtSignal(str)` on a
   module-level singleton so open viewers can subscribe and
   re-render.

**Verify**:

- Unit: `apply_theme` mutates both chrome and viewport
  state; reading `theme_viewport.get_theme().name` matches.
- Manual: open mesh viewer, call `apply_theme(window, "paper")`
  programmatically, confirm stylesheet changes AND theme-change
  signal fires.

### Step 3 — Model viewer: outline + flat matte

**Files**: `viewers/scene/brep_scene.py`,
`viewers/model_viewer.py` (both touched)

In `build_brep_scene`:

- Accept optional `theme: ViewportTheme` argument;
  `theme = theme or get_theme()`.
- For each `plotter.add_mesh(...)` call, add
  `silhouette={"color": theme.outline_color,
  "line_width": theme.outline_silhouette_px,
  "feature_angle": 25}`. PyVista's silhouette filter
  generates BRep-feature edges from the dihedral-angle
  threshold — matches §2.1.
- `smooth_shading=False`, `lighting=True`, diffuse=0.9,
  specular=0.0 to get flat matte per §2.1.

In `model_viewer.py`:

- On construction, subscribe to `theme_changed`. On signal,
  rebuild the scene (easier than patching live props).

**Verify**:

- Manual: open
  `examples/cantilever_solid_frame_gmsh.py` in model viewer
  under Neutral Studio → bodies render flat matte with
  continuous black outlines at silhouettes and feature
  edges.
- Manual: `apply_theme(window, "paper")` → outlines thicken
  to 2.0 px silhouette on `#FAFAFA`.
- Reference screenshot: commit to
  `tests/viewers/reference/model_viewer_{theme}.png`.

### Step 4 — Mesh viewer: body-relative edges + soft shading

**Files**: `viewers/scene/mesh_scene.py`,
`viewers/core/color_manager.py`,
`viewers/mesh_viewer.py` (all touched)

Add to `color_manager.py`:

```python
def body_relative_edge_color(
    body_rgb: tuple[int, int, int],
    theme: ViewportTheme,
) -> tuple[float, float, float, float]:
    """Return RGBA edge color shifted `shift_pct` toward the
    opposite luminance extreme of `body_rgb`, at theme opacity."""
```

Luminance threshold 128/255: lighter bodies → edge shifted
toward `#000`; darker bodies → edge shifted toward `#fff`.
Shift percentage from `theme.mesh_line_shift_pct`.

In `build_mesh_scene`:

- Compute per-body edge color; pass as
  `edge_color=<color>` or a per-cell edge color array to
  `add_mesh`.
- `smooth_shading=True`, configure plotter lights for one
  key + one fill + ambient per §2.2.

**Verify**:

- Unit: `body_relative_edge_color((230,230,230), PAPER)`
  returns darker shade; `body_relative_edge_color(
  (30,30,60), NEUTRAL_STUDIO)` returns lighter shade.
- Manual: open `examples/example_plate_viewer_v2.ipynb` in
  mesh viewer → mesh lines legible but subordinate to fill,
  color shifts per body.

### Step 5 — Radial vignette background

**File**: `src/apeGmsh/viewers/scene/background.py` (new)

Three functions:

```python
def set_linear_background(plotter, top_hex, bottom_hex): ...
def set_radial_vignette(plotter, center_hex, edge_hex): ...
def set_flat_corner_falloff(plotter, base_hex, falloff_hex): ...
```

`set_linear_background` delegates to
`plotter.set_background(top, bottom)` (native).

`set_radial_vignette` — PyVista does not expose radial
gradients. Implementation: attach a screen-aligned
background plane textured with a procedurally generated
radial-gradient `numpy` array (256×256 suffices); use
`plotter.add_background_image` or attach to the far plane
with depth test off. Single actor, re-bound on window resize.

`set_flat_corner_falloff` (for Paper) — same technique with
a very soft corner darkening in the ≤5% range.

Dispatch from the viewers based on `theme.background_mode`.

**Verify**:

- Manual: Neutral Studio → radial vignette visible, brighter
  center, darker corners. Paper → near-white with faint
  corner darkening. Mocha → linear top-to-bottom gradient.
- Resize window → background resizes correctly.

**Risk**: PyVista add_background_image behavior differs
across backends (Qt vs trame). Test both. If trame breaks,
fallback to linear gradient in trame only.

### Step 6 — Axis scene builders

**File**: `src/apeGmsh/viewers/scene/axis_scene.py` (new)

Four pure functions matching [[apeGmsh_visualization]] §2.3
convention (scene builders are `def`, no session reference,
return actors/data):

```python
def build_corner_triad(plotter, theme) -> vtkActor: ...
def build_origin_triad(plotter, model_diagonal, theme,
                       depth_tested: bool = True
                       ) -> dict[str, vtkActor]: ...
def build_reference_grid(plotter, xy_bbox, z_level, theme
                         ) -> vtkActor: ...
def build_bounding_box(plotter, bbox, theme
                       ) -> tuple[vtkActor, list[vtkActor]]: ...
```

Sizing per aesthetic doc §3.5 table. Grid adaptive spacing
(nearest power of 10). Red X-ruler + green Y-ruler at major
line weight.

Each returns actors the viewer can show/hide via a
`VisibilityManager`-style wrapper.

Default state on viewer open:

- `ModelViewer`: corner triad on (dark themes) / off (Paper);
  origin triad on; grid off; bbox off.
- `MeshViewer`: same.
- `Results`: corner triad as per theme; origin triad off;
  grid off; bbox off.

Expose toggles in `viewers/ui/mesh_tabs.DisplayTab` and
`viewers/ui/model_tabs._filter_view_tabs` — these already
exist for visibility toggles; adding four axis-toggle
checkboxes is additive.

**Verify**:

- Manual: each indicator shown/hidden individually; switching
  theme updates triad hub color, grid line color, bbox color
  without re-opening viewer.
- Unit: `build_reference_grid` on a 3×4 model produces
  correct major/minor spacing (check actor bounds).
- Reference screenshots: one per indicator per theme.

### Step 7 — Node glyph aesthetic

**File**: `viewers/scene/glyph_points.py` (touched)

- `build_node_cloud` takes a `theme` arg; default color is
  `theme.node_accent`.
- Returned actor exposes an `on_top` property — setting
  `True` disables depth test (`actor.SetForceOpaque(True);
  actor.GetProperty().SetOpacity(1); plotter.disable_depth_peeling()`
  on that actor), setting `False` restores depth testing.
- Pick/hover state priority already in `ColorManager` — no
  changes needed here; the change is the *idle* color.

**Verify**:

- Manual: switching theme changes node color instantly on
  signal receipt.
- Manual: toggling "on top" shows all nodes through solid
  geometry; toggling off re-hides occluded nodes.

### Step 8 — Results viewer colormap enforcement

**File**: `viewers/results/Results.py` (touched, small)

In `Results.plot(...)` or equivalent entry:

- If `cmap is None`: use `theme.cmap_seq` for unsigned fields,
  `theme.cmap_div` for diverging.
- If `cmap in {"jet", "rainbow", "turbo"}`: emit
  `warnings.warn("Rainbow colormaps are perceptually
  non-uniform and misleading; apeGmsh defaults use {...}. See
  apeGmsh_aesthetic.md §7.")` once per call. Do NOT block —
  the user's call wins.

Also update the external-viewer dispatch in
`Results.viewer(...)` to pass the current theme name
downstream (the external viewer can ignore it for now — this
is a one-line forward-compatibility ticket).

**Verify**:

- Unit: `Results.plot(field)` in Mocha → actor cmap is
  viridis. In Paper → cividis. Explicit `cmap='jet'` emits
  warning.
- Unit: diverging field auto-selects coolwarm (Mocha) or
  BrBG (Paper).

### Step 9 — UI: theme selector + preference persistence

**File**: `viewers/ui/viewer_window.py` (touched),
`viewers/ui/preferences.py` (touched)

- Add `View → Theme` menu with three QAction entries in a
  QActionGroup (exclusive checkable). Default checked =
  `preferences.theme` (from QSettings, fallback
  `"neutral_studio"`).
- On trigger, call `apply_theme(self, name)` from Step 2 and
  persist to `preferences.theme`.
- `preferences.py` gains `theme: str` field backed by
  `QSettings("apeGmsh", "viewers").value("theme", ...)`.

**Verify**:

- Manual: menu appears, three radios, switching takes effect
  immediately, selection persists across viewer close/reopen
  and across process restart.

### Step 10 — Acceptance gallery

**File**: `tests/viewers/aesthetic_gallery.md` (new)

Run each of the following examples in each of the three
themes and commit one screenshot per (example × theme ×
viewer) combination:

- `cantilever_solid_frame_gmsh.py` (model viewer)
- `example_plate_viewer_v2.ipynb` (mesh viewer)
- `example_frame3D_slab_opensees_manual_results.ipynb`
  (results viewer)

3 examples × 3 themes × 1 viewer each = 9 screenshots.
Inline them in the gallery .md with captions tying each back
to the aesthetic doc section it demonstrates.

Reviewer approves by visual inspection against the design
reference images we discussed (SolidWorks assembly for model,
Ansys image 2 for mesh, muted-colormap result for results).

---

## 5. Testing strategy

- **Unit**: dataclass instance correctness, color math
  helpers (body-relative shift, grid spacing), enum
  membership, warning behavior on jet/rainbow.
- **Integration**: scene-builder functions return non-empty
  actors when given a minimal model; theme-switch signal
  fires and observers receive.
- **Visual regression**: committed reference screenshots
  under `tests/viewers/reference/` — SSIM > 0.9 at 256×256
  rather than byte-for-byte (GPU variance makes pixel-exact
  diff brittle). CI headless via `xvfb` or `Xvfb`.
- **Smoke**: import `apeGmsh.viewers.theme_viewport` in a
  pure-Python (no Qt, no VTK) environment succeeds.

---

## 6. Risks and open questions

- **Radial vignette rendering path (Step 5)**. PyVista does
  not expose radial gradient natively. The background-image
  approach is the safest but needs a prototype before Step 5
  starts; if it reveals backend-specific issues (trame,
  remote), we fall back to linear gradient in those backends
  and scope radial to Qt.
- **Silhouette filter on batched meshes (Step 3)**. The
  existing `EntityRegistry` merges geometry per-dimension for
  batched picking. PyVista's `silhouette=True` kwarg applies
  per-mesh; confirm it still produces clean outlines on the
  merged UnstructuredGrid — if not, outline rendering needs
  a separate pass pre-merge.
- **Theme switch on an open window (Step 2)**. Some PyVista
  properties (custom shaders, baked background textures) do
  not update live. Plan budgets a scene rebuild on theme
  change, which is simpler than selective prop-patching even
  if slightly slower. Confirm rebuild time is under 500 ms
  on a medium model; if not, revisit.
- **Hex color tuning**. Aesthetic doc §4 values are v1
  starting points. Expect to iterate once Step 10 gallery is
  in hand. The *rules* (§6, §7, RGB=XYZ) are fixed; the
  specific hex codes are expected to move.
- **QSettings key collision**. `preferences.theme` might
  collide with the existing chrome theme preference if one
  exists. Check `preferences.py` before Step 9; rename
  either side if needed.

---

## 7. Milestones

Each milestone is independently reviewable and demoable.

- **M1 — Foundation** (Steps 1, 2). Infrastructure exists.
  No user-visible change — internal only.
- **M2 — Model viewer aesthetic** (Step 3). First visible
  delivery. CAD-style geometry review is live in all three
  themes.
- **M3 — Mesh viewer aesthetic** (Steps 4, 5). Studio-render
  mesh review across themes, including the radial vignette.
- **M4 — World references** (Step 6). Corner triad, origin
  triad, grid, bbox — all four toggleable across themes.
- **M5 — Polish** (Steps 7, 8, 9). Nodes, results colormap,
  UI selector.
- **M6 — Gallery** (Step 10). Visual acceptance.

A reviewer can approve M1 before M2 begins; each builds on
the prior without blocking it.

---

## 8. References

- [[apeGmsh_aesthetic]] — the design spec this plan executes.
- [[apeGmsh_visualization]] — the module layout this plan
  touches (§§2–5).
- [[apeGmsh_principles]] — tenet **(viii)** "the viewer is
  core and environment-aware" is the motivation.
