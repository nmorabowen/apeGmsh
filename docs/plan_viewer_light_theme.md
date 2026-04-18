---
title: Viewer Light Theme + Theme Switching — Plan
aliases: [viewer-light-theme, plan_viewer_light_theme]
tags: [apeGmsh, viewer, theme, ui, plan]
---

# Viewer Light Theme + Theme Switching — Plan

> [!summary] One-line
> The viewers today hardcode a single Catppuccin Mocha (dark) theme in
> `viewers/ui/theme.py`, with the palette's hex constants leaking into
> ~10 widget files. This plan adds a white / greyscale light theme,
> wires the existing (unconnected) Theme combo in `PreferencesTab`,
> and extracts the minimum palette abstraction needed so both themes
> stay in sync without each widget growing an `if dark else light`
> branch.

## Reported need

User request:

> On the viewers we now have only the cappuccino theme, lets plan for
> a light theme, this should be white with mainly black and
> greyscale. We need to figure out a way for the user to select their
> theme, and make sure the theme serves its purpose.

Current state:

- Single theme in `src/apeGmsh/viewers/ui/theme.py` — Catppuccin Mocha
  (12 palette constants + one big f-string `STYLESHEET`).
- A `Theme` `QComboBox` with items `["Dark", "Light"]` already exists
  in `src/apeGmsh/viewers/ui/preferences.py:129-133`, plus an
  `on_theme: Callable[[str], None]` callback parameter at line 52. But
  **nothing wires `on_theme`** — selecting "Light" today is a no-op.
- ~30 hardcoded hex colors live in widget files outside `theme.py`
  (grep below). Some are chrome (would break on white), most are
  semantic content (active-group green, warning yellow, error red)
  that should survive theme switching with minor contrast tweaks.

---

## Architectural context

### Today's theme surface

```
theme.py
 ├─ Constants: BASE, MANTLE, SURFACE0..2, TEXT, SUBTEXT, OVERLAY,
 │              BLUE, GREEN, YELLOW, PEACH, RED, BG_TOP, BG_BOTTOM
 ├─ STYLESHEET  (single f-string, ~240 lines of QSS)
 └─ styled_group(title) helper

Consumers
 ├─ viewer_window.py:124  → window.setStyleSheet(STYLESHEET)
 ├─ viewer_window.py:134  → plotter.set_background(BG_TOP, top=BG_BOTTOM)
 └─ viewer_window.py:201,208  → icon color "#cdd6f4" (NOT imported
                                 from theme; copy-pasted)
```

### Scattered hex constants (by role)

Grep of `src/apeGmsh/viewers/**/*.py` for `#[0-9a-fA-F]{6}`:

| File | Role | Change with theme? |
|---|---|---|
| `ui/_browser_tab.py:131,138,162,172` | Active (green) / inactive (blue) group label | **Yes** — needs darker shade on white |
| `ui/_parts_tree.py:23,24,99,150,162,180,196,230,239` | Part color, untracked warning, empty-label, dim-icon fallback, error red | **Partially** — empty-label must flip; semantic colors may need contrast-adjusted variant |
| `ui/_selection_tree.py:24-27,98,126-127,152-153` | Per-dim icon colors | Semantic (keep hue, adjust contrast) |
| `ui/constraints_tab.py:29-44,99,110-111,201` | Constraint-kind palette, empty-label, warning banner | Palette: semantic; empty-label: **yes** |
| `ui/loads_tab.py:28-34,78,89-90` | Pattern-cycle palette, empty-label, warning banner | Same as above |
| `ui/mass_tab.py:55,66-67` | Empty-label, warning banner | **Yes** |
| `ui/viewer_window.py:201,208` | Icon color (`#cdd6f4`) | **Yes** — near-invisible on white |
| `ui/preferences.py:138,140,154` | Default pick color (`#E74C3C`), color-swatch border (`#999`) | Pick color is content (stays); border must flip |
| `model_viewer.py:284,326-327,371-372` | 3D point-label shape/text colors | **Yes** — dark label on dark bg becomes dark label on white bg |
| `core/color_manager.py:31-39` | VTK `PICK_RGB`, `HOVER_RGB`, `_DIM_COLORS` (warm white, grey, blue, slate) | **Yes** — warm white geometry invisible on white bg |
| `viewers/geom_transf_viewer.py:*` | Standalone HTML/three.js viewer CSS | **Out of scope** — separate artifact, not a Qt widget |

### Two failure modes a light theme must avoid

1. **Invisible geometry.** VTK `_DIM_COLORS[0] = warm white (#E8D5B7)`
   is near-white and disappears on a white viewport background. Light
   theme needs a contrast-adjusted content palette.
2. **Hardcoded chrome bleeds through.** Even if `STYLESHEET` is
   rebuilt for light, a `#cdd6f4` icon or a `#6c7086` empty-label is
   still set via `setStyleSheet` or `QColor` at widget construction
   time and won't respond to theme change.

### Key invariants the design must preserve

1. **Single source of truth for chrome.** Widgets should not know what
   theme is active — they ask for a semantic role ("muted text",
   "warning border") and get the right hex.
2. **Live switching.** Selecting "Light" in Preferences should update
   all open viewers without a reopen. Qt supports this via
   `QApplication.setStyleSheet`, but `plotter.set_background` and
   VTK actor colors require explicit re-push.
3. **Content palette survives theme change.** Semantic colors
   (active-group green, warning yellow, error red) stay semantic —
   only their exact shade adjusts for contrast. A user who creates a
   group in dark mode should see it still highlighted as "the active
   one" in light mode.

---

## Decisions captured (resolved upfront)

| Decision | Resolution | Source |
|---|---|---|
| Palette architecture | **`Palette` dataclass** with named roles (base, mantle, text, …), one `PALETTE_DARK` + one `PALETTE_LIGHT` instance, one `build_stylesheet(palette) -> str` factory. Rejected alternative: two parallel stylesheet strings (would duplicate ~240 lines of QSS and drift). | CLAUDE.md §2 — minimum code; §1 — explicit over duplicated |
| How users select theme | **Existing `PreferencesTab` combo, wired through.** No menu bar entry, no keyboard shortcut in this plan. If user wants a menu shortcut later, it's a 5-LOC add. | Already in UI, not worth adding a second control |
| Persistence | **`QSettings("apeGmsh", "viewer")`.** Native Qt, no extra dependency, writes to the standard OS location (registry on Windows, `~/.config/apeGmsh/viewer.conf` on Linux). | Already have Qt; avoid hand-rolling a JSON-prefs file |
| Scope of "theme" | **Chrome + viewport background + VTK `_DIM_COLORS` + icon color + empty-label + warning-banner.** NOT the semantic content palette (active-group green, constraint-kind hues, load-pattern cycle) — those keep their hue but get a contrast-adjusted variant picked from the active palette. | Explicit to avoid scope creep into "redo every color" |
| Live switching vs requires reopen | **Live switching.** Every open viewer window observes a `ThemeManager` and re-pushes stylesheet + background + VTK `_DIM_COLORS` on change. | UX — nobody wants "close and reopen to see your new theme" |
| Default theme on first run | **Dark** (Catppuccin Mocha). Preserves existing behavior for users who don't touch the preference. | Zero-friction migration |
| Accent color in light mode | **Neutral dark grey**, not Catppuccin blue. Matches user's "mainly black and greyscale" direction. Slider handles, focus borders, the pick color swatch all use a `#333` accent on white. | User: "mainly black and greyscale" |

> [!note] On the "rejected" alternative
> A two-stylesheet-strings approach would have been faster to ship
> but would duplicate all ~240 lines of QSS. Every future change to
> chrome (e.g. a new button style) would need to be made twice, and
> in practice they would drift. The dataclass-+-factory route is
> ~30 LOC more upfront and eliminates the drift class of bug.

> [!warning] VTK `_DIM_COLORS` is a quiet dependency
> `ColorManager._DIM_COLORS` is currently a module-level constant
> (`core/color_manager.py:35-40`). Making it theme-aware requires
> either (a) turning it into a method that reads from the active
> palette, or (b) re-pushing default colors to all actors on theme
> change. Option (b) is surgical and preferred — `ColorManager`
> already has `reset_all()` in its public API (check the file before
> implementing).

---

## Root causes

### Bug / gap 1 — No Light stylesheet exists

`viewers/ui/theme.py:44-274` is a single f-string hard-bound to the
Catppuccin constants. No alternative palette is defined anywhere.

### Bug / gap 2 — `on_theme` callback is a no-op hook

`viewers/ui/preferences.py:129-133`:

```python
self._theme_combo = QtWidgets.QComboBox()
self._theme_combo.addItems(["Dark", "Light"])
if on_theme:
    self._theme_combo.currentTextChanged.connect(on_theme)
```

The combo exists, the callback slot exists, but `model_viewer.py` and
`mesh_viewer.py` never pass `on_theme=` when constructing
`PreferencesTab`. Selecting "Light" fires `currentTextChanged("Light")`
→ into the void.

### Bug / gap 3 — Chrome hex constants duplicated outside `theme.py`

Counted at least **18 occurrences** of Catppuccin hex literals in
widget files that should have imported from `theme.py`:

- `_browser_tab.py` — 4× `#a6e3a1`, 4× `#89b4fa`
- `_parts_tree.py` — `_PART_COLOR = "#a6e3a1"`, `_UNTRACKED_COLOR =
  "#f9e2af"`, empty-label `#6c7086`, fallback `#cdd6f4`, error `#f38ba8`
- `_selection_tree.py` — `_DIM_ICON_COLOR` dict (4 hues), fallback
  `#cdd6f4`
- `constraints_tab.py` — 12-entry kind palette + empty-label
  `#6c7086` + warning-banner `#f9e2af`
- `loads_tab.py` — 7-entry pattern cycle + empty-label `#6c7086` +
  warning-banner `#f9e2af`
- `mass_tab.py` — empty-label `#6c7086` + warning-banner `#f9e2af`
- `viewer_window.py:201,208` — icon color `#cdd6f4`
- `preferences.py:140,154` — swatch-border `#999`
- `model_viewer.py:284,326-327,371-372` — point-label shape/text
  colors

Of these, roughly half are **chrome** (empty-label, swatch-border,
icon, point-label backgrounds) and must follow the theme; the other
half are **semantic content** (active-group green, constraint-kind
palette) and need a contrast-adjusted variant but keep their hue
identity.

### Bug / gap 4 — Viewport gradient is hardcoded

`viewer_window.py:134`:

```python
self._qt_interactor.set_background(BG_TOP, top=BG_BOTTOM)
```

Called once at construction. No re-push path on theme change.

### Bug / gap 5 — `_DIM_COLORS` is theme-agnostic

`core/color_manager.py:35-40`:

```python
_DIM_COLORS = {
    0: np.array([232, 213, 183], dtype=np.uint8),  # #E8D5B7 warm white
    1: np.array([170, 170, 170], dtype=np.uint8),  # grey
    2: np.array([91, 141, 184],  dtype=np.uint8),  # steel blue
    3: np.array([90, 110, 130],  dtype=np.uint8),  # slate
}
```

`#E8D5B7` (warm white, dim 0 = points) is near-invisible on a white
viewport. Needs a light-mode variant: near-black for points, a
slightly darker slate for volumes, etc.

---

## Fix plan — phased

> [!important] Phase ordering
> Phase 1 (palette + factory) is a **prerequisite** for everything
> else. Phases 2, 3, 4 are independent after Phase 1 and can be
> parallelized. Phase 5 (persistence) lands last.

### Phase 1 — Palette abstraction + light palette

**Goal:** `theme.py` exposes `Palette`, `PALETTE_DARK`,
`PALETTE_LIGHT`, `build_stylesheet(palette) -> str`, and a
`ThemeManager` singleton. All existing Catppuccin constants still
importable (backwards-compat alias).

**Files & changes:**

1. **`src/apeGmsh/viewers/ui/theme.py`** — rewrite:

   ```python
   from __future__ import annotations
   from dataclasses import dataclass
   from typing import Callable

   # ── Palette ───────────────────────────────────────────────────
   @dataclass(frozen=True)
   class Palette:
       """Chrome + semantic roles for a viewer theme."""
       name: str            # "dark" | "light"
       # Surfaces
       base: str            # main window background
       mantle: str          # bars, headers
       surface0: str        # borders, input bg
       surface1: str        # hover
       surface2: str        # pressed
       # Text
       text: str            # primary text
       subtext: str         # secondary text (labels)
       overlay: str         # muted text (empty-label)
       # Accent
       accent: str          # slider handles, focus borders
       # Viewport gradient
       bg_top: str
       bg_bottom: str
       # Icon color (viewer_window toolbar)
       icon: str
       # Semantic (contrast-adjusted per theme)
       success: str         # active group
       warning: str         # staged items, warning banner
       error: str           # errors, picked nodes
       info: str            # inactive group, cell data
       # VTK content colors (RGB 0..255 tuples)
       dim_pt: tuple[int, int, int]
       dim_crv: tuple[int, int, int]
       dim_srf: tuple[int, int, int]
       dim_vol: tuple[int, int, int]

   # ── Dark (Catppuccin Mocha, existing) ─────────────────────────
   PALETTE_DARK = Palette(
       name="dark",
       base="#1e1e2e", mantle="#181825",
       surface0="#313244", surface1="#45475a", surface2="#585b70",
       text="#cdd6f4", subtext="#bac2de", overlay="#a6adc8",
       accent="#89b4fa",
       bg_top="#1a1a2e", bg_bottom="#16213e",
       icon="#cdd6f4",
       success="#a6e3a1", warning="#f9e2af",
       error="#f38ba8", info="#89b4fa",
       dim_pt=(232, 213, 183),    # warm white
       dim_crv=(170, 170, 170),
       dim_srf=(91, 141, 184),
       dim_vol=(90, 110, 130),
   )

   # ── Light (white + greyscale, user-requested) ─────────────────
   PALETTE_LIGHT = Palette(
       name="light",
       base="#ffffff", mantle="#f4f4f4",
       surface0="#e0e0e0", surface1="#d0d0d0", surface2="#b8b8b8",
       text="#1a1a1a", subtext="#3a3a3a", overlay="#666666",
       accent="#333333",                # neutral dark grey, per request
       bg_top="#fafafa", bg_bottom="#e8e8e8",
       icon="#1a1a1a",
       # Semantic: darker shades that read on white
       success="#2d8659",               # darker green
       warning="#b8860b",               # dark goldenrod
       error="#c1272d",                 # darker red
       info="#1f5fa8",                  # darker blue
       # VTK: inverted contrast — dark geometry on white
       dim_pt=(30, 30, 30),
       dim_crv=(80, 80, 80),
       dim_srf=(70, 110, 150),
       dim_vol=(50, 70, 90),
   )

   PALETTES = {"dark": PALETTE_DARK, "light": PALETTE_LIGHT}

   # ── Stylesheet factory ────────────────────────────────────────
   def build_stylesheet(p: Palette) -> str:
       return f"""
           QMainWindow {{ background-color: {p.base}; }}
           QMenuBar    {{ background-color: {p.mantle};
                          color: {p.text};
                          border-bottom: 1px solid {p.surface0}; }}
           /* ... rest of the existing stylesheet, with every
                  hex replaced by a {p.*} field ... */
       """

   # ── Back-compat: keep STYLESHEET importable ──────────────────
   # (downstream modules that imported STYLESHEET directly keep
   # working until Phase 2 rewires them.)
   STYLESHEET = build_stylesheet(PALETTE_DARK)
   BASE, MANTLE, SURFACE0 = PALETTE_DARK.base, PALETTE_DARK.mantle, PALETTE_DARK.surface0
   # ... (all the old constants aliased) ...

   # ── Theme manager (singleton) ────────────────────────────────
   class ThemeManager:
       """Global current theme + observer list.

       Designed to be monkey-patchable in tests (swap the singleton).
       """
       def __init__(self) -> None:
           self._current: Palette = PALETTE_DARK
           self._observers: list[Callable[[Palette], None]] = []

       @property
       def current(self) -> Palette:
           return self._current

       def set_theme(self, name: str) -> None:
           key = name.lower()
           if key not in PALETTES:
               raise ValueError(f"Unknown theme: {name!r}")
           new = PALETTES[key]
           if new is self._current:
               return
           self._current = new
           for cb in list(self._observers):
               try:
                   cb(new)
               except Exception:
                   import logging
                   logging.getLogger("apeGmsh.viewer.theme").exception(
                       "theme observer failed: %r", cb,
                   )

       def subscribe(self, cb: Callable[[Palette], None]) -> Callable[[], None]:
           """Returns an unsubscribe function."""
           self._observers.append(cb)
           return lambda: self._observers.remove(cb)

   THEME = ThemeManager()
   ```

2. **Verify back-compat:** every existing
   `from .theme import STYLESHEET, BG_TOP, BG_BOTTOM, …` in the tree
   still resolves. Run `python -c "from apeGmsh.viewers.ui import
   theme; print(theme.STYLESHEET[:50])"` as a smoke check.

**Acceptance:**
- `theme.PALETTES["dark"]` and `theme.PALETTES["light"]` both resolve
  to `Palette` instances.
- `build_stylesheet(PALETTE_DARK) == STYLESHEET` character-for-
  character (refactor preserves existing rendering).
- `THEME.set_theme("light")` fires every subscribed observer with
  `PALETTE_LIGHT`.
- New test `tests/test_theme_palette.py` covers the above.

---

### Phase 2 — Wire the theme combo + apply theme live

**Goal:** selecting "Dark" / "Light" in `PreferencesTab` reaches
`ThemeManager` and visibly changes every open viewer.

**Files & changes:**

1. **`src/apeGmsh/viewers/ui/viewer_window.py:124`** — instead of a
   one-shot `setStyleSheet(STYLESHEET)`, subscribe to `THEME`:

   ```python
   from .theme import THEME, build_stylesheet

   def _apply_palette(palette):
       self._window.setStyleSheet(build_stylesheet(palette))
       self._qt_interactor.set_background(
           palette.bg_top, top=palette.bg_bottom,
       )
       # Also re-push icon color to toolbar actions
       self._refresh_toolbar_icons(palette.icon)

   _apply_palette(THEME.current)                    # initial
   self._unsub_theme = THEME.subscribe(_apply_palette)  # live updates
   ```

2. **`src/apeGmsh/viewers/ui/viewer_window.py:101-106`** (`closeEvent`)
   — call `self._unsub_theme()` to avoid leaking observers. Note the
   `closeEvent` already has the silent-exception bug documented in
   `plan_viewer_pg_persistence.md` Phase 0; do not regress it.

3. **`src/apeGmsh/viewers/ui/viewer_window.py:197-210`** — factor the
   icon-color into a helper `_refresh_toolbar_icons(hex)` so the
   subscription callback above can call it. The existing `_IC =
   "#cdd6f4"` (line 208) becomes a lookup from the current palette.

4. **`src/apeGmsh/viewers/model_viewer.py`** — locate the
   `PreferencesTab(...)` constructor call (around the UI-wiring
   section) and add:

   ```python
   on_theme=lambda name: THEME.set_theme(name),
   ```

5. **`src/apeGmsh/viewers/mesh_viewer.py`** — same wiring as
   `model_viewer.py`.

6. **`src/apeGmsh/viewers/ui/preferences.py:129-133`** — seed the
   combo to the current theme (not always "Dark"):

   ```python
   from .theme import THEME
   self._theme_combo = QtWidgets.QComboBox()
   self._theme_combo.addItems(["Dark", "Light"])
   self._theme_combo.setCurrentText(THEME.current.name.capitalize())
   ```

**Acceptance:**
- Manual: open viewer → Preferences → switch Theme to "Light" →
  entire window flips to white/grey instantly (no reopen).
- Manual: with two viewers open (model + mesh), switching theme in
  one updates both.
- No observer leak: opening/closing 5 viewers leaves
  `THEME._observers` at the same length (assert in a new test if easy
  to do without Qt).

---

### Phase 3 — Replace hardcoded chrome in widgets

**Goal:** widgets no longer hardcode `#6c7086`, `#cdd6f4`, `#999`.
They import from `theme.THEME.current` (or subscribe for live
update).

**Strategy:** two tiers.

**Tier A (static chrome)** — widgets that apply a color once at
construction but rarely redraw on theme change. Accept "requires
reopen for some panels" as a pragmatic limit for tab widgets that
are cheap to throw away.

Files:
- `ui/mass_tab.py:55,66-67` (empty-label + warning banner)
- `ui/loads_tab.py:78,89-90`
- `ui/constraints_tab.py:99,110-111`
- `ui/preferences.py:140,154` (swatch border)
- `model_viewer.py:284,326-327,371-372` (point-label shape/text)

Change: replace `"#6c7086"` → `f"{THEME.current.overlay}"` etc. Wrap
the setStyleSheet calls in a `_apply_chrome(palette)` method on each
tab, called from `__init__` with `THEME.current`. Optional: subscribe
to `THEME` for live update in a follow-up PR — this plan does not
require it.

**Tier B (tree foreground colors)** — `_browser_tab.py`,
`_parts_tree.py`, `_selection_tree.py`. These DO react to state
changes (active-group highlighting re-colors items via
`update_active`), so they naturally repaint on re-state. Wire the
hex lookups through `THEME.current.success` / `.info` / `.error` etc.

Files:
- `ui/_browser_tab.py:131,138,162,172` — `"#a6e3a1"` →
  `THEME.current.success`; `"#89b4fa"` → `THEME.current.info`
- `ui/_parts_tree.py:23,24` — same substitution; for `#f38ba8` →
  `THEME.current.error`
- `ui/_selection_tree.py:24-27` — `_DIM_ICON_COLOR` dict becomes a
  function or property that reads from `THEME.current`

> [!warning] Semantic hues that must survive theme switch
> Active-group "green" is a semantic role, not a literal hex. In
> dark mode `p.success = #a6e3a1` (mint green on blue-black); in
> light mode `p.success = #2d8659` (dark green on white). Both read
> as "this group is active" to the user. **Do not** collapse
> semantic roles into pure greyscale in light mode — the user asked
> for "white + black + greyscale" for the chrome, but the CAD
> affordances (active group, warning, error) still need hue. Verify
> this reading with the user if unsure.

**`ui/constraints_tab.py:29-44` and `ui/loads_tab.py:28-34`** — the
constraint-kind palette and load-pattern cycle are pure content, not
theme chrome. Leave their hex constants as-is. On light theme they
are displayed on `p.base = white`, which is already the WCAG-intended
context for saturated hues — most of those Catppuccin pastels read
acceptably on white. Flag for user review if any pair becomes hard
to read.

**Acceptance:**
- Grep `src/apeGmsh/viewers/**/*.py -e '#[0-9a-fA-F]{6}'` after this
  phase: remaining literals are either (a) inside `theme.py`,
  (b) inside `constraints_tab.py` / `loads_tab.py` content palette,
  or (c) inside `geom_transf_viewer.py` (out of scope).
- Manual: all labels / empty-states / tree foregrounds read legibly
  in both themes.

---

### Phase 4 — Viewport gradient + VTK `_DIM_COLORS`

**Goal:** the 3D viewport (pyvista viewport background + entity
default colors) follows the theme.

**Files & changes:**

1. **`src/apeGmsh/viewers/ui/viewer_window.py:134`** — already covered
   by Phase 2's `_apply_palette`. Verify it re-pushes on theme
   change.

2. **`src/apeGmsh/viewers/core/color_manager.py:35-40`** — turn
   `_DIM_COLORS` into a property that reads from the active theme:

   ```python
   @staticmethod
   def _dim_colors():
       from apeGmsh.viewers.ui.theme import THEME
       p = THEME.current
       return {
           0: np.array(p.dim_pt,  dtype=np.uint8),
           1: np.array(p.dim_crv, dtype=np.uint8),
           2: np.array(p.dim_srf, dtype=np.uint8),
           3: np.array(p.dim_vol, dtype=np.uint8),
       }
   ```

   Replace `_DIM_COLORS[dim]` call sites with `self._dim_colors()[dim]`.

3. **`src/apeGmsh/viewers/model_viewer.py`** — on theme change, call
   `color_manager.reset_all()` (confirm it exists; if not, wire a
   `color_manager.reapply_defaults()` helper). Re-pushes the new
   default RGB to every actor.

4. **`src/apeGmsh/viewers/ui/viewer_window.py`** — `_apply_palette`
   should invoke step 3 after the stylesheet flip. Ordering: Qt
   chrome first, then VTK content, then `plotter.render()`.

**Acceptance:**
- Manual: open a model with points, curves, surfaces, volumes →
  switch to light theme → all entities remain visible (no
  white-on-white).
- Manual: switch back to dark → warm-white points reappear.

---

### Phase 5 — Persistence

**Goal:** user's theme choice survives viewer close / app restart.

**Files & changes:**

1. **`src/apeGmsh/viewers/ui/theme.py`** — extend `ThemeManager`:

   ```python
   def __init__(self) -> None:
       self._current: Palette = self._load_saved() or PALETTE_DARK
       self._observers: list[Callable[[Palette], None]] = []

   @staticmethod
   def _load_saved() -> Palette | None:
       try:
           from qtpy.QtCore import QSettings
           s = QSettings("apeGmsh", "viewer")
           name = s.value("theme", "dark")
           return PALETTES.get(str(name).lower())
       except Exception:
           # Qt not importable (e.g. headless test) — fall through
           return None

   def set_theme(self, name: str) -> None:
       # ... existing logic ...
       try:
           from qtpy.QtCore import QSettings
           QSettings("apeGmsh", "viewer").setValue("theme", new.name)
       except Exception:
           pass  # best-effort persistence
   ```

2. Verify that the first `THEME` reference during Python import does
   NOT touch Qt (otherwise importing `theme.py` in a headless test
   fails). The lazy `_load_saved` pattern above handles this — Qt is
   imported inside the staticmethod, not at module level.

**Acceptance:**
- Manual: switch theme → close viewer → reopen → theme persists.
- New test `tests/test_theme_persistence.py` with a `QSettings` stub:
  `set_theme("light")` writes "light"; next `ThemeManager()` reads
  "light".

---

## Tests

All new tests go under `tests/`. Theme tests are headless —
palette/builder/manager are pure Python. We do NOT write Qt
integration tests for the live-switch flow; that is manual per
"Acceptance" above.

### `tests/test_theme_palette.py` (new)

```python
"""Phase 1: palette dataclass + stylesheet factory."""
import pytest

from apeGmsh.viewers.ui import theme


def test_both_palettes_defined():
    assert "dark" in theme.PALETTES
    assert "light" in theme.PALETTES
    assert theme.PALETTES["dark"].name == "dark"
    assert theme.PALETTES["light"].name == "light"


def test_build_stylesheet_returns_non_empty_qss():
    qss = theme.build_stylesheet(theme.PALETTE_DARK)
    assert "QMainWindow" in qss
    assert theme.PALETTE_DARK.base in qss
    assert theme.PALETTE_DARK.text in qss


def test_stylesheet_uses_light_palette_for_light():
    qss = theme.build_stylesheet(theme.PALETTE_LIGHT)
    # Chrome colors from the light palette are present
    assert theme.PALETTE_LIGHT.base in qss
    assert theme.PALETTE_LIGHT.text in qss
    # And the dark palette's base is NOT (unless by coincidence)
    if theme.PALETTE_DARK.base != theme.PALETTE_LIGHT.base:
        assert theme.PALETTE_DARK.base not in qss


def test_backcompat_stylesheet_constant_still_exists():
    # Downstream code that imported STYLESHEET must keep working.
    assert theme.STYLESHEET == theme.build_stylesheet(theme.PALETTE_DARK)


def test_theme_manager_set_theme_fires_observers():
    tm = theme.ThemeManager()
    received: list[theme.Palette] = []
    unsub = tm.subscribe(lambda p: received.append(p))
    tm.set_theme("light")
    assert received == [theme.PALETTE_LIGHT]
    unsub()
    tm.set_theme("dark")
    assert received == [theme.PALETTE_LIGHT]  # unsubscribed, no new entry


def test_theme_manager_idempotent():
    tm = theme.ThemeManager()
    received: list[theme.Palette] = []
    tm.subscribe(lambda p: received.append(p))
    tm.set_theme("dark")  # already dark
    assert received == []


def test_theme_manager_rejects_unknown():
    tm = theme.ThemeManager()
    with pytest.raises(ValueError):
        tm.set_theme("solarized")


def test_observer_exception_is_logged_not_raised(caplog):
    import logging
    tm = theme.ThemeManager()
    def bad(p): raise RuntimeError("boom")
    tm.subscribe(bad)
    with caplog.at_level(logging.ERROR, logger="apeGmsh.viewer.theme"):
        tm.set_theme("light")  # must not raise
    assert any("boom" in str(r) for r in caplog.records)
```

### `tests/test_theme_persistence.py` (new)

```python
"""Phase 5: theme choice survives across ThemeManager instances."""
import pytest

pytest.importorskip("qtpy.QtCore")

from qtpy.QtCore import QSettings

from apeGmsh.viewers.ui import theme


@pytest.fixture(autouse=True)
def _clean_settings():
    s = QSettings("apeGmsh", "viewer-test")
    s.clear()
    # Swap the org so we don't pollute real user prefs
    theme.ThemeManager._settings_org = "apeGmsh"
    theme.ThemeManager._settings_app = "viewer-test"
    yield
    s.clear()


def test_theme_choice_persists():
    tm = theme.ThemeManager()
    tm.set_theme("light")
    tm2 = theme.ThemeManager()
    assert tm2.current is theme.PALETTE_LIGHT
```

(Only lands if Phase 5 ships. If the plan ends at Phase 4, skip this.)

---

## Visual QA checklist (manual)

After all phases land, a reviewer runs through:

1. Launch a viewer → default is dark (matches today).
2. Preferences → Theme → "Light" → chrome flips to white/grey
   instantly.
3. All four dim types (points, curves, surfaces, volumes) remain
   visible in light mode.
4. Active group (green) / inactive group (blue) labels in Browser tab
   are still distinguishable in light mode.
5. Warning banner in Loads / Constraints / Mass tabs is still a
   warning (yellow/orange, not muddy).
6. Error messages (red) still read as error.
7. Pick (selection) highlight still pops against both backgrounds.
8. Hover gold highlight still works.
9. 3D point labels (node tags) are legible — dark-text-on-light-
   background in light theme.
10. Toolbar icons are visible in both themes.
11. Close viewer → reopen → theme is remembered.
12. Open two viewers at once → switching theme in one updates the
    other (if Phase 2 is complete).

---

## What this plan does NOT change

Per `CLAUDE.md` §3 (Surgical Changes), the following are **out of
scope** and should be left untouched:

- `geom_transf_viewer.py` — standalone HTML/three.js viewer with its
  own CSS. Themeable later if needed; not a Qt widget.
- `constraints_tab.py` / `loads_tab.py` **content palettes** (the
  kind-to-color maps). They are semantic content, not theme chrome.
  If the user later reports readability issues on white, pick this
  up as a follow-up tuning pass.
- `ColorManager.PICK_RGB` / `HOVER_RGB` — intentional
  selection/highlight colors. They read on both backgrounds. Leave
  alone.
- Adding a third or custom theme — architecturally supported by
  `PALETTES` dict but not requested.
- Menu-bar / keyboard-shortcut access to theme switching —
  `PreferencesTab` combo is enough.
- Per-widget theme overrides — every widget follows the global theme.
- VTK-level color-blind-safe palette — separate accessibility ticket.
- Migrating existing `STYLESHEET` / `BG_TOP` / `BG_BOTTOM` imports
  one-by-one. The Phase 1 back-compat aliases let downstream code
  keep working; modernize as part of Phases 2–3 touches, not a
  separate mass-rename pass.

---

## Acceptance — overall

The feature is complete when:

1. **All new tests pass** (`pytest tests/test_theme_*.py`).
2. **All existing tests still pass** (`pytest tests/`).
3. **Manual visual QA** — all 12 checklist items pass on the user's
   target machine.
4. **Grep cleanliness** — `rg '#[0-9a-fA-F]{6}' src/apeGmsh/viewers/`
   returns only:
   - `theme.py` (both palettes)
   - `constraints_tab.py` and `loads_tab.py` content palettes
   - `geom_transf_viewer.py` (out of scope)
   - `color_manager.py` (PICK/HOVER, which are intentional)
5. **No observer leak** on viewer open/close cycles.
6. **Theme persists across restart** (Phase 5).
7. `docs/architecture.md` gets a one-line note that the viewer has a
   theme contract routed through `ThemeManager`.

---

## Cheat sheet for implementation order

```
1. Phase 1 — theme.py rewrite (1 file, ~150 LOC source)
2. Run existing tests — confirm back-compat (STYLESHEET still there)
3. Add tests/test_theme_palette.py — confirm green
4. Phase 2 — wire on_theme + _apply_palette (3 files, ~40 LOC)
5. Manual: switch theme, verify chrome flips
6. Phase 4 — viewport + VTK _DIM_COLORS (2 files, ~25 LOC)
7. Phase 3 — replace hardcoded hex in widgets (~8 files, ~30 LOC)
   [can be parallelized with 4 or 6]
8. Manual visual QA pass in both themes
9. Phase 5 — QSettings persistence (1 file, ~15 LOC)
10. Add tests/test_theme_persistence.py — confirm green
11. Update docs/architecture.md (one line)
12. CHANGELOG entry
```

Net delta: ~250 LOC source, ~120 LOC tests.

---

*Cross-references:*
[[architecture]] · [[apeGmsh_architecture]] ·
[[plan_viewer_pg_persistence]] · [[plan_viewer_box_select]] ·
`src/apeGmsh/viewers/ui/theme.py` ·
`src/apeGmsh/viewers/ui/preferences.py` ·
`src/apeGmsh/viewers/ui/viewer_window.py` ·
`src/apeGmsh/viewers/core/color_manager.py`
