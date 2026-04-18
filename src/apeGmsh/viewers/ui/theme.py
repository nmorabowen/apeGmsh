"""
Viewer theme — palette dataclass + stylesheet factory.

Three palettes ship: ``catppuccin_mocha`` (dark), ``neutral_studio``
(dark grey radial vignette), and ``paper`` (near-white for figures).
``ThemeManager`` is a singleton observable that the viewer window
subscribes to; swapping the current palette re-renders the stylesheet
and fires observers so VTK content can re-push too.

The Palette unifies Qt chrome + viewport rendering in one dataclass,
per the `apeGmsh_aesthetic.md` spec — themes switch both together.

Usage::

    from apeGmsh.viewers.ui.theme import THEME, build_stylesheet
    window.setStyleSheet(build_stylesheet(THEME.current))
    unsubscribe = THEME.subscribe(lambda p: refresh(p))
    # later...
    THEME.set_theme("paper")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


# ======================================================================
# Palette
# ======================================================================

@dataclass(frozen=True)
class Palette:
    """Chrome + viewport roles for a viewer theme.

    Names (``catppuccin_mocha``, ``neutral_studio``, ``paper``) are
    stable IDs referenced by ``PALETTES`` and persisted via QSettings.
    Hex codes are v1 starting values per ``apeGmsh_aesthetic.md`` §4;
    the aesthetic *rules* are non-negotiable, the specific values are
    expected to be tuned against screenshots.
    """

    name: str                               # stable theme id
    # ── Qt chrome — surfaces ────────────────────────────────────────
    base: str                               # main window background
    mantle: str                             # bars, headers
    surface0: str                           # borders, input bg
    surface1: str                           # hover
    surface2: str                           # pressed
    # ── Qt chrome — text ────────────────────────────────────────────
    text: str                               # primary text
    subtext: str                            # secondary text (labels)
    overlay: str                            # muted text (empty-label)
    # ── Qt chrome — accent / semantic ───────────────────────────────
    accent: str                             # slider handles, focus borders
    icon: str                               # viewer_window toolbar icons
    success: str                            # active group
    warning: str                            # staged items, warning banner
    error: str                              # errors, picked nodes
    info: str                               # inactive group, cell data
    # ── Viewport — background ───────────────────────────────────────
    background_mode: Literal["radial", "linear", "flat_corner"]
    bg_top: str                             # linear=top · radial=center · flat=base
    bg_bottom: str                          # linear=bottom · radial=edge · flat=corner-falloff
    # ── Viewport — per-dimension idle colors (RGB 0..255) ───────────
    dim_pt: tuple[int, int, int]
    dim_crv: tuple[int, int, int]
    dim_srf: tuple[int, int, int]
    dim_vol: tuple[int, int, int]
    # ── Viewport — interaction state colors (RGB 0..255) ────────────
    hover_rgb: tuple[int, int, int]             # hovered entity
    pick_rgb: tuple[int, int, int]              # picked entity
    hidden_rgb: tuple[int, int, int]            # hidden entity (usually bg-matched)
    # ── Viewport — body palette (for multi-body coloring; v2 consumer) ──
    body_palette: tuple[str, ...]
    # ── Viewport — model-viewer outlines (BRep silhouette + feature) ─
    outline_color: str
    outline_silhouette_px: float
    outline_feature_px: float
    # ── Viewport — mesh-viewer edges (show_edges line color) ────────
    mesh_edge_color: str                    # hex color for mesh element edges
    # ── Viewport — nodes (0D glyphs) ────────────────────────────────
    node_accent: str
    # ── Viewport — axis scene / grid / bbox ─────────────────────────
    grid_major: str
    grid_minor: str
    bbox_color: str
    bbox_line_px: float
    # ── Viewport — results colormap defaults ────────────────────────
    cmap_seq: str                           # sequential (unsigned fields)
    cmap_div: str                           # diverging (signed fields)
    # ── Viewport — rendering intensity ──────────────────────────────
    ao_intensity: Literal["none", "light", "moderate"]
    corner_triad_default: bool              # default visibility of corner gizmo


# ──────────────────────────────────────────────────────────────────────
# Catppuccin Mocha (dark — chrome preserved from v1.0)
# ──────────────────────────────────────────────────────────────────────

PALETTE_CATPPUCCIN_MOCHA = Palette(
    name="catppuccin_mocha",
    # Chrome (unchanged from original Mocha stylesheet)
    base="#1e1e2e", mantle="#181825",
    surface0="#313244", surface1="#45475a", surface2="#585b70",
    text="#cdd6f4", subtext="#bac2de", overlay="#a6adc8",
    accent="#89b4fa", icon="#cdd6f4",
    success="#a6e3a1", warning="#f9e2af",
    error="#f38ba8", info="#89b4fa",
    # Background: radial Base→Crust vignette (lighter center, deeper corners)
    background_mode="radial",
    bg_top="#313244", bg_bottom="#11111b",
    # Idle per-dim (CAD-neutral — gray fills, black wire/points)
    dim_pt=(0, 0, 0),           # black — nodes
    dim_crv=(0, 0, 0),           # black — curves
    dim_srf=(210, 210, 210),    # light gray — surfaces
    dim_vol=(210, 210, 210),    # light gray — volumes
    # Interaction — gold hover, red pick, black hidden (bg-matched)
    hover_rgb=(255, 215, 0),
    pick_rgb=(231, 76, 60),
    hidden_rgb=(0, 0, 0),
    # Body palette (multi-body coloring — reserved for v2 consumer)
    body_palette=(
        "#74c7ec",  # Sapphire
        "#fab387",  # Peach
        "#a6e3a1",  # Green
        "#cba6f7",  # Mauve
        "#f5e0dc",  # Rosewater
    ),
    # Model-viewer outlines — pure black, heavier for CAD emphasis
    outline_color="#000000",
    outline_silhouette_px=2.5,
    outline_feature_px=1.5,
    # Mesh-viewer edges — black (CAD-neutral, max contrast on gray fills)
    mesh_edge_color="#000000",
    # Nodes — pure black
    node_accent="#000000",
    # Axis scene
    grid_major="#45475a",       # Surface1
    grid_minor="#313244",       # Surface0
    bbox_color="#7f849c",       # Overlay1
    bbox_line_px=1.0,
    # Results colormaps
    cmap_seq="viridis",
    cmap_div="coolwarm",
    # Rendering
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ──────────────────────────────────────────────────────────────────────
# Neutral Studio (dark grey radial — product-photography aesthetic)
# ──────────────────────────────────────────────────────────────────────

PALETTE_NEUTRAL_STUDIO = Palette(
    name="neutral_studio",
    # Chrome: dark neutral with steel-blue accent (matches body color)
    base="#141414", mantle="#1f1f1f",
    surface0="#2a2a2a", surface1="#3a3a3a", surface2="#4a4a4a",
    text="#d0d0d0", subtext="#a0a0a0", overlay="#707070",
    accent="#7aa2d7", icon="#d0d0d0",
    success="#6ca872", warning="#d4a44a",
    error="#d47272", info="#7aa2d7",
    # Background: radial #4a4a4a center → #0f0f0f edge (lighter center for pronounced vignette)
    background_mode="radial",
    bg_top="#4a4a4a", bg_bottom="#0f0f0f",
    # Idle per-dim (CAD-neutral — gray fills, black wire/points)
    dim_pt=(0, 0, 0),           # black — nodes
    dim_crv=(0, 0, 0),           # black — curves
    dim_srf=(210, 210, 210),    # light gray — surfaces
    dim_vol=(210, 210, 210),    # light gray — volumes
    # Interaction — gold hover, red pick, black hidden (bg-matched)
    hover_rgb=(255, 215, 0),
    pick_rgb=(231, 76, 60),
    hidden_rgb=(0, 0, 0),
    body_palette=(
        "#5B8DB8",  # steel blue
        "#A9A878",  # olive
        "#4A4A4A",  # graphite
        "#A8C8B5",  # mint
        "#EAE6DE",  # warm off-white
    ),
    # Model-viewer outlines — pure black, heavier for CAD emphasis
    outline_color="#000000",
    outline_silhouette_px=2.5,
    outline_feature_px=1.5,
    # Mesh-viewer edges — black (CAD-neutral, max contrast on gray fills)
    mesh_edge_color="#000000",
    # Nodes — pure black
    node_accent="#000000",
    # Axis scene
    grid_major="#3a3a3a",
    grid_minor="#2a2a2a",
    bbox_color="#9a9a9a",
    bbox_line_px=1.0,
    cmap_seq="viridis",
    cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ──────────────────────────────────────────────────────────────────────
# Paper (near-white — figure / EOS-slides aesthetic)
# ──────────────────────────────────────────────────────────────────────

PALETTE_PAPER = Palette(
    name="paper",
    # Chrome: light with deep-blue accent
    base="#FAFAFA", mantle="#F5F5F5",
    surface0="#E8E8E8", surface1="#D8D8D8", surface2="#C0C0C0",
    text="#202020", subtext="#3a3a3a", overlay="#666666",
    accent="#2E5C8A", icon="#202020",
    success="#2d8659", warning="#b8860b",
    error="#c1272d", info="#2E5C8A",
    # Background: flat #FAFAFA with soft corner falloff
    background_mode="flat_corner",
    bg_top="#FAFAFA", bg_bottom="#EFEFEF",
    # Idle per-dim (CAD-neutral — slightly darker gray on white)
    dim_pt=(0, 0, 0),           # black — nodes
    dim_crv=(0, 0, 0),           # black — curves
    dim_srf=(192, 192, 192),    # medium gray — surfaces
    dim_vol=(192, 192, 192),    # medium gray — volumes
    # Interaction — amber hover (tuned for light bg), red pick, white hidden
    hover_rgb=(224, 168, 0),
    pick_rgb=(193, 39, 45),
    hidden_rgb=(250, 250, 250),
    body_palette=(
        "#8BA8C4",  # steel blue
        "#B9B681",  # olive-tan
        "#A0C893",  # spring green
        "#2F2F30",  # rubber black
        "#E8E0C8",  # cream
    ),
    # Model-viewer outlines — heavier on white, CAD emphasis
    outline_color="#000000",
    outline_silhouette_px=3.0,
    outline_feature_px=1.8,
    # Mesh-viewer edges — dark gray (softer than pure black on white bg)
    mesh_edge_color="#303030",
    # Nodes — pure black
    node_accent="#000000",
    # Axis scene
    grid_major="#d0d0d0",
    grid_minor="#e8e8e8",
    bbox_color="#000000",
    bbox_line_px=1.0,
    # Results colormaps — cividis is colorblind-safe on light bg
    cmap_seq="cividis",
    cmap_div="BrBG",
    # AO lighter on white (too much feels dirty)
    ao_intensity="light",
    corner_triad_default=False,
)


PALETTES: dict[str, Palette] = {
    "catppuccin_mocha": PALETTE_CATPPUCCIN_MOCHA,
    "neutral_studio":   PALETTE_NEUTRAL_STUDIO,
    "paper":            PALETTE_PAPER,
}


# ──────────────────────────────────────────────────────────────────────
# Legacy aliases — map prior names from v1.0 → current canonical ids.
# Kept so saved QSettings from before the rename still resolve. Not for
# new code.
# ──────────────────────────────────────────────────────────────────────

_THEME_ALIASES: dict[str, str] = {
    "dark":  "catppuccin_mocha",
    "light": "paper",
}


# Legacy module-level symbols (back-compat for call sites importing
# PALETTE_DARK / PALETTE_LIGHT directly).
PALETTE_DARK = PALETTE_CATPPUCCIN_MOCHA
PALETTE_LIGHT = PALETTE_PAPER


# ======================================================================
# Stylesheet factory
# ======================================================================

def build_stylesheet(p: Palette) -> str:
    """Render the viewer QSS for a given palette."""
    return f"""
    QMainWindow {{
        background-color: {p.base};
    }}
    QMenuBar {{
        background-color: {p.mantle};
        color: {p.text};
        border-bottom: 1px solid {p.surface0};
    }}
    QMenuBar::item:selected {{
        background-color: {p.surface1};
    }}
    QMenu {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.surface0};
    }}
    QMenu::item:selected {{
        background-color: {p.surface1};
    }}
    QToolBar {{
        background-color: {p.mantle};
        border: 1px solid {p.surface0};
        spacing: 2px;
        padding: 2px;
    }}
    QToolBar QToolButton {{
        background-color: {p.surface0};
        color: {p.text};
        border: 1px solid {p.surface1};
        border-radius: 3px;
        padding: 4px 8px;
        font-size: 11px;
    }}
    QToolBar QToolButton:hover {{
        background-color: {p.surface1};
    }}
    QToolBar QToolButton:pressed {{
        background-color: {p.surface2};
    }}
    QToolBar QToolButton:checked {{
        background-color: rgba(100, 180, 255, 60);
        border: 1px solid rgba(100, 180, 255, 120);
    }}
    QStatusBar {{
        background-color: {p.mantle};
        color: {p.overlay};
        border-top: 1px solid {p.surface0};
        font-size: 11px;
    }}
    QSplitter::handle {{
        background-color: {p.surface0};
        width: 2px;
        height: 2px;
    }}
    QTabWidget::pane {{
        border: 1px solid {p.surface0};
        background: {p.base};
    }}
    QTabBar::tab {{
        background: {p.mantle};
        color: {p.overlay};
        padding: 6px 12px;
        border: 1px solid {p.surface0};
        border-bottom: none;
    }}
    QTabBar::tab:selected {{
        background: {p.base};
        color: {p.text};
    }}
    QTabBar::tab:hover {{
        background: {p.surface1};
    }}
    QDockWidget {{
        color: {p.text};
    }}
    QDockWidget::title {{
        background: {p.mantle};
        padding: 4px;
        border: 1px solid {p.surface0};
    }}
    /* ── Form widgets ────────────────────────────────────── */
    QComboBox {{
        background-color: {p.surface0};
        color: {p.text};
        border: 1px solid {p.surface1};
        border-radius: 3px;
        padding: 2px 6px;
        font-size: 11px;
    }}
    QSpinBox, QDoubleSpinBox {{
        background-color: {p.surface0};
        color: {p.text};
        border: 1px solid {p.surface1};
        border-radius: 3px;
        padding: 2px 4px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 16px;
        border-left: 1px solid {p.surface1};
        border-bottom: 1px solid {p.surface1};
        border-top-right-radius: 3px;
        background-color: {p.surface0};
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
        background-color: {p.surface1};
    }}
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {{
        background-color: {p.surface2};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 5px solid {p.text};
        width: 0px;
        height: 0px;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 16px;
        border-left: 1px solid {p.surface1};
        border-top: 1px solid {p.surface1};
        border-bottom-right-radius: 3px;
        background-color: {p.surface0};
    }}
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {p.surface1};
    }}
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
        background-color: {p.surface2};
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {p.text};
        width: 0px;
        height: 0px;
    }}
    QCheckBox {{
        color: {p.subtext};
        font-size: 11px;
        spacing: 6px;
    }}
    QPushButton {{
        background-color: {p.surface0};
        color: {p.text};
        border: 1px solid {p.surface1};
        border-radius: 3px;
        padding: 3px 8px;
        font-size: 11px;
    }}
    QPushButton:hover {{
        background-color: {p.surface1};
    }}
    QPushButton:pressed {{
        background-color: {p.surface2};
    }}
    QLabel {{
        color: {p.subtext};
    }}
    QSlider::groove:horizontal {{
        background: {p.surface0};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {p.accent};
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }}
    QTreeWidget {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.surface0};
        font-size: 12px;
    }}
    QTreeWidget::item:selected {{
        background-color: {p.surface1};
    }}
    QTreeWidget::item:hover {{
        background-color: {p.surface0};
    }}
    QHeaderView::section {{
        background-color: {p.mantle};
        color: {p.overlay};
        border: 1px solid {p.surface0};
        padding: 4px;
        font-weight: bold;
    }}
    QTextEdit {{
        background-color: {p.base};
        color: {p.text};
        border: 1px solid {p.surface0};
    }}
    QGroupBox {{
        color: {p.text};
        border: 1px solid {p.surface0};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 12px;
        font-weight: bold;
        font-size: 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }}
    /* ── Dialogs ─────────────────────────────────────────── */
    QDialog {{
        background-color: {p.base};
        color: {p.text};
    }}
    QLineEdit {{
        background-color: {p.surface0};
        color: {p.text};
        border: 1px solid {p.surface1};
        border-radius: 3px;
        padding: 4px 6px;
    }}
    QMessageBox {{
        background-color: {p.base};
        color: {p.text};
    }}
    """


# ======================================================================
# Theme manager (observable singleton)
# ======================================================================

class ThemeManager:
    """Global current theme + observer list.

    Observers are called with the new ``Palette`` whenever
    ``set_theme`` changes the current theme. Intended to be a singleton
    (``THEME``) but instantiable for tests.
    """

    _settings_org = "apeGmsh"
    _settings_app = "viewer"

    def __init__(self) -> None:
        self._current: Palette = self._load_saved() or PALETTE_DARK
        self._observers: list[Callable[[Palette], None]] = []

    @property
    def current(self) -> Palette:
        return self._current

    def set_theme(self, name: str) -> None:
        key = name.lower()
        key = _THEME_ALIASES.get(key, key)
        if key not in PALETTES:
            raise ValueError(f"Unknown theme: {name!r}")
        new = PALETTES[key]
        if new is self._current:
            return
        self._current = new
        self._save(new)
        for cb in list(self._observers):
            try:
                cb(new)
            except Exception:
                import logging
                logging.getLogger("apeGmsh.viewer.theme").exception(
                    "theme observer failed: %r", cb,
                )

    def subscribe(
        self, cb: Callable[[Palette], None]
    ) -> Callable[[], None]:
        """Register observer. Returns an unsubscribe callable."""
        self._observers.append(cb)

        def _unsub() -> None:
            try:
                self._observers.remove(cb)
            except ValueError:
                pass

        return _unsub

    # ── Persistence (QSettings, best-effort) ──────────────────────────

    @classmethod
    def _load_saved(cls) -> Palette | None:
        try:
            from qtpy.QtCore import QSettings
        except Exception:
            return None
        try:
            s = QSettings(cls._settings_org, cls._settings_app)
            name = s.value("theme", "catppuccin_mocha")
            key = str(name).lower()
            key = _THEME_ALIASES.get(key, key)
            return PALETTES.get(key)
        except Exception:
            return None

    @classmethod
    def _save(cls, palette: Palette) -> None:
        try:
            from qtpy.QtCore import QSettings
        except Exception:
            return
        try:
            QSettings(cls._settings_org, cls._settings_app).setValue(
                "theme", palette.name,
            )
        except Exception:
            pass


THEME = ThemeManager()


# ======================================================================
# Back-compat constants (existing call sites keep working)
# ======================================================================

BASE      = PALETTE_DARK.base
MANTLE    = PALETTE_DARK.mantle
SURFACE0  = PALETTE_DARK.surface0
SURFACE1  = PALETTE_DARK.surface1
SURFACE2  = PALETTE_DARK.surface2
TEXT      = PALETTE_DARK.text
SUBTEXT   = PALETTE_DARK.subtext
OVERLAY   = PALETTE_DARK.overlay
BLUE      = PALETTE_DARK.info
GREEN     = PALETTE_DARK.success
YELLOW    = PALETTE_DARK.warning
PEACH     = "#fab387"
RED       = PALETTE_DARK.error
BG_TOP    = PALETTE_DARK.bg_top
BG_BOTTOM = PALETTE_DARK.bg_bottom

STYLESHEET = build_stylesheet(PALETTE_DARK)


# ======================================================================
# Helper
# ======================================================================

def styled_group(title: str):
    """Create a QGroupBox with the current theme applied globally."""
    from qtpy.QtWidgets import QGroupBox
    return QGroupBox(title)
