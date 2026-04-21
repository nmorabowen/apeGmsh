"""
apeGmshViewer theme — palette dataclass + stylesheet factory.

Standalone counterpart to ``apeGmsh.viewers.ui.theme``. The two viewers
share the same visual language (see ``architecture/apeGmsh_aesthetic.md``)
but persist their state independently:

- Current theme id  -> QSettings under ``apeGmshViewer/viewer``
- Custom themes     -> ``<config>/apeGmshViewer/themes/*.json``

``ThemeManager`` is an observable singleton (``THEME``); the viewer
window subscribes so swapping the current palette re-renders the
stylesheet and fires observers for the VTK viewport to re-push.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


# ======================================================================
# Palette
# ======================================================================

@dataclass(frozen=True)
class Palette:
    """Chrome + viewport roles for a viewer theme."""

    name: str                               # stable theme id
    # -- Qt chrome -- surfaces --------------------------------------
    base: str                               # main window background
    mantle: str                             # bars, headers
    surface0: str                           # borders, input bg
    surface1: str                           # hover
    surface2: str                           # pressed
    # -- Qt chrome -- text ------------------------------------------
    text: str                               # primary text
    subtext: str                            # secondary text (labels)
    overlay: str                            # muted text (empty-label)
    # -- Qt chrome -- accent / semantic ------------------------------
    accent: str                             # slider handles, focus borders
    icon: str                               # toolbar icons
    success: str
    warning: str
    error: str
    info: str
    # -- Viewport -- background -------------------------------------
    background_mode: Literal["radial", "linear", "flat_corner"]
    bg_top: str                             # linear=top / radial=center / flat=base
    bg_bottom: str                          # linear=bottom / radial=edge / flat=corner-falloff
    # -- Viewport -- per-dimension idle colors (RGB 0..255) ---------
    dim_pt: tuple[int, int, int]
    dim_crv: tuple[int, int, int]
    dim_srf: tuple[int, int, int]
    dim_vol: tuple[int, int, int]
    # -- Viewport -- interaction state colors (RGB 0..255) ----------
    hover_rgb: tuple[int, int, int]
    pick_rgb: tuple[int, int, int]
    hidden_rgb: tuple[int, int, int]
    # -- Viewport -- body palette (multi-body coloring) -------------
    body_palette: tuple[str, ...]
    # -- Viewport -- BRep outlines ----------------------------------
    outline_color: str
    outline_silhouette_px: float
    outline_feature_px: float
    # -- Viewport -- mesh-edge color --------------------------------
    mesh_edge_color: str
    # -- Viewport -- nodes (0D glyphs) ------------------------------
    node_accent: str
    # -- Viewport -- origin-marker overlay --------------------------
    origin_marker_color: str
    # -- Viewport -- axis scene / grid / bbox -----------------------
    grid_major: str
    grid_minor: str
    bbox_color: str
    bbox_line_px: float
    # -- Viewport -- results colormap defaults ----------------------
    cmap_seq: str                           # sequential (unsigned fields)
    cmap_div: str                           # diverging (signed fields)
    # -- Viewport -- rendering intensity ----------------------------
    ao_intensity: Literal["none", "light", "moderate"]
    corner_triad_default: bool


# ----------------------------------------------------------------------
# Catppuccin Mocha
# ----------------------------------------------------------------------

PALETTE_CATPPUCCIN_MOCHA = Palette(
    name="catppuccin_mocha",
    base="#1e1e2e", mantle="#181825",
    surface0="#313244", surface1="#45475a", surface2="#585b70",
    text="#cdd6f4", subtext="#bac2de", overlay="#a6adc8",
    accent="#89b4fa", icon="#cdd6f4",
    success="#a6e3a1", warning="#f9e2af",
    error="#f38ba8", info="#89b4fa",
    background_mode="radial",
    bg_top="#313244", bg_bottom="#11111b",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(210, 210, 210), dim_vol=(210, 210, 210),
    hover_rgb=(255, 215, 0),
    pick_rgb=(231, 76, 60),
    hidden_rgb=(0, 0, 0),
    body_palette=(
        "#74c7ec", "#fab387", "#a6e3a1", "#cba6f7", "#f5e0dc",
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#fab387",
    grid_major="#45475a", grid_minor="#313244",
    bbox_color="#7f849c", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ----------------------------------------------------------------------
# Neutral Studio
# ----------------------------------------------------------------------

PALETTE_NEUTRAL_STUDIO = Palette(
    name="neutral_studio",
    base="#141414", mantle="#1f1f1f",
    surface0="#2a2a2a", surface1="#3a3a3a", surface2="#4a4a4a",
    text="#d0d0d0", subtext="#a0a0a0", overlay="#707070",
    accent="#7aa2d7", icon="#d0d0d0",
    success="#6ca872", warning="#d4a44a",
    error="#d47272", info="#7aa2d7",
    background_mode="radial",
    bg_top="#4a4a4a", bg_bottom="#0f0f0f",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(210, 210, 210), dim_vol=(210, 210, 210),
    hover_rgb=(255, 215, 0),
    pick_rgb=(231, 76, 60),
    hidden_rgb=(0, 0, 0),
    body_palette=(
        "#5B8DB8", "#A9A878", "#4A4A4A", "#A8C8B5", "#EAE6DE",
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#d4a44a",
    grid_major="#3a3a3a", grid_minor="#2a2a2a",
    bbox_color="#9a9a9a", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ----------------------------------------------------------------------
# Paper
# ----------------------------------------------------------------------

PALETTE_PAPER = Palette(
    name="paper",
    base="#FAFAFA", mantle="#F5F5F5",
    surface0="#E8E8E8", surface1="#D8D8D8", surface2="#C0C0C0",
    text="#202020", subtext="#3a3a3a", overlay="#666666",
    accent="#2E5C8A", icon="#202020",
    success="#2d8659", warning="#b8860b",
    error="#c1272d", info="#2E5C8A",
    background_mode="flat_corner",
    bg_top="#FAFAFA", bg_bottom="#EFEFEF",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(192, 192, 192), dim_vol=(192, 192, 192),
    hover_rgb=(224, 168, 0),
    pick_rgb=(193, 39, 45),
    hidden_rgb=(250, 250, 250),
    body_palette=(
        "#8BA8C4", "#B9B681", "#A0C893", "#2F2F30", "#E8E0C8",
    ),
    outline_color="#000000",
    outline_silhouette_px=3.0, outline_feature_px=1.8,
    mesh_edge_color="#303030",
    node_accent="#000000",
    origin_marker_color="#b8860b",
    grid_major="#d0d0d0", grid_minor="#e8e8e8",
    bbox_color="#000000", bbox_line_px=1.0,
    cmap_seq="cividis", cmap_div="BrBG",
    ao_intensity="light",
    corner_triad_default=False,
)


# ----------------------------------------------------------------------
# Catppuccin Latte
# ----------------------------------------------------------------------

PALETTE_CATPPUCCIN_LATTE = Palette(
    name="catppuccin_latte",
    base="#eff1f5", mantle="#e6e9ef",
    surface0="#ccd0da", surface1="#bcc0cc", surface2="#acb0be",
    text="#4c4f69", subtext="#5c5f77", overlay="#6c6f85",
    accent="#1e66f5", icon="#4c4f69",
    success="#40a02b", warning="#df8e1d",
    error="#d20f39", info="#1e66f5",
    background_mode="flat_corner",
    bg_top="#eff1f5", bg_bottom="#dce0e8",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(192, 192, 192), dim_vol=(192, 192, 192),
    hover_rgb=(223, 142, 29),
    pick_rgb=(210, 15, 57),
    hidden_rgb=(239, 241, 245),
    body_palette=(
        "#1e66f5", "#fe640b", "#40a02b", "#8839ef", "#dc8a78",
    ),
    outline_color="#000000",
    outline_silhouette_px=3.0, outline_feature_px=1.8,
    mesh_edge_color="#303030",
    node_accent="#000000",
    origin_marker_color="#fe640b",
    grid_major="#bcc0cc", grid_minor="#ccd0da",
    bbox_color="#000000", bbox_line_px=1.0,
    cmap_seq="cividis", cmap_div="BrBG",
    ao_intensity="light",
    corner_triad_default=False,
)


# ----------------------------------------------------------------------
# Solarized Dark
# ----------------------------------------------------------------------

PALETTE_SOLARIZED_DARK = Palette(
    name="solarized_dark",
    base="#002b36", mantle="#073642",
    surface0="#0f4a58", surface1="#586e75", surface2="#657b83",
    text="#eee8d5", subtext="#93a1a1", overlay="#839496",
    accent="#268bd2", icon="#eee8d5",
    success="#859900", warning="#b58900",
    error="#dc322f", info="#268bd2",
    background_mode="radial",
    bg_top="#073642", bg_bottom="#00212b",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(210, 210, 210), dim_vol=(210, 210, 210),
    hover_rgb=(181, 137, 0),
    pick_rgb=(220, 50, 47),
    hidden_rgb=(0, 43, 54),
    body_palette=(
        "#268bd2", "#cb4b16", "#859900", "#6c71c4", "#2aa198",
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#b58900",
    grid_major="#586e75", grid_minor="#073642",
    bbox_color="#93a1a1", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ----------------------------------------------------------------------
# Solarized Light
# ----------------------------------------------------------------------

PALETTE_SOLARIZED_LIGHT = Palette(
    name="solarized_light",
    base="#fdf6e3", mantle="#eee8d5",
    surface0="#e1dac0", surface1="#93a1a1", surface2="#839496",
    text="#073642", subtext="#586e75", overlay="#657b83",
    accent="#268bd2", icon="#073642",
    success="#859900", warning="#b58900",
    error="#dc322f", info="#268bd2",
    background_mode="flat_corner",
    bg_top="#fdf6e3", bg_bottom="#eee8d5",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(192, 192, 192), dim_vol=(192, 192, 192),
    hover_rgb=(181, 137, 0),
    pick_rgb=(220, 50, 47),
    hidden_rgb=(253, 246, 227),
    body_palette=(
        "#268bd2", "#cb4b16", "#859900", "#6c71c4", "#2aa198",
    ),
    outline_color="#000000",
    outline_silhouette_px=3.0, outline_feature_px=1.8,
    mesh_edge_color="#303030",
    node_accent="#000000",
    origin_marker_color="#cb4b16",
    grid_major="#c9c3a8", grid_minor="#e1dac0",
    bbox_color="#073642", bbox_line_px=1.0,
    cmap_seq="cividis", cmap_div="BrBG",
    ao_intensity="light",
    corner_triad_default=False,
)


# ----------------------------------------------------------------------
# Nord
# ----------------------------------------------------------------------

PALETTE_NORD = Palette(
    name="nord",
    base="#2e3440", mantle="#242933",
    surface0="#3b4252", surface1="#434c5e", surface2="#4c566a",
    text="#eceff4", subtext="#d8dee9", overlay="#a3b1c2",
    accent="#88c0d0", icon="#eceff4",
    success="#a3be8c", warning="#ebcb8b",
    error="#bf616a", info="#81a1c1",
    background_mode="radial",
    bg_top="#3b4252", bg_bottom="#1b1f27",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(210, 210, 210), dim_vol=(210, 210, 210),
    hover_rgb=(235, 203, 139),
    pick_rgb=(191, 97, 106),
    hidden_rgb=(46, 52, 64),
    body_palette=(
        "#88c0d0", "#d08770", "#a3be8c", "#b48ead", "#8fbcbb",
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#ebcb8b",
    grid_major="#434c5e", grid_minor="#3b4252",
    bbox_color="#a3b1c2", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ----------------------------------------------------------------------
# Tokyo Night
# ----------------------------------------------------------------------

PALETTE_TOKYO_NIGHT = Palette(
    name="tokyo_night",
    base="#1a1b26", mantle="#16161e",
    surface0="#24283b", surface1="#414868", surface2="#565f89",
    text="#c0caf5", subtext="#a9b1d6", overlay="#787c99",
    accent="#7aa2f7", icon="#c0caf5",
    success="#9ece6a", warning="#e0af68",
    error="#f7768e", info="#7aa2f7",
    background_mode="radial",
    bg_top="#24283b", bg_bottom="#0d0e14",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(210, 210, 210), dim_vol=(210, 210, 210),
    hover_rgb=(224, 175, 104),
    pick_rgb=(247, 118, 142),
    hidden_rgb=(26, 27, 38),
    body_palette=(
        "#7aa2f7", "#ff9e64", "#9ece6a", "#bb9af7", "#7dcfff",
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#ff9e64",
    grid_major="#414868", grid_minor="#24283b",
    bbox_color="#787c99", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ----------------------------------------------------------------------
# Gruvbox Dark
# ----------------------------------------------------------------------

PALETTE_GRUVBOX_DARK = Palette(
    name="gruvbox_dark",
    base="#282828", mantle="#1d2021",
    surface0="#3c3836", surface1="#504945", surface2="#665c54",
    text="#ebdbb2", subtext="#d5c4a1", overlay="#bdae93",
    accent="#83a598", icon="#ebdbb2",
    success="#b8bb26", warning="#fabd2f",
    error="#fb4934", info="#83a598",
    background_mode="radial",
    bg_top="#3c3836", bg_bottom="#1d2021",
    dim_pt=(0, 0, 0), dim_crv=(0, 0, 0),
    dim_srf=(210, 210, 210), dim_vol=(210, 210, 210),
    hover_rgb=(250, 189, 47),
    pick_rgb=(251, 73, 52),
    hidden_rgb=(40, 40, 40),
    body_palette=(
        "#83a598", "#fe8019", "#b8bb26", "#d3869b", "#8ec07c",
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#fe8019",
    grid_major="#504945", grid_minor="#3c3836",
    bbox_color="#a89984", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ----------------------------------------------------------------------
# High Contrast
# ----------------------------------------------------------------------

PALETTE_HIGH_CONTRAST = Palette(
    name="high_contrast",
    base="#000000", mantle="#000000",
    surface0="#1a1a1a", surface1="#333333", surface2="#4d4d4d",
    text="#ffffff", subtext="#e0e0e0", overlay="#b0b0b0",
    accent="#ffff00", icon="#ffffff",
    success="#00ff00", warning="#ffff00",
    error="#ff0000", info="#00ffff",
    background_mode="flat_corner",
    bg_top="#000000", bg_bottom="#000000",
    dim_pt=(255, 255, 255), dim_crv=(255, 255, 255),
    dim_srf=(210, 210, 210), dim_vol=(210, 210, 210),
    hover_rgb=(255, 255, 0),
    pick_rgb=(255, 0, 0),
    hidden_rgb=(0, 0, 0),
    body_palette=(
        "#ffff00", "#00ffff", "#ff00ff", "#00ff00", "#ff8800",
    ),
    outline_color="#ffffff",
    outline_silhouette_px=3.0, outline_feature_px=2.0,
    mesh_edge_color="#ffffff",
    node_accent="#ffff00",
    origin_marker_color="#ff00ff",
    grid_major="#4d4d4d", grid_minor="#1a1a1a",
    bbox_color="#ffffff", bbox_line_px=1.5,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="none",
    corner_triad_default=True,
)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

PALETTES: dict[str, Palette] = {
    "catppuccin_mocha": PALETTE_CATPPUCCIN_MOCHA,
    "catppuccin_latte": PALETTE_CATPPUCCIN_LATTE,
    "neutral_studio":   PALETTE_NEUTRAL_STUDIO,
    "paper":            PALETTE_PAPER,
    "solarized_dark":   PALETTE_SOLARIZED_DARK,
    "solarized_light":  PALETTE_SOLARIZED_LIGHT,
    "nord":             PALETTE_NORD,
    "tokyo_night":      PALETTE_TOKYO_NIGHT,
    "gruvbox_dark":     PALETTE_GRUVBOX_DARK,
    "high_contrast":    PALETTE_HIGH_CONTRAST,
}

_BUILTIN_THEME_IDS: frozenset[str] = frozenset(PALETTES.keys())

_THEME_ALIASES: dict[str, str] = {
    "dark":  "catppuccin_mocha",
    "light": "paper",
}


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
    QComboBox {{
        background-color: {p.surface0};
        color: {p.text};
        border: 1px solid {p.surface1};
        border-radius: 3px;
        padding: 2px 6px;
        font-size: 11px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.base};
        color: {p.text};
        selection-background-color: {p.surface1};
        selection-color: {p.text};
        border: 1px solid {p.surface0};
        outline: 0;
    }}
    QComboBox QAbstractItemView::item {{
        padding: 4px 8px;
        color: {p.text};
    }}
    QComboBox QAbstractItemView::item:hover {{
        background-color: {p.surface1};
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
    """Global current theme + observer list for apeGmshViewer.

    Observers are called with the new ``Palette`` whenever
    ``set_theme`` changes the current theme. Custom user-authored
    themes are loaded from ``<config>/apeGmshViewer/themes/*.json``
    at construction and merged into ``PALETTES``. Built-ins always
    take precedence.
    """

    _settings_org = "apeGmshViewer"
    _settings_app = "viewer"

    def __init__(self) -> None:
        self._load_custom_themes()
        self._current: Palette = self._load_saved() or PALETTE_CATPPUCCIN_MOCHA
        self._observers: list[Callable[[Palette], None]] = []

    # -- custom theme persistence -------------------------------------

    @classmethod
    def themes_dir(cls) -> "object":
        from pathlib import Path
        try:
            from qtpy.QtCore import QStandardPaths
            root = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppConfigLocation
            )
            if root:
                return Path(root) / "apeGmshViewer" / "themes"
        except Exception:
            pass
        return Path.home() / ".config" / "apeGmshViewer" / "themes"

    @classmethod
    def _load_custom_themes(cls) -> None:
        import json
        import logging
        from dataclasses import fields

        directory = cls.themes_dir()
        try:
            if not directory.exists():  # type: ignore[union-attr]
                return
        except Exception:
            return

        log = logging.getLogger("apeGmshViewer.theme")
        valid = {f.name for f in fields(Palette)}

        for path in sorted(directory.glob("*.json")):  # type: ignore[union-attr]
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                for k in ("dim_pt", "dim_crv", "dim_srf", "dim_vol",
                          "hover_rgb", "pick_rgb", "hidden_rgb"):
                    if k in data and isinstance(data[k], list):
                        data[k] = tuple(data[k])
                if "body_palette" in data and isinstance(data["body_palette"], list):
                    data["body_palette"] = tuple(data["body_palette"])
                kept = {k: v for k, v in data.items() if k in valid}
                if "name" not in kept:
                    log.warning("skipping theme %s: no 'name' field", path)
                    continue
                pal_name = str(kept["name"])
                if pal_name in _BUILTIN_THEME_IDS:
                    log.warning(
                        "skipping custom theme %s: name %r collides with built-in",
                        path, pal_name,
                    )
                    continue
                PALETTES[pal_name] = Palette(**kept)
            except Exception:
                log.exception("failed to load custom theme %s", path)

    @classmethod
    def save_custom_theme(cls, palette: Palette) -> "object":
        import json
        from dataclasses import asdict
        directory = cls.themes_dir()
        directory.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        if palette.name in _BUILTIN_THEME_IDS:
            raise ValueError(
                f"{palette.name!r} is a built-in theme and cannot be overwritten",
            )
        path = directory / f"{palette.name}.json"  # type: ignore[union-attr, operator]
        payload = json.dumps(asdict(palette), indent=2, sort_keys=True)
        path.write_text(payload, encoding="utf-8")
        PALETTES[palette.name] = palette
        return path

    @classmethod
    def delete_custom_theme(cls, name: str) -> bool:
        if name in _BUILTIN_THEME_IDS:
            raise ValueError(f"{name!r} is a built-in theme and cannot be deleted")
        directory = cls.themes_dir()
        path = directory / f"{name}.json"  # type: ignore[union-attr, operator]
        removed = False
        try:
            path.unlink()
            removed = True
        except FileNotFoundError:
            pass
        except Exception:
            import logging
            logging.getLogger("apeGmshViewer.theme").exception(
                "failed to delete custom theme %s", name,
            )
        PALETTES.pop(name, None)
        return removed

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
                logging.getLogger("apeGmshViewer.theme").exception(
                    "theme observer failed: %r", cb,
                )

    def subscribe(
        self, cb: Callable[[Palette], None]
    ) -> Callable[[], None]:
        self._observers.append(cb)

        def _unsub() -> None:
            try:
                self._observers.remove(cb)
            except ValueError:
                pass

        return _unsub

    # -- QSettings persistence ----------------------------------------

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
