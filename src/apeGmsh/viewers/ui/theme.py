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
    # ── Viewport — origin-marker overlay (reference points) ─────────
    origin_marker_color: str
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
    # ── Viewport — ResultsViewer substrate (FEM mesh) ───────────────
    # Defaults match the legacy hardcoded values in results_viewer.py,
    # so every existing palette continues to render identically until
    # individual themes choose to override.
    substrate_color: str = "#bfbfbf"        # surface fill of the FEM mesh
    substrate_edge_color: str = "#444444"   # element-edge line color


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
    # Origin marker — Catppuccin Peach (gold-amber, stands out on gray)
    origin_marker_color="#fab387",
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
    # Origin marker — warm amber (contrasts with steel-blue body palette)
    origin_marker_color="#d4a44a",
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
    # Origin marker — deep amber (visible on white, not garish)
    origin_marker_color="#b8860b",
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


# ──────────────────────────────────────────────────────────────────────
# Catppuccin Latte (light counterpart to Mocha — warm off-white)
# ──────────────────────────────────────────────────────────────────────

PALETTE_CATPPUCCIN_LATTE = Palette(
    name="catppuccin_latte",
    # Chrome — Latte spec
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
    hover_rgb=(223, 142, 29),       # Latte Yellow
    pick_rgb=(210, 15, 57),         # Latte Red
    hidden_rgb=(239, 241, 245),     # base
    body_palette=(
        "#1e66f5",  # Blue
        "#fe640b",  # Peach
        "#40a02b",  # Green
        "#8839ef",  # Mauve
        "#dc8a78",  # Rosewater
    ),
    outline_color="#000000",
    outline_silhouette_px=3.0, outline_feature_px=1.8,
    mesh_edge_color="#303030",
    node_accent="#000000",
    origin_marker_color="#fe640b",  # Peach
    grid_major="#bcc0cc", grid_minor="#ccd0da",
    bbox_color="#000000", bbox_line_px=1.0,
    cmap_seq="cividis", cmap_div="BrBG",
    ao_intensity="light",
    corner_triad_default=False,
)


# ──────────────────────────────────────────────────────────────────────
# Solarized Dark
# ──────────────────────────────────────────────────────────────────────

PALETTE_SOLARIZED_DARK = Palette(
    name="solarized_dark",
    # Chrome — Solarized dark (base03..base3)
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
    hover_rgb=(181, 137, 0),        # Solarized yellow
    pick_rgb=(220, 50, 47),         # Solarized red
    hidden_rgb=(0, 43, 54),         # base03
    body_palette=(
        "#268bd2",  # blue
        "#cb4b16",  # orange
        "#859900",  # green
        "#6c71c4",  # violet
        "#2aa198",  # cyan
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#b58900",  # yellow
    grid_major="#586e75", grid_minor="#073642",
    bbox_color="#93a1a1", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ──────────────────────────────────────────────────────────────────────
# Solarized Light
# ──────────────────────────────────────────────────────────────────────

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
    origin_marker_color="#cb4b16",  # orange
    grid_major="#c9c3a8", grid_minor="#e1dac0",
    bbox_color="#073642", bbox_line_px=1.0,
    cmap_seq="cividis", cmap_div="BrBG",
    ao_intensity="light",
    corner_triad_default=False,
)


# ──────────────────────────────────────────────────────────────────────
# Nord (cool Nordic dark)
# ──────────────────────────────────────────────────────────────────────

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
    hover_rgb=(235, 203, 139),      # Nord13 yellow
    pick_rgb=(191, 97, 106),        # Nord11 red
    hidden_rgb=(46, 52, 64),        # base
    body_palette=(
        "#88c0d0",  # frost cyan
        "#d08770",  # aurora orange
        "#a3be8c",  # aurora green
        "#b48ead",  # aurora purple
        "#8fbcbb",  # frost teal
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#ebcb8b",  # yellow
    grid_major="#434c5e", grid_minor="#3b4252",
    bbox_color="#a3b1c2", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ──────────────────────────────────────────────────────────────────────
# Tokyo Night (modern dark, purple/blue accents)
# ──────────────────────────────────────────────────────────────────────

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
    hover_rgb=(224, 175, 104),      # Tokyo yellow
    pick_rgb=(247, 118, 142),       # Tokyo red
    hidden_rgb=(26, 27, 38),
    body_palette=(
        "#7aa2f7",  # blue
        "#ff9e64",  # orange
        "#9ece6a",  # green
        "#bb9af7",  # purple
        "#7dcfff",  # cyan
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#ff9e64",  # orange
    grid_major="#414868", grid_minor="#24283b",
    bbox_color="#787c99", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ──────────────────────────────────────────────────────────────────────
# Gruvbox Dark (earthy retro)
# ──────────────────────────────────────────────────────────────────────

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
    hover_rgb=(250, 189, 47),       # Gruvbox yellow
    pick_rgb=(251, 73, 52),         # Gruvbox red
    hidden_rgb=(40, 40, 40),
    body_palette=(
        "#83a598",  # aqua/blue
        "#fe8019",  # orange
        "#b8bb26",  # green
        "#d3869b",  # pink
        "#8ec07c",  # teal
    ),
    outline_color="#000000",
    outline_silhouette_px=2.5, outline_feature_px=1.5,
    mesh_edge_color="#000000",
    node_accent="#000000",
    origin_marker_color="#fe8019",  # orange
    grid_major="#504945", grid_minor="#3c3836",
    bbox_color="#a89984", bbox_line_px=1.0,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="moderate",
    corner_triad_default=True,
)


# ──────────────────────────────────────────────────────────────────────
# High Contrast (accessibility — pure black/white/yellow)
# ──────────────────────────────────────────────────────────────────────

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
    dim_pt=(255, 255, 255),         # white — max contrast on black
    dim_crv=(255, 255, 255),        # white
    dim_srf=(210, 210, 210),        # light gray
    dim_vol=(210, 210, 210),
    hover_rgb=(255, 255, 0),        # pure yellow
    pick_rgb=(255, 0, 0),           # pure red
    hidden_rgb=(0, 0, 0),
    body_palette=(
        "#ffff00", "#00ffff", "#ff00ff", "#00ff00", "#ff8800",
    ),
    outline_color="#ffffff",        # white outlines on black bg
    outline_silhouette_px=3.0, outline_feature_px=2.0,
    mesh_edge_color="#ffffff",
    node_accent="#ffff00",          # yellow nodes — visible on black
    origin_marker_color="#ff00ff",  # magenta — unmistakable
    grid_major="#4d4d4d", grid_minor="#1a1a1a",
    bbox_color="#ffffff", bbox_line_px=1.5,
    cmap_seq="viridis", cmap_div="coolwarm",
    ao_intensity="none",
    corner_triad_default=True,
)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

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

# Frozen snapshot of built-in theme ids — used to protect them from
# custom-theme overrides / deletion.
_BUILTIN_THEME_IDS: frozenset[str] = frozenset(PALETTES.keys())


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


def _hex_to_rgb(hex_str: str) -> "tuple[int, int, int]":
    """Parse ``#rrggbb`` (or ``rrggbb``) into an ``(r, g, b)`` triple."""
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgba(hex_str: str, alpha: float) -> str:
    """Format a CSS ``rgba(r, g, b, a)`` from ``#rrggbb`` + alpha."""
    r, g, b = _hex_to_rgb(hex_str)
    return f"rgba({r}, {g}, {b}, {alpha})"


def build_stylesheet(p: Palette, density: object = None) -> str:
    """Render the viewer QSS for a given palette + density.

    ``density`` is a :class:`density.DensityTokens` instance — left as
    ``object`` here to avoid an import cycle. Pass ``None`` to use a
    sensible compact default; the live viewer always passes the
    current :data:`density.DENSITY` value.
    """
    # Density-driven sizing. Inlining a tiny default avoids importing
    # the density module at module load time (theme.py is imported
    # very early via the static ``STYLESHEET`` constant).
    if density is not None:
        d_row_h = int(getattr(density, "row_h", 22))
        d_pad_x = int(getattr(density, "pad_x", 8))
        d_pad_y = int(getattr(density, "pad_y", 4))
        d_fs_body = float(getattr(density, "fs_body", 11.5))
        d_fs_head = float(getattr(density, "fs_head", 11.0))
    else:
        d_row_h, d_pad_x, d_pad_y = 22, 8, 4
        d_fs_body, d_fs_head = 11.5, 11.0

    # LayoutMetrics — corners + dock-separator width. Imported here
    # rather than at module top because theme.py is loaded very early
    # and we want to avoid widening that import surface.
    from ._layout_metrics import LAYOUT
    rad = LAYOUT.corner_radius
    rad_sm = LAYOUT.corner_radius_small
    rad_lg = LAYOUT.corner_radius_large
    sep_w = LAYOUT.dock_separator_width
    split_w = LAYOUT.splitter_handle_width
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
        width: {split_w}px;
        height: {split_w}px;
    }}
    /* Dock separator — the gap (and drag-handle) between adjacent
       QDockWidgets, and between docks and the central widget. Width
       comes from LayoutMetrics.dock_separator_width. */
    QMainWindow::separator {{
        background-color: {p.surface0};
        width: {sep_w}px;
        height: {sep_w}px;
    }}
    QMainWindow::separator:hover {{
        background-color: {p.accent};
    }}
    QTabWidget::pane {{
        border: 1px solid {p.surface0};
        background: {p.base};
    }}
    /* Tab bar background — the empty strip beyond the last tab.
       Without this, QTabBar inherits Qt's default (white on Windows),
       which is the white strip visible to the right of tabified
       docks like Plots/Details. */
    QTabBar {{
        background: {p.base};
        border: none;
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
        titlebar-close-icon: url(none);
        titlebar-normal-icon: url(none);
    }}
    /* Dock title bar — must be obviously visible since this is the
       drag handle for moving / floating / tabifying the dock.
       Stronger contrast + explicit min-height guarantees a grabbable
       strip even when the surrounding chrome is the same hue family. */
    QDockWidget::title {{
        background: {p.surface1};
        color: {p.text};
        padding: 6px 10px;
        border-top: 1px solid {p.accent};
        border-bottom: 1px solid {p.surface0};
        border-top-left-radius: {rad}px;
        border-top-right-radius: {rad}px;
        text-align: left;
        font-weight: bold;
        font-size: 11px;
    }}
    QDockWidget::close-button, QDockWidget::float-button {{
        background: {p.surface0};
        border: 1px solid {p.surface1};
        border-radius: {rad_sm}px;
        padding: 1px;
    }}
    QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
        background: {p.accent};
    }}
    /* Dock interior — without these rules, the QScrollArea wrapping
       each panel falls through to Qt's default (white on Windows),
       producing visible color seams against the themed chrome. */
    QScrollArea {{
        background-color: {p.base};
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: {p.base};
    }}
    QStackedWidget {{
        background-color: {p.base};
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
    /* Popup list (separate widget — inherits OS defaults without this) */
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
        border-radius: {rad}px;
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

    /* ── Results viewer chrome ─────────────────────────────── */
    /* DiagramSettingsTab empty-state hint — italic muted text shown
       when no diagram is selected. Themed via overlay color. */
    QLabel#DiagramSettingsEmptyHint {{
        color: {p.overlay};
        font-style: italic;
    }}

    /* Outline tree (left rail) */
    QFrame#OutlineHeader {{
        background-color: {p.mantle};
        border-bottom: 1px solid {p.surface0};
    }}
    QLabel#OutlineHeaderLabel {{
        color: {p.overlay};
        font-size: 10px;
        letter-spacing: 1px;
    }}
    QPushButton#OutlineInsertButton {{
        background-color: transparent;
        border: 1px solid transparent;
        color: {p.text};
        padding: 2px 6px;
        font-size: 11px;
    }}
    QPushButton#OutlineInsertButton:hover {{
        background-color: {p.surface0};
        border-color: {p.surface1};
    }}
    QTreeWidget#OutlineTreeWidget {{
        border: none;
    }}

    /* Plot pane (right rail, top) */
    QFrame#PlotPaneHeader, QFrame#PlotPaneNewPlot {{
        background-color: {p.mantle};
        border-bottom: 1px solid {p.surface0};
    }}
    QLabel#PlotPaneHeaderLabel {{
        color: {p.overlay};
        font-size: 10px;
        letter-spacing: 1px;
    }}
    QLabel#PlotPaneEmpty {{
        color: {p.overlay};
        font-size: 11px;
    }}
    QFrame#PlotPaneTabRow {{
        border-left: 2px solid transparent;
        background: transparent;
    }}
    QFrame#PlotPaneTabRow[active="true"] {{
        border-left: 2px solid {p.accent};
        background-color: {p.surface0};
    }}
    QLabel#PlotPaneTabDot {{
        color: {p.overlay};
        font-size: 10px;
    }}
    QFrame#PlotPaneTabRow[active="true"] QLabel#PlotPaneTabDot {{
        color: {p.accent};
    }}
    QLabel#PlotPaneTabLabel {{
        color: {p.text};
        font-size: 11px;
    }}
    QToolButton#PlotPaneTabClose {{
        color: {p.overlay};
        border: none;
    }}
    QToolButton#PlotPaneTabClose:hover {{
        color: {p.text};
    }}

    /* Details panel (right rail, bottom) */
    QWidget#DetailsPanel {{
        background-color: {p.mantle};
        border-top: 1px solid {p.surface0};
    }}
    QFrame#DetailsHeader {{
        background-color: {p.base};
        border-bottom: 1px solid {p.surface0};
    }}
    QLabel#DetailsHeaderLabel {{
        color: {p.overlay};
        font-size: 10px;
        letter-spacing: 1px;
    }}
    QLabel#DetailsHeaderMeta {{
        color: {p.overlay};
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 10px;
    }}

    /* Probe palette HUD (viewport overlay) */
    QFrame#ProbeHUD {{
        background-color: {_rgba(p.mantle, 0.92)};
        border: 1px solid {p.surface0};
        border-radius: 6px;
    }}
    QFrame#ProbeHUD QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 3px;
        color: {p.text};
        padding: 3px;
        font-size: 14px;
    }}
    QFrame#ProbeHUD QToolButton:hover {{
        background-color: {p.surface0};
        border-color: {p.surface1};
    }}
    QFrame#ProbeHUD QToolButton[active="true"] {{
        background-color: {_rgba(p.accent, 0.18)};
        border: 1px solid {p.accent};
        color: {p.text};
    }}
    QFrame#ProbeHUDSep {{
        color: {p.surface0};
    }}

    /* Pick readout HUD (top-left viewport overlay) */
    QFrame#PickReadoutHUD {{
        background-color: {_rgba(p.mantle, 0.92)};
        border: 1px solid {p.surface0};
        border-radius: 6px;
    }}
    QLabel#PickReadoutHeader {{
        color: {p.text};
        font-weight: 600;
        font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
        font-size: 11px;
    }}
    QLabel#PickReadoutCoords {{
        color: {p.subtext};
        font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
        font-size: 10px;
    }}
    QLabel#PickReadoutValues {{
        color: {p.text};
        font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
        font-size: 10px;
    }}
    QLabel#PickReadoutHint {{
        color: {p.overlay};
        font-size: 9px;
        margin-top: 2px;
    }}

    /* Inline kind picker (popover under outline header) */
    QFrame#OutlineKindPicker {{
        background-color: {p.surface0};
        border-bottom: 1px solid {p.mantle};
    }}
    QToolButton#OutlineKindBtn {{
        background-color: {p.mantle};
        border: 1px solid {p.surface0};
        border-radius: 4px;
        color: {p.text};
        font-size: 10px;
        padding: 4px 6px;
    }}
    QToolButton#OutlineKindBtn:hover {{
        background-color: {p.surface1};
        border-color: {p.accent};
    }}
    QToolButton#OutlineKindBtn:pressed {{
        background-color: {p.surface2};
    }}

    /* Density-driven sizing (B++ §5 RV_DENSITY) */
    QTreeWidget {{
        font-size: {d_fs_body}px;
    }}
    QTreeWidget::item {{
        padding-top: {d_pad_y}px;
        padding-bottom: {d_pad_y}px;
        padding-left: {d_pad_x}px;
        min-height: {d_row_h}px;
    }}
    QFrame#OutlineHeader QLabel {{
        font-size: {d_fs_head}px;
    }}
    QFrame#PlotPaneHeader QLabel {{
        font-size: {d_fs_head}px;
    }}
    QFrame#DetailsHeader QLabel {{
        font-size: {d_fs_head}px;
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

    Custom user-authored themes are loaded from the per-user theme
    directory (``<config>/apeGmsh/themes/*.json``) at construction and
    merged into ``PALETTES``. Built-ins always take precedence — a custom
    theme with a built-in name is ignored with a warning.
    """

    _settings_org = "apeGmsh"
    _settings_app = "viewer"

    def __init__(self) -> None:
        self._load_custom_themes()
        self._current: Palette = self._load_saved() or PALETTE_DARK
        self._observers: list[Callable[[Palette], None]] = []

    # ── custom theme persistence ─────────────────────────────────────

    @classmethod
    def themes_dir(cls) -> "object":
        """Platform-appropriate directory holding custom-theme JSON files."""
        from pathlib import Path
        try:
            from qtpy.QtCore import QStandardPaths
            root = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppConfigLocation
            )
            if root:
                return Path(root) / "apeGmsh" / "themes"
        except Exception:
            pass
        return Path.home() / ".config" / "apeGmsh" / "themes"

    @classmethod
    def _load_custom_themes(cls) -> None:
        """Scan the themes dir and merge any user palettes into ``PALETTES``."""
        import json
        import logging
        from dataclasses import fields

        directory = cls.themes_dir()
        try:
            if not directory.exists():  # type: ignore[union-attr]
                return
        except Exception:
            return

        log = logging.getLogger("apeGmsh.viewer.theme")
        valid = {f.name for f in fields(Palette)}

        for path in sorted(directory.glob("*.json")):  # type: ignore[union-attr]
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                # Coerce list-of-3 ints back to tuples (JSON has no tuple).
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
        """Persist ``palette`` to the themes dir. Returns the file path."""
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
        """Remove a custom theme from disk and ``PALETTES``. Returns True if deleted."""
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
            logging.getLogger("apeGmsh.viewer.theme").exception(
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
