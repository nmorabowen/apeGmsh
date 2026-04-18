"""
Viewer theme — palette dataclass + stylesheet factory.

Two palettes ship today (dark = Catppuccin Mocha, light = white +
greyscale). ``ThemeManager`` is a singleton observable that the viewer
window subscribes to; swapping the current palette re-renders the
stylesheet and fires observers so VTK content can re-push too.

Usage::

    from apeGmsh.viewers.ui.theme import THEME, build_stylesheet
    window.setStyleSheet(build_stylesheet(THEME.current))
    unsubscribe = THEME.subscribe(lambda p: refresh(p))
    # later...
    THEME.set_theme("light")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ======================================================================
# Palette
# ======================================================================

@dataclass(frozen=True)
class Palette:
    """Chrome + semantic roles for a viewer theme."""

    name: str                               # "dark" | "light"
    # Surfaces
    base: str                               # main window background
    mantle: str                             # bars, headers
    surface0: str                           # borders, input bg
    surface1: str                           # hover
    surface2: str                           # pressed
    # Text
    text: str                               # primary text
    subtext: str                            # secondary text (labels)
    overlay: str                            # muted text (empty-label)
    # Accent
    accent: str                             # slider handles, focus borders
    # Viewport gradient
    bg_top: str
    bg_bottom: str
    # Icon color (viewer_window toolbar)
    icon: str
    # Semantic (contrast-adjusted per theme)
    success: str                            # active group
    warning: str                            # staged items, warning banner
    error: str                              # errors, picked nodes
    info: str                               # inactive group, cell data
    # VTK content colors (RGB 0..255 tuples)
    dim_pt: tuple[int, int, int]
    dim_crv: tuple[int, int, int]
    dim_srf: tuple[int, int, int]
    dim_vol: tuple[int, int, int]


# ──────────────────────────────────────────────────────────────────────
# Dark (Catppuccin Mocha, existing)
# ──────────────────────────────────────────────────────────────────────

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
    dim_pt=(232, 213, 183),     # warm white
    dim_crv=(170, 170, 170),
    dim_srf=(91, 141, 184),
    dim_vol=(90, 110, 130),
)


# ──────────────────────────────────────────────────────────────────────
# Light (white + greyscale, user-requested)
# ──────────────────────────────────────────────────────────────────────

PALETTE_LIGHT = Palette(
    name="light",
    base="#ffffff", mantle="#f4f4f4",
    surface0="#e0e0e0", surface1="#d0d0d0", surface2="#b8b8b8",
    text="#1a1a1a", subtext="#3a3a3a", overlay="#666666",
    accent="#333333",                       # neutral dark grey
    bg_top="#fafafa", bg_bottom="#e8e8e8",
    icon="#1a1a1a",
    success="#2d8659",                      # darker green
    warning="#b8860b",                      # dark goldenrod
    error="#c1272d",                        # darker red
    info="#1f5fa8",                         # darker blue
    dim_pt=(30, 30, 30),
    dim_crv=(80, 80, 80),
    dim_srf=(70, 110, 150),
    dim_vol=(50, 70, 90),
)


PALETTES: dict[str, Palette] = {
    "dark": PALETTE_DARK,
    "light": PALETTE_LIGHT,
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
            name = s.value("theme", "dark")
            return PALETTES.get(str(name).lower())
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
