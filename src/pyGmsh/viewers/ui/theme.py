"""
Catppuccin Mocha theme — shared stylesheet and widget helpers.

Provides a single source of truth for the viewer's visual theme.
All viewers (model, mesh, results) use these constants and helpers
for a consistent look.

Usage::

    from pyGmsh.viewers.ui.theme import STYLESHEET, styled_group
    window.setStyleSheet(STYLESHEET)
    group = styled_group("Display")
"""
from __future__ import annotations


# ======================================================================
# Catppuccin Mocha color palette
# ======================================================================

BASE      = "#1e1e2e"   # main background
MANTLE    = "#181825"   # bars, headers
SURFACE0  = "#313244"   # borders, input bg
SURFACE1  = "#45475a"   # hover
SURFACE2  = "#585b70"   # pressed
TEXT      = "#cdd6f4"   # primary text
SUBTEXT   = "#bac2de"   # secondary text (labels)
OVERLAY   = "#a6adc8"   # muted text
BLUE      = "#89b4fa"   # accent (slider handles, links)
GREEN     = "#a6e3a1"   # active group
YELLOW    = "#f9e2af"   # staged items
PEACH     = "#fab387"   # cell data
RED       = "#f38ba8"   # errors, picked nodes


# ======================================================================
# Global stylesheet
# ======================================================================

STYLESHEET = f"""
    QMainWindow {{
        background-color: {BASE};
    }}
    QMenuBar {{
        background-color: {MANTLE};
        color: {TEXT};
        border-bottom: 1px solid {SURFACE0};
    }}
    QMenuBar::item:selected {{
        background-color: {SURFACE1};
    }}
    QMenu {{
        background-color: {BASE};
        color: {TEXT};
        border: 1px solid {SURFACE0};
    }}
    QMenu::item:selected {{
        background-color: {SURFACE1};
    }}
    QToolBar {{
        background-color: {MANTLE};
        border: 1px solid {SURFACE0};
        spacing: 2px;
        padding: 2px;
    }}
    QToolBar QToolButton {{
        background-color: {SURFACE0};
        color: {TEXT};
        border: 1px solid {SURFACE1};
        border-radius: 3px;
        padding: 4px 8px;
        font-size: 11px;
    }}
    QToolBar QToolButton:hover {{
        background-color: {SURFACE1};
    }}
    QToolBar QToolButton:pressed {{
        background-color: {SURFACE2};
    }}
    QToolBar QToolButton:checked {{
        background-color: rgba(100, 180, 255, 60);
        border: 1px solid rgba(100, 180, 255, 120);
    }}
    QStatusBar {{
        background-color: {MANTLE};
        color: {OVERLAY};
        border-top: 1px solid {SURFACE0};
        font-size: 11px;
    }}
    QSplitter::handle {{
        background-color: {SURFACE0};
        width: 2px;
        height: 2px;
    }}
    QTabWidget::pane {{
        border: 1px solid {SURFACE0};
        background: {BASE};
    }}
    QTabBar::tab {{
        background: {MANTLE};
        color: {OVERLAY};
        padding: 6px 12px;
        border: 1px solid {SURFACE0};
        border-bottom: none;
    }}
    QTabBar::tab:selected {{
        background: {BASE};
        color: {TEXT};
    }}
    QTabBar::tab:hover {{
        background: {SURFACE1};
    }}
    QDockWidget {{
        color: {TEXT};
    }}
    QDockWidget::title {{
        background: {MANTLE};
        padding: 4px;
        border: 1px solid {SURFACE0};
    }}
    /* ── Form widgets ────────────────────────────────────── */
    QComboBox {{
        background-color: {SURFACE0};
        color: {TEXT};
        border: 1px solid {SURFACE1};
        border-radius: 3px;
        padding: 2px 6px;
        font-size: 11px;
    }}
    QSpinBox, QDoubleSpinBox {{
        background-color: {SURFACE0};
        color: {TEXT};
        border: 1px solid {SURFACE1};
        border-radius: 3px;
        padding: 2px 4px;
    }}
    QCheckBox {{
        color: {SUBTEXT};
        font-size: 11px;
        spacing: 6px;
    }}
    QPushButton {{
        background-color: {SURFACE0};
        color: {TEXT};
        border: 1px solid {SURFACE1};
        border-radius: 3px;
        padding: 3px 8px;
        font-size: 11px;
    }}
    QPushButton:hover {{
        background-color: {SURFACE1};
    }}
    QPushButton:pressed {{
        background-color: {SURFACE2};
    }}
    QLabel {{
        color: {SUBTEXT};
    }}
    QSlider::groove:horizontal {{
        background: {SURFACE0};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {BLUE};
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }}
    QTreeWidget {{
        background-color: {BASE};
        color: {TEXT};
        border: 1px solid {SURFACE0};
        font-size: 12px;
    }}
    QTreeWidget::item:selected {{
        background-color: {SURFACE1};
    }}
    QTreeWidget::item:hover {{
        background-color: {SURFACE0};
    }}
    QHeaderView::section {{
        background-color: {MANTLE};
        color: {OVERLAY};
        border: 1px solid {SURFACE0};
        padding: 4px;
        font-weight: bold;
    }}
    QTextEdit {{
        background-color: {BASE};
        color: {TEXT};
        border: 1px solid {SURFACE0};
    }}
    QGroupBox {{
        color: {TEXT};
        border: 1px solid {SURFACE0};
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
        background-color: {BASE};
        color: {TEXT};
    }}
    QLineEdit {{
        background-color: {SURFACE0};
        color: {TEXT};
        border: 1px solid {SURFACE1};
        border-radius: 3px;
        padding: 4px 6px;
    }}
    QMessageBox {{
        background-color: {BASE};
        color: {TEXT};
    }}
"""


# ======================================================================
# Helper
# ======================================================================

def styled_group(title: str):
    """Create a QGroupBox with Catppuccin theme (inherits from STYLESHEET)."""
    from qtpy.QtWidgets import QGroupBox
    return QGroupBox(title)
