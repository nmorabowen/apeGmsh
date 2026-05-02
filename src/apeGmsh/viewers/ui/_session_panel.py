"""SessionPanel — viewer-level settings dock for the post-solve viewer.

Hosts session-scoped controls that don't belong inside the model
(Outline) or per-diagram (Details) flows. Phase 1 ships the theme
picker; later additions land here as new sections (density, layout
reset, file actions, …).
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class SessionPanel:
    """Right-rail dock for viewer-level settings (theme, …).

    Subscribes to :data:`THEME` so the displayed theme stays in sync
    when something outside the panel switches it (e.g. an external
    keyboard shortcut, or another viewer instance updating settings).
    """

    def __init__(self) -> None:
        QtWidgets, QtCore = _qt()
        from .theme import THEME, PALETTES

        self._theme = THEME
        self._palettes = PALETTES

        widget = QtWidgets.QWidget()
        widget.setObjectName("SessionPanel")
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Theme picker ───────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self._theme_combo = QtWidgets.QComboBox()
        for name in sorted(PALETTES.keys()):
            self._theme_combo.addItem(name, name)
        # Reflect the current theme without firing the change signal.
        idx = self._theme_combo.findData(THEME.current.name)
        if idx >= 0:
            self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentIndex(idx)
            self._theme_combo.blockSignals(False)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_chosen)
        form.addRow("Theme:", self._theme_combo)

        outer.addLayout(form)

        # Trailing stretch so controls pack at the top of the dock —
        # any spare vertical space ends up below, not between widgets.
        outer.addStretch(1)

        # Track external theme changes (e.g. another consumer calling
        # THEME.set_theme) so the combo stays accurate.
        self._unsub_theme = THEME.subscribe(self._on_theme_changed_externally)

        self._widget = widget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def close(self) -> None:
        """Detach observers — call when the host window closes."""
        try:
            self._unsub_theme()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_theme_chosen(self, _idx: int) -> None:
        name = self._theme_combo.currentData()
        if name is None:
            return
        self._theme.set_theme(name)

    def _on_theme_changed_externally(self, palette) -> None:
        idx = self._theme_combo.findData(palette.name)
        if idx < 0 or idx == self._theme_combo.currentIndex():
            return
        self._theme_combo.blockSignals(True)
        self._theme_combo.setCurrentIndex(idx)
        self._theme_combo.blockSignals(False)
