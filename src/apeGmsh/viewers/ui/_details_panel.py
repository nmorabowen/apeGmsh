"""DetailsPanel — contextual right-rail panel under the plot pane (B++ §4.4).

Hidden when nothing is selected, or when a tree group is selected.
Renders inside the same column as the plot pane (right rail) and
shares its width.

For B2 the only contextual content is per-diagram styling — the
existing :class:`DiagramSettingsTab` widget is hosted here directly.
Stage / Probe / Plot details fill in as those tree groups become
first-class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..diagrams._base import Diagram

if TYPE_CHECKING:
    from ._diagram_settings_tab import DiagramSettingsTab


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class DetailsPanel:
    """Selection-driven details panel.

    Parameters
    ----------
    settings_tab
        The existing :class:`DiagramSettingsTab` whose widget is
        re-hosted as the body for diagram selections.
    """

    _MAX_HEIGHT = 220   # spec §4.4

    def __init__(self, settings_tab: "DiagramSettingsTab") -> None:
        QtWidgets, QtCore = _qt()
        self._settings_tab = settings_tab

        widget = QtWidgets.QWidget()
        widget.setObjectName("DetailsPanel")
        widget.setMaximumHeight(self._MAX_HEIGHT)
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────
        header = QtWidgets.QFrame()
        header.setObjectName("DetailsHeader")
        header.setFixedHeight(24)
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(10, 0, 10, 0)
        header_lay.setSpacing(6)

        title = QtWidgets.QLabel("DETAILS")
        title.setObjectName("DetailsHeaderLabel")
        header_lay.addWidget(title)
        header_lay.addStretch(1)

        meta = QtWidgets.QLabel("")
        meta.setObjectName("DetailsHeaderMeta")
        header_lay.addWidget(meta)
        self._meta_label = meta

        outer.addWidget(header)

        # ── Body — re-host the DiagramSettingsTab widget ───────────
        # We reparent settings_tab.widget into our outer layout. The
        # DiagramSettingsTab subscribes to director.on_diagrams_changed
        # itself; nothing needs to change about its observer wiring.
        body_holder = QtWidgets.QWidget()
        body_lay = QtWidgets.QVBoxLayout(body_holder)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.addWidget(settings_tab.widget)
        outer.addWidget(body_holder, stretch=1)

        widget.setStyleSheet(
            "#DetailsPanel { "
            "background: rgba(255, 255, 255, 0.03); "
            "border-top: 1px solid rgba(255, 255, 255, 0.08); "
            "} "
            "#DetailsHeader { "
            "background: rgba(255, 255, 255, 0.04); "
            "border-bottom: 1px solid rgba(255, 255, 255, 0.06); "
            "} "
            "#DetailsHeaderLabel { "
            "color: rgba(255, 255, 255, 0.6); "
            "font-size: 10px; letter-spacing: 1px; "
            "} "
            "#DetailsHeaderMeta { "
            "color: rgba(255, 255, 255, 0.5); "
            "font-family: 'JetBrains Mono', ui-monospace, monospace; "
            "font-size: 10px; "
            "}"
        )

        self._widget = widget
        self.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def show_diagram(self, diagram: Diagram) -> None:
        """Display the per-diagram settings for ``diagram``."""
        self._settings_tab.set_selected(diagram)
        self._meta_label.setText(diagram.display_label())
        self._widget.setVisible(True)

    def clear(self) -> None:
        """Hide the panel — no selection, or non-leaf selection."""
        self._settings_tab.set_selected(None)
        self._meta_label.setText("")
        self._widget.setVisible(False)
