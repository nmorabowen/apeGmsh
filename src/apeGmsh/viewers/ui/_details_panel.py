"""DetailsPanel — contextual placeholder dock.

After the dock split, the layer stack lives in the dedicated Diagram
dock and the geometry editor lives in the dedicated Geometry dock.
The Details dock is reserved for canvas-driven contextual content
(contour scalebar edits, picked node / element readouts, …) which
will populate it as those interactions land.

For now the panel renders an idle hint so the dock isn't visually
empty. The ``settings_tab`` / ``geometry_panel`` constructor args are
retained for backwards-compatible call sites and ignored.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._layout_metrics import LAYOUT

if TYPE_CHECKING:
    from ._diagram_settings_tab import DiagramSettingsTab
    from ._geometry_settings_panel import GeometrySettingsPanel


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class DetailsPanel:
    """Placeholder panel for the Details dock."""

    def __init__(
        self,
        settings_tab: "DiagramSettingsTab | None" = None,
        geometry_panel: "GeometrySettingsPanel | None" = None,
    ) -> None:
        QtWidgets, _ = _qt()
        widget = QtWidgets.QWidget()
        widget.setObjectName("DetailsPanel")
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QtWidgets.QFrame()
        header.setObjectName("DetailsHeader")
        header.setFixedHeight(LAYOUT.details_header_height)
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(10, 0, 10, 0)
        header_lay.setSpacing(6)
        title = QtWidgets.QLabel("DETAILS")
        title.setObjectName("DetailsHeaderLabel")
        header_lay.addWidget(title)
        header_lay.addStretch(1)
        outer.addWidget(header)

        hint = QtWidgets.QLabel(
            "Click an item on the canvas to inspect it here.",
        )
        hint.setObjectName("DetailsHint")
        hint.setAlignment(_qt()[1].Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        outer.addWidget(hint, stretch=1)

        self._widget = widget

    @property
    def widget(self) -> Any:
        return self._widget
