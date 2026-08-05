"""EmptyStateHUD — first-run starter card for the 3-D viewport (R1.4).

A glass card parented to the QtInteractor's widget, floated
bottom-center (PickReadoutHUD owns top-left, ProbePaletteHUD
top-right). Shown by ResultsViewer when the viewer boots with zero
diagrams: a short title plus one-click starter buttons so the grey
mesh isn't a dead end.

Dumb widget: the caller supplies the button callbacks and labels; the
HUD only shows/hides itself (dismiss × hides it locally). Hiding on
the first DIAGRAM_ATTACHED is the viewer's job.
"""
from __future__ import annotations

from typing import Any, Callable


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class EmptyStateHUD:
    """Bottom-center starter card with one-click actions.

    Parameters
    ----------
    viewport_widget
        The widget over which the HUD floats (the QtInteractor's
        rendering widget).
    on_primary
        Fired when the primary action button is clicked.
    on_secondary
        Fired when the secondary action button is clicked.
    """

    _MARGIN = 16

    def __init__(
        self,
        viewport_widget: Any,
        *,
        on_primary: Callable[[], None],
        on_secondary: Callable[[], None],
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._viewport = viewport_widget

        widget = QtWidgets.QFrame(parent=viewport_widget)
        widget.setObjectName("EmptyStateHUD")
        widget.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._widget = widget

        lay = QtWidgets.QVBoxLayout(widget)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)

        # Title row: label + dismiss ×.
        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(8)
        self._title = QtWidgets.QLabel("No diagrams yet")
        self._title.setObjectName("EmptyStateTitle")
        title_row.addWidget(self._title)
        title_row.addStretch(1)
        from ._icon_factory import bind_button_glyph
        self._btn_dismiss = QtWidgets.QToolButton(widget)
        self._btn_dismiss.setObjectName("EmptyStateDismiss")
        bind_button_glyph(self._btn_dismiss, "close", size=16)
        self._btn_dismiss.setToolTip("Dismiss")
        self._btn_dismiss.clicked.connect(self.hide)
        title_row.addWidget(self._btn_dismiss)
        lay.addLayout(title_row)

        # Action row: primary + secondary starter buttons. Labels are
        # caller-set via set_primary / set_secondary (an empty label
        # hides the button).
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self._btn_primary = QtWidgets.QPushButton(widget)
        self._btn_primary.setObjectName("EmptyStateAction")
        self._btn_primary.clicked.connect(on_primary)
        # Hidden until a setter provides a label — the "empty label
        # hides it" contract must hold before the first set_* call too,
        # or show() without setters paints a blank button.
        self._btn_primary.setVisible(False)
        btn_row.addWidget(self._btn_primary)
        self._btn_secondary = QtWidgets.QPushButton(widget)
        self._btn_secondary.setObjectName("EmptyStateAction")
        self._btn_secondary.clicked.connect(on_secondary)
        self._btn_secondary.setVisible(False)
        btn_row.addWidget(self._btn_secondary)
        lay.addLayout(btn_row)

        # Reposition on viewport resize via event filter.
        from ._viewport_hud import _ResizeFilter
        self._filter = _ResizeFilter(self.reposition)
        viewport_widget.installEventFilter(self._filter)

        # Hidden until the viewer decides the boot is diagram-less.
        widget.hide()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def show(self) -> None:
        self._widget.show()
        self._widget.raise_()
        self.reposition()

    def hide(self) -> None:
        self._widget.hide()

    def set_primary(self, label: str, enabled: bool = True) -> None:
        """Set the primary button's label; empty label hides it."""
        self._btn_primary.setText(label)
        self._btn_primary.setEnabled(bool(enabled))
        self._btn_primary.setVisible(bool(label))
        self._relayout()

    def set_secondary(self, label: str, enabled: bool = True) -> None:
        """Set the secondary button's label; empty label hides it."""
        self._btn_secondary.setText(label)
        self._btn_secondary.setEnabled(bool(enabled))
        self._btn_secondary.setVisible(bool(label))
        self._relayout()

    def _relayout(self) -> None:
        """Re-fit + re-center after a label change on a visible card.

        Without this a setter called after :meth:`show` leaves the
        frame at its previous size and a longer label clips."""
        if self._widget.isVisible():
            self.reposition()

    def reposition(self) -> None:
        """Move the HUD to the viewport's bottom-center."""
        try:
            vw = self._viewport.width()
            vh = self._viewport.height()
        except Exception:
            return
        self._widget.adjustSize()
        x = (vw - self._widget.width()) // 2
        y = vh - self._widget.height() - self._MARGIN
        self._widget.move(max(0, x), max(0, y))
        self._widget.raise_()
