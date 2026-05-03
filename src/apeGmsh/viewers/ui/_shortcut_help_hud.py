"""ShortcutHelpHUD — floating "?" button on the viewport.

Parented to the QtInteractor's widget and repositioned to the
viewport's bottom-right corner on resize. Clicking the button pops
up a small panel listing the keyboard shortcuts mapped in the
results viewer.

Shortcut entries are passed in by the caller so this HUD can be
reused in any viewer with its own mapping.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class ShortcutHelpHUD:
    """Floating "?" button + popup for keyboard shortcuts.

    Parameters
    ----------
    viewport_widget
        The widget over which the button floats.
    entries
        Iterable of ``(keys, description)`` pairs, e.g.
        ``[("Esc", "Deselect"), ("Ctrl+H", "Toggle focus mode")]``.
    """

    _MARGIN = 12
    _SIZE = 28

    def __init__(
        self,
        viewport_widget: Any,
        entries: Iterable[tuple[str, str]],
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._viewport = viewport_widget
        self._entries: list[tuple[str, str]] = list(entries)

        btn = QtWidgets.QToolButton(parent=viewport_widget)
        btn.setObjectName("ShortcutHelpButton")
        btn.setText("?")
        btn.setToolTip("Keyboard shortcuts")
        btn.setFixedSize(self._SIZE, self._SIZE)
        btn.clicked.connect(self._show_popup)
        self._btn = btn

        self._popup: Optional[Any] = None

        self._filter = _ResizeFilter(self.reposition)
        viewport_widget.installEventFilter(self._filter)

        btn.show()
        btn.raise_()
        self.reposition()

    @property
    def widget(self) -> Any:
        return self._btn

    def reposition(self) -> None:
        """Move the button into the viewport's bottom-right corner."""
        try:
            vw = self._viewport.width()
            vh = self._viewport.height()
        except Exception:
            return
        x = vw - self._btn.width() - self._MARGIN
        y = vh - self._btn.height() - self._MARGIN
        self._btn.move(max(0, x), max(0, y))
        self._btn.raise_()

    def _show_popup(self) -> None:
        QtWidgets, QtCore = _qt()
        popup = QtWidgets.QFrame(parent=self._viewport)
        popup.setObjectName("ShortcutHelpPopup")
        popup.setWindowFlags(
            QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint,
        )
        popup.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        lay = QtWidgets.QVBoxLayout(popup)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        title = QtWidgets.QLabel("Keyboard shortcuts")
        title.setObjectName("ShortcutHelpTitle")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        lay.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        for row, (keys, desc) in enumerate(self._entries):
            k = QtWidgets.QLabel(keys)
            k.setObjectName("ShortcutHelpKey")
            d = QtWidgets.QLabel(desc)
            d.setObjectName("ShortcutHelpDesc")
            grid.addWidget(k, row, 0)
            grid.addWidget(d, row, 1)
        lay.addLayout(grid)

        # Anchor the popup just above the "?" button.
        btn_global = self._btn.mapToGlobal(self._btn.rect().topLeft())
        popup.adjustSize()
        px = btn_global.x() + self._btn.width() - popup.width()
        py = btn_global.y() - popup.height() - 6
        popup.move(max(0, px), max(0, py))
        popup.show()
        self._popup = popup


# =====================================================================
# _ResizeFilter — Qt event filter that fires a callback on Resize
# =====================================================================


def _resize_filter_factory():
    from qtpy import QtCore

    class _ResizeFilter(QtCore.QObject):
        def __init__(self, callback: Callable[[], None]) -> None:
            super().__init__()
            self._cb = callback

        def eventFilter(self, _obj, event):   # noqa: N802 — Qt naming
            if event.type() == QtCore.QEvent.Resize:
                try:
                    self._cb()
                except Exception:
                    pass
            return False
    return _ResizeFilter


def _ResizeFilter(callback: Callable[[], None]):  # noqa: N802 — match Qt conv.
    return _resize_filter_factory()(callback)
