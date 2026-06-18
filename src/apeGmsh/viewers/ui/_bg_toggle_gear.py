"""Floating ⚙ button that toggles the viewport background colour."""
from __future__ import annotations


def attach_bg_toggle(parent_widget, plotter) -> None:
    """Attach a floating gear button to *parent_widget*.

    Parented to the widget but outside any layout, so it floats over the
    VTK render area.  Menu offers white (flat) or dark (theme gradient).
    """
    from qtpy import QtCore, QtWidgets

    _is_white = [False]

    btn = QtWidgets.QToolButton(parent_widget)
    btn.setText("⚙")
    btn.setToolTip("Background")
    btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
    btn.setAutoRaise(True)
    btn.setStyleSheet(
        "QToolButton { font-size: 9px; padding: 1px 2px;"
        " background: rgba(127,127,127,90); border-radius: 3px; }"
        "QToolButton::menu-indicator { image: none; }"
    )
    btn.setFixedSize(16, 16)

    def _set_white():
        _is_white[0] = True
        try:
            r = plotter.renderer
            r.SetTexturedBackground(False)
            r.SetGradientBackground(False)
            r.SetBackground(1.0, 1.0, 1.0)
            plotter.render()
        except Exception:
            pass

    def _set_dark():
        _is_white[0] = False
        try:
            from apeGmsh.viewers.scene.background import apply_background
            from apeGmsh.viewers.ui.theme import THEME
            apply_background(plotter, THEME.current)
            plotter.render()
        except Exception:
            pass

    menu = QtWidgets.QMenu(btn)
    menu.addAction("White background", _set_white)
    menu.addAction("Dark background", _set_dark)
    btn.setMenu(menu)

    _MARGIN = 8

    def _reposition():
        x = parent_widget.width() - btn.width() - _MARGIN
        btn.move(max(_MARGIN, x), _MARGIN)
        btn.raise_()

    class _Anchor(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Type.Resize:
                _reposition()
            return False

    anchor = _Anchor(parent_widget)
    parent_widget.installEventFilter(anchor)
    _reposition()
    btn.show()
