"""Shortcuts help — a top-menu "Help → Shortcuts" entry + a list dialog.

Each viewer registers its own ``(keys, description)`` mapping via
:func:`add_help_shortcuts_menu`; triggering **Help → Shortcuts** (or
``F1``) shows a dialog listing them. This replaces the old floating "?"
HUD so all three viewers (model / mesh / results) expose their shortcuts
the same way, in the top menu bar.
"""
from __future__ import annotations

from typing import Any, Iterable


def _qt():
    from qtpy import QtCore, QtGui, QtWidgets
    return QtWidgets, QtGui, QtCore


def _build_dialog(parent: Any, entries: list[tuple[str, str]], title: str) -> Any:
    """A modeless ``QDialog`` listing ``(keys, description)`` rows."""
    QtWidgets, QtGui, QtCore = _qt()

    dlg = QtWidgets.QDialog(parent)
    dlg.setObjectName("ShortcutsDialog")
    dlg.setWindowTitle(title)
    try:
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    except Exception:
        pass

    lay = QtWidgets.QVBoxLayout(dlg)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(8)

    heading = QtWidgets.QLabel(title)
    heading.setObjectName("ShortcutHelpTitle")
    f = heading.font()
    f.setBold(True)
    heading.setFont(f)
    lay.addWidget(heading)

    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(18)
    grid.setVerticalSpacing(4)
    for row, (keys, desc) in enumerate(entries):
        k = QtWidgets.QLabel(str(keys))
        k.setObjectName("ShortcutHelpKey")
        d = QtWidgets.QLabel(str(desc))
        d.setObjectName("ShortcutHelpDesc")
        grid.addWidget(k, row, 0)
        grid.addWidget(d, row, 1)
    lay.addLayout(grid)

    btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    btns.rejected.connect(dlg.close)
    btns.accepted.connect(dlg.close)
    lay.addWidget(btns)
    return dlg


def add_help_shortcuts_menu(
    window: Any,
    entries: Iterable[tuple[str, str]],
    *,
    dialog_title: str = "Keyboard shortcuts",
) -> Any:
    """Add (or replace) a **Help → Shortcuts** entry on ``window``'s menu bar.

    Parameters
    ----------
    window
        The ``QMainWindow`` (every viewer exposes one as ``win.window``).
    entries
        Iterable of ``(keys, description)`` pairs — the viewer's own
        shortcut mapping.
    dialog_title
        Title for the popup + its heading.

    Idempotent: a pre-existing ``Help`` menu is removed first, so a viewer
    can rebuild it. Returns the created ``QMenu``.
    """
    QtWidgets, QtGui, QtCore = _qt()
    entries = list(entries)

    menu_bar = window.menuBar()
    # Drop any pre-existing Help menu (rebuild-safe).
    for action in list(menu_bar.actions()):
        if action.text() == "Help":
            menu_bar.removeAction(action)

    menu = menu_bar.addMenu("Help")
    act = QtWidgets.QAction("Shortcuts", window)
    try:
        act.setShortcut(QtGui.QKeySequence("F1"))
        # ApplicationShortcut so F1 fires even when the VTK viewport has
        # focus (a WindowShortcut never fires from the QtInteractor —
        # see the VTK keyboard-shortcuts note).
        act.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    except Exception:
        pass

    def _show() -> None:
        _build_dialog(window, entries, dialog_title).show()

    act.triggered.connect(_show)
    menu.addAction(act)
    return menu


__all__ = ["add_help_shortcuts_menu"]
