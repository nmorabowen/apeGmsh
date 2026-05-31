"""Help → Shortcuts menu (``ui/_shortcuts_help.py``).

Headless Qt: builds the menu on a bare QMainWindow, triggers the action,
and asserts the dialog lists the entries. No viewer / VTK construction.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy.QtWidgets")

from apeGmsh.viewers.ui._shortcuts_help import add_help_shortcuts_menu


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


ENTRIES = [
    ("Esc", "Deselect"),
    ("Q", "Close window"),
    ("N / E / G", "Pick mode"),
]


def _window():
    from qtpy import QtWidgets
    return QtWidgets.QMainWindow()


def test_adds_help_menu_with_shortcuts_action(qapp):
    win = _window()
    menu = add_help_shortcuts_menu(win, ENTRIES)
    assert "Help" in [a.text() for a in win.menuBar().actions()]
    assert "Shortcuts" in [a.text() for a in menu.actions()]


def test_shortcuts_action_opens_dialog_listing_entries(qapp):
    from qtpy import QtWidgets

    win = _window()
    menu = add_help_shortcuts_menu(win, ENTRIES)
    act = next(a for a in menu.actions() if a.text() == "Shortcuts")
    act.trigger()

    dlg = win.findChild(QtWidgets.QDialog, "ShortcutsDialog")
    assert dlg is not None
    keys = [
        w.text() for w in dlg.findChildren(QtWidgets.QLabel)
        if w.objectName() == "ShortcutHelpKey"
    ]
    descs = [
        w.text() for w in dlg.findChildren(QtWidgets.QLabel)
        if w.objectName() == "ShortcutHelpDesc"
    ]
    assert keys == ["Esc", "Q", "N / E / G"]
    assert descs == ["Deselect", "Close window", "Pick mode"]
    dlg.close()


def test_help_menu_is_idempotent(qapp):
    win = _window()
    add_help_shortcuts_menu(win, ENTRIES)
    add_help_shortcuts_menu(win, ENTRIES)   # rebuild, not duplicate
    help_menus = [a for a in win.menuBar().actions() if a.text() == "Help"]
    assert len(help_menus) == 1


def test_returns_menu_with_f1_shortcut(qapp):
    from qtpy import QtGui

    win = _window()
    menu = add_help_shortcuts_menu(win, ENTRIES)
    act = next(a for a in menu.actions() if a.text() == "Shortcuts")
    assert act.shortcut() == QtGui.QKeySequence("F1")
