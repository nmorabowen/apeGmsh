"""Close and delete the top-level widgets a test leaves behind (#1571).

A panel whose buttons connect to lambdas that capture the panel is kept
alive by its own connections: the C++ side holds the lambda, so Python's
collector never frees it. Such a widget outlives its test and is still
alive at interpreter exit, where PySide 6.12's ``destroyQCoreApplication``
deletes it and its children and segfaults in ``QObject::~QObject``.
Deleting the widget at teardown drops its connections, which breaks the
cycle, and nothing is left for the exit cleanup.

``tests/viewers/conftest.py`` runs this once per viewer module; the two
modules that crashed on their own also run it once per test.
"""
from __future__ import annotations

import sys
from collections.abc import Iterator


def _qt():
    """``(QtWidgets, QtCore)`` of the binding already imported, else ``None``.

    A module that never imported Qt has made no widget, and importing it
    here would make Qt a dependency of that module.
    """
    for pkg in ("PySide6", "qtpy"):
        widgets = sys.modules.get(f"{pkg}.QtWidgets")
        if widgets is not None:
            return widgets, sys.modules[f"{pkg}.QtCore"]
    return None


def _top_levels() -> list:
    qt = _qt()
    app = None if qt is None else qt[0].QApplication.instance()
    return [] if app is None else list(app.topLevelWidgets())


def _hooks() -> tuple:
    return sys.excepthook, sys.unraisablehook


def disposing_new_top_levels(where: str = "this test") -> Iterator[None]:
    """Yield, then delete every top-level widget created meanwhile.

    ``before`` holds the wrappers of the widgets that already existed, so
    shiboken hands back the same wrapper for each and ``is`` compares C++
    objects: a widget that existed before is never touched, nor is the
    QApplication (not a widget). The process hooks a viewer's log router
    installs (``sys.excepthook``, ``sys.unraisablehook``) are restored
    first: the router writes to a widget about to be deleted, and a slot
    that raises on it would re-enter the hook without end. (pytest
    resets ``warnings.showwarning`` around every test itself.) Fails, naming
    ``where``, if a new widget survives the deletion.
    """
    before = _top_levels()
    hooks = _hooks()
    yield
    if _hooks() != hooks:
        sys.excepthook, sys.unraisablehook = hooks

    def new() -> list:
        return [w for w in _top_levels() if not any(w is b for b in before)]

    leaked = new()
    if not leaked:
        return
    QtCore = _qt()[1]
    for w in leaked:
        w.close()
        w.deleteLater()
    del leaked
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete)
    QtCore.QCoreApplication.processEvents()
    survivors = new()
    assert not survivors, (
        f"{where}: top-level widgets survived teardown: "
        f"{[type(w).__name__ for w in survivors]}")
