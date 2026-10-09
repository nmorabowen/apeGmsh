"""Close and delete the top-level widgets a test leaves behind (#1571).

A panel whose buttons connect to lambdas that capture the panel is kept
alive by its own connections: the C++ side holds the lambda, so Python's
collector never frees it. Such a widget outlives its test and is still
alive at interpreter exit, where PySide 6.12's ``destroyQCoreApplication``
deletes it and its children and segfaults in ``QObject::~QObject``.
Deleting the widget at teardown drops its connections, which breaks the
cycle, and nothing is left for the exit cleanup.

Use it as an autouse fixture in a module whose tests build widgets::

    @pytest.fixture(autouse=True)
    def _disposes_its_widgets():
        yield from disposing_new_top_levels()
"""
from __future__ import annotations

import sys
from collections.abc import Iterator


def _top_levels() -> list:
    # A test that never imported Qt has made no widget, and importing it
    # here would make Qt a dependency of that test.
    QtWidgets = sys.modules.get("qtpy.QtWidgets")
    if QtWidgets is None:
        return []
    app = QtWidgets.QApplication.instance()
    return [] if app is None else list(app.topLevelWidgets())


def disposing_new_top_levels() -> Iterator[None]:
    """Yield to the test, then delete every top-level widget it created.

    ``before`` holds the wrappers of the widgets that already existed, so
    shiboken hands back the same wrapper for each and ``is`` compares C++
    objects. Fails loudly if a new widget survives the deletion.
    """
    before = _top_levels()
    yield

    def new() -> list:
        return [w for w in _top_levels() if not any(w is b for b in before)]

    leaked = new()
    if not leaked:
        return
    from qtpy import QtCore

    for w in leaked:
        w.close()
        w.deleteLater()
    del leaked
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete)
    QtCore.QCoreApplication.processEvents()
    survivors = new()
    assert not survivors, (
        f"top-level widgets survived teardown: "
        f"{[type(w).__name__ for w in survivors]}")
