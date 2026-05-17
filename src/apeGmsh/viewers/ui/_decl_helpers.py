"""Shared widget builders for the Loads / Masses declaration panels.

Free functions (no inheritance) so the two panels stay flat and
self-contained — matching the Boolean/Transform panel idiom — while
not duplicating the core-grid / Advanced-box / target-row / declared
-list scaffolding twice.
"""
from __future__ import annotations

from typing import Any, Callable


def qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


def dsb(default: float = 0.0) -> Any:
    """A wide-range QDoubleSpinBox."""
    QtWidgets, _ = qt()
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(-1.0e12, 1.0e12)
    s.setDecimals(4)
    s.setValue(default)
    return s


def vec3(grid: Any, row: int, label: str,
         d: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    """Add ``label  [x][y][z]`` to *grid* at *row*; return {x,y,z}."""
    QtWidgets, _ = qt()
    grid.addWidget(QtWidgets.QLabel(label), row, 0)
    cell = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(cell)
    h.setContentsMargins(0, 0, 0, 0)
    out = {}
    for k, dv in zip("xyz", d):
        sb = dsb(dv)
        out[k] = sb
        h.addWidget(sb)
    grid.addWidget(cell, row, 1)
    return out


def scalar(grid: Any, row: int, key: str, label: str,
           widget: Any) -> Any:
    """Add ``label  [widget]`` to *grid* at *row*; return *widget*."""
    QtWidgets, _ = qt()
    grid.addWidget(QtWidgets.QLabel(label), row, 0)
    grid.addWidget(widget, row, 1)
    return widget


def advanced_box() -> tuple[Any, Any, Any]:
    """Collapsible 'Advanced' group.

    Returns ``(groupbox, inner_grid, toggle_fn)``. The box is
    checkable and starts collapsed (inner hidden); toggling the
    check shows/hides the inner grid container.
    """
    QtWidgets, _ = qt()
    box = QtWidgets.QGroupBox("Advanced")
    box.setCheckable(True)
    box.setChecked(False)
    outer = QtWidgets.QVBoxLayout(box)
    outer.setContentsMargins(6, 4, 6, 6)
    inner = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(inner)
    grid.setContentsMargins(0, 0, 0, 0)
    inner.setVisible(False)
    outer.addWidget(inner)
    box.toggled.connect(inner.setVisible)
    return box, grid, inner


def target_row(on_set: Callable[[], None],
                on_clear: Callable[[], None]) -> tuple[Any, Any]:
    """``Target: <none>  [Set from selection][Clear]`` row.

    Returns ``(widget, label)`` — caller updates *label* text.
    """
    QtWidgets, _ = qt()
    w = QtWidgets.QGroupBox("Target")
    v = QtWidgets.QVBoxLayout(w)
    lbl = QtWidgets.QLabel("— (select a Physical Group / Label)")
    v.addWidget(lbl)
    row = QtWidgets.QHBoxLayout()
    b1 = QtWidgets.QPushButton("Set from selection")
    b1.clicked.connect(lambda: on_set())
    b2 = QtWidgets.QPushButton("Clear")
    b2.clicked.connect(lambda: on_clear())
    row.addWidget(b1, 1)
    row.addWidget(b2)
    v.addLayout(row)
    return w, lbl


def declared_tree() -> Any:
    """Two-column tree (item, target) for the declared list."""
    QtWidgets, _ = qt()
    t = QtWidgets.QTreeWidget()
    t.setColumnCount(2)
    t.setHeaderLabels(["Declared", "Target"])
    t.setRootIsDecorated(True)
    t.setUniformRowHeights(True)
    return t
