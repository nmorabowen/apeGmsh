"""
Mesh Tabs — UI components for the mesh viewer.

* **MeshInfoTab**: Shows picked element/node details
* **DisplayTab**: Color mode, label toggles, wireframe/edges
* **MeshFilterTab**: Node visibility, mesh dims, element types

Usage::

    from pyGmsh.viewers.ui.mesh_tabs import MeshInfoTab, DisplayTab, MeshFilterTab
    info = MeshInfoTab()
    display = DisplayTab(on_color_mode=..., on_elem_labels=...)
    filter_tab = MeshFilterTab(dims=[1,2,3], on_filter=...)
"""
from __future__ import annotations

from typing import Any, Callable


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


# ======================================================================
# MeshInfoTab — picked element/node details
# ======================================================================

class MeshInfoTab:
    """Tree widget showing details of picked elements and nodes."""

    def __init__(self) -> None:
        QtWidgets, _, _ = _qt()

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Property", "Value"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        layout.addWidget(self._tree)

    def show_element(self, elem_tag: int, elem_data: dict) -> None:
        """Display element details."""
        QtWidgets, _, _ = _qt()
        self._tree.clear()
        root = QtWidgets.QTreeWidgetItem(self._tree)
        root.setText(0, f"Element {elem_tag}")
        root.setExpanded(True)

        for key, val in [
            ("Type", elem_data.get("type_name", "?")),
            ("Dim", str(elem_data.get("dim", "?"))),
            ("BRep", str(elem_data.get("brep_dt", "?"))),
            ("Nodes", str(elem_data.get("nodes", []))),
        ]:
            item = QtWidgets.QTreeWidgetItem(root)
            item.setText(0, key)
            item.setText(1, val)

    def show_node(self, node_tag: int, coords: Any) -> None:
        """Display node details."""
        QtWidgets, _, _ = _qt()
        self._tree.clear()
        root = QtWidgets.QTreeWidgetItem(self._tree)
        root.setText(0, f"Node {node_tag}")
        root.setExpanded(True)

        if coords is not None:
            for i, label in enumerate(["X", "Y", "Z"]):
                item = QtWidgets.QTreeWidgetItem(root)
                item.setText(0, label)
                item.setText(1, f"{coords[i]:.6f}")

    def show_summary(self, n_nodes: int, n_elems: int, n_picked: int = 0) -> None:
        """Display mesh summary."""
        QtWidgets, _, _ = _qt()
        self._tree.clear()
        for label, val in [
            ("Nodes", str(n_nodes)),
            ("Elements", str(n_elems)),
            ("Picked", str(n_picked)),
        ]:
            item = QtWidgets.QTreeWidgetItem(self._tree)
            item.setText(0, label)
            item.setText(1, val)

    def clear(self) -> None:
        self._tree.clear()


# ======================================================================
# DisplayTab — color mode, labels, wireframe
# ======================================================================

COLOR_MODES = ["Default", "Partition", "Quality", "Element Type", "Physical Group"]


class DisplayTab:
    """Color mode dropdown + label/wireframe toggles."""

    def __init__(
        self,
        *,
        on_color_mode: Callable[[str], None] | None = None,
        on_node_labels: Callable[[bool], None] | None = None,
        on_elem_labels: Callable[[bool], None] | None = None,
        on_wireframe: Callable[[bool], None] | None = None,
        on_show_edges: Callable[[bool], None] | None = None,
    ) -> None:
        QtWidgets, _, _ = _qt()

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Color mode
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Color mode:"))
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(COLOR_MODES)
        if on_color_mode:
            self._combo.currentTextChanged.connect(on_color_mode)
        row.addWidget(self._combo)
        layout.addLayout(row)

        # Label toggles
        self._cb_node_labels = QtWidgets.QCheckBox("Node labels")
        if on_node_labels:
            self._cb_node_labels.toggled.connect(on_node_labels)
        layout.addWidget(self._cb_node_labels)

        self._cb_elem_labels = QtWidgets.QCheckBox("Element labels")
        if on_elem_labels:
            self._cb_elem_labels.toggled.connect(on_elem_labels)
        layout.addWidget(self._cb_elem_labels)

        # Wireframe / edges
        self._cb_wireframe = QtWidgets.QCheckBox("Wireframe")
        if on_wireframe:
            self._cb_wireframe.toggled.connect(on_wireframe)
        layout.addWidget(self._cb_wireframe)

        self._cb_edges = QtWidgets.QCheckBox("Show edges")
        self._cb_edges.setChecked(True)
        if on_show_edges:
            self._cb_edges.toggled.connect(on_show_edges)
        layout.addWidget(self._cb_edges)

        layout.addStretch()


# ======================================================================
# MeshFilterTab — dimension/type/group filtering
# ======================================================================

class MeshFilterTab:
    """Checkboxes for mesh dimension, element type, and physical group filtering."""

    def __init__(
        self,
        dims: list[int],
        *,
        on_filter_changed: Callable[[set[int]], None] | None = None,
    ) -> None:
        QtWidgets, _, _ = _qt()
        self._on_filter = on_filter_changed

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(QtWidgets.QLabel("Mesh dimensions:"))

        self._dim_cbs: dict[int, object] = {}
        dim_labels = {1: "1D (Lines)", 2: "2D (Surfaces)", 3: "3D (Volumes)"}
        for d in sorted(dims):
            cb = QtWidgets.QCheckBox(dim_labels.get(d, f"dim={d}"))
            cb.setChecked(True)
            cb.toggled.connect(self._on_dim_toggled)
            self._dim_cbs[d] = cb
            layout.addWidget(cb)

        layout.addStretch()

    def _on_dim_toggled(self, _checked: bool) -> None:
        active = {d for d, cb in self._dim_cbs.items() if cb.isChecked()}
        if self._on_filter:
            self._on_filter(active)
