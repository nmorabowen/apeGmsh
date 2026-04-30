"""Inspector tab — picked entity (node / element) details + history shortcut.

Phase 5 v1 surface: explicit node-ID / element-ID input fields. The
user enters an ID, the tab fetches:

* For nodes: coordinates, current values from the active diagrams'
  components.
* For elements: connectivity, element-type info.

A "Open time history…" button spawns a ``TimeHistoryPanel`` for the
node + component pair. Interactive 3-D pick integration arrives in a
follow-up phase (uses ``PickEngine``); for now, the data path through
the Director is fully exercised via explicit IDs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class InspectorTab:
    """Read picked / requested entity + show its data."""

    def __init__(
        self,
        director: "ResultsDirector",
        on_open_history: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        QtWidgets, _ = _qt()
        self._director = director
        self._on_open_history = on_open_history

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Node block ─────────────────────────────────────────────
        node_box = QtWidgets.QGroupBox("Node")
        node_grid = QtWidgets.QGridLayout(node_box)

        node_grid.addWidget(QtWidgets.QLabel("Node ID:"), 0, 0)
        self._node_id_edit = QtWidgets.QLineEdit()
        self._node_id_edit.setPlaceholderText("e.g. 1")
        self._node_id_edit.editingFinished.connect(self._refresh_node)
        node_grid.addWidget(self._node_id_edit, 0, 1)

        self._node_lookup_btn = QtWidgets.QPushButton("Lookup")
        self._node_lookup_btn.clicked.connect(self._refresh_node)
        node_grid.addWidget(self._node_lookup_btn, 0, 2)

        node_grid.addWidget(QtWidgets.QLabel("Coords:"), 1, 0)
        self._node_coords_label = QtWidgets.QLabel("—")
        node_grid.addWidget(self._node_coords_label, 1, 1, 1, 2)

        node_grid.addWidget(QtWidgets.QLabel("Values:"), 2, 0)
        self._node_values_text = QtWidgets.QPlainTextEdit()
        self._node_values_text.setReadOnly(True)
        self._node_values_text.setMaximumHeight(160)
        node_grid.addWidget(self._node_values_text, 2, 1, 1, 2)

        history_row = QtWidgets.QHBoxLayout()
        history_row.addWidget(QtWidgets.QLabel("Component:"))
        self._hist_component_edit = QtWidgets.QLineEdit()
        self._hist_component_edit.setPlaceholderText("displacement_x")
        history_row.addWidget(self._hist_component_edit, stretch=1)
        self._hist_btn = QtWidgets.QPushButton("Open time history…")
        self._hist_btn.clicked.connect(self._open_history)
        history_row.addWidget(self._hist_btn)
        node_grid.addLayout(history_row, 3, 0, 1, 3)

        layout.addWidget(node_box)

        # ── Element block ───────────────────────────────────────────
        elem_box = QtWidgets.QGroupBox("Element")
        elem_grid = QtWidgets.QGridLayout(elem_box)

        elem_grid.addWidget(QtWidgets.QLabel("Element ID:"), 0, 0)
        self._elem_id_edit = QtWidgets.QLineEdit()
        self._elem_id_edit.setPlaceholderText("e.g. 1")
        self._elem_id_edit.editingFinished.connect(self._refresh_element)
        elem_grid.addWidget(self._elem_id_edit, 0, 1)

        self._elem_lookup_btn = QtWidgets.QPushButton("Lookup")
        self._elem_lookup_btn.clicked.connect(self._refresh_element)
        elem_grid.addWidget(self._elem_lookup_btn, 0, 2)

        elem_grid.addWidget(QtWidgets.QLabel("Type:"), 1, 0)
        self._elem_type_label = QtWidgets.QLabel("—")
        elem_grid.addWidget(self._elem_type_label, 1, 1, 1, 2)

        elem_grid.addWidget(QtWidgets.QLabel("Connectivity:"), 2, 0)
        self._elem_conn_label = QtWidgets.QLabel("—")
        self._elem_conn_label.setWordWrap(True)
        elem_grid.addWidget(self._elem_conn_label, 2, 1, 1, 2)

        layout.addWidget(elem_box)

        layout.addStretch(1)

        self._widget = widget

        director.subscribe_step(lambda _i: self._refresh_node())
        director.subscribe_stage(lambda _id: self._refresh_node())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def set_node_id(self, node_id: int) -> None:
        self._node_id_edit.setText(str(int(node_id)))
        self._refresh_node()

    def set_element_id(self, element_id: int) -> None:
        self._elem_id_edit.setText(str(int(element_id)))
        self._refresh_element()

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _refresh_node(self) -> None:
        text = self._node_id_edit.text().strip()
        if not text:
            self._node_coords_label.setText("—")
            self._node_values_text.setPlainText("")
            return
        try:
            node_id = int(text)
        except ValueError:
            self._node_coords_label.setText("(not an integer)")
            self._node_values_text.setPlainText("")
            return

        # Coords
        coords = self._lookup_node_coords(node_id)
        if coords is None:
            self._node_coords_label.setText(f"node {node_id} not found")
            self._node_values_text.setPlainText("")
            return
        self._node_coords_label.setText(
            f"({coords[0]:.4g}, {coords[1]:.4g}, {coords[2]:.4g})"
        )

        # Components — pull from the active diagrams' selectors
        components = self._collect_active_components()
        if not components:
            self._node_values_text.setPlainText(
                "(add diagrams to see values here)"
            )
            return

        values = self._director.read_at_pick(node_id, components)
        if not values:
            self._node_values_text.setPlainText("(no values at current step)")
            return
        lines = [
            f"step {self._director.step_index}"
            + (
                f"   t = {self._director.current_time():.4g}"
                if self._director.current_time() is not None else ""
            ),
            "",
        ]
        for c, v in values.items():
            lines.append(f"  {c:24s} = {v:.6g}")
        self._node_values_text.setPlainText("\n".join(lines))

    def _refresh_element(self) -> None:
        text = self._elem_id_edit.text().strip()
        if not text:
            self._elem_type_label.setText("—")
            self._elem_conn_label.setText("—")
            return
        try:
            element_id = int(text)
        except ValueError:
            self._elem_type_label.setText("(not an integer)")
            self._elem_conn_label.setText("—")
            return

        type_name, conn = self._lookup_element(element_id)
        if type_name is None:
            self._elem_type_label.setText(f"element {element_id} not found")
            self._elem_conn_label.setText("—")
            return
        self._elem_type_label.setText(type_name)
        if conn is None:
            self._elem_conn_label.setText("—")
        else:
            self._elem_conn_label.setText(
                ", ".join(str(int(x)) for x in conn)
            )

    def _open_history(self) -> None:
        node_text = self._node_id_edit.text().strip()
        component = self._hist_component_edit.text().strip()
        if not node_text or not component:
            return
        try:
            node_id = int(node_text)
        except ValueError:
            return
        if self._on_open_history is not None:
            try:
                self._on_open_history(node_id, component)
            except Exception as exc:
                import sys
                print(
                    f"[InspectorTab] open_history failed: {exc}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # FEM lookups
    # ------------------------------------------------------------------

    def _collect_active_components(self) -> list[str]:
        """Components currently in use by any attached diagram."""
        out: list[str] = []
        seen: set[str] = set()
        for d in self._director.registry.diagrams():
            if not d.is_attached:
                continue
            comp = d.spec.selector.component
            if comp not in seen:
                seen.add(comp)
                out.append(comp)
        return out

    def _lookup_node_coords(
        self, node_id: int,
    ) -> "Optional[ndarray]":
        fem = self._director.fem
        if fem is None:
            return None
        node_ids_arr = np.asarray(list(fem.nodes.ids), dtype=np.int64)
        coords_arr = np.asarray(fem.nodes.coords, dtype=np.float64)
        idx_match = np.where(node_ids_arr == int(node_id))[0]
        if idx_match.size == 0:
            return None
        return coords_arr[idx_match[0]]

    def _lookup_element(
        self, element_id: int,
    ) -> "tuple[Optional[str], Optional[ndarray]]":
        fem = self._director.fem
        if fem is None:
            return None, None
        for group in fem.elements:
            ids = np.asarray(group.ids, dtype=np.int64)
            idx_match = np.where(ids == int(element_id))[0]
            if idx_match.size == 0:
                continue
            type_name = group.element_type.name
            conn = np.asarray(group.connectivity[idx_match[0]], dtype=np.int64)
            return type_name, conn
        return None, None
