"""BooleanPanel — Qt tab driving ``g.model.boolean`` from the viewer.

Two operand slots (Objects / Tools), each filled from the current
viewer selection via a "Set from selection" button (mirrors the
FreeCAD / SolidWorks boolean workflow). The panel is pure UI — it
captures operands + options and fires a single ``on_apply`` callback;
``model_viewer`` owns the library call + scene rebuild (same split as
``_parts_fuse``).

``on_apply(op, objects, tools, opts)``:
    op      "fuse" | "cut" | "intersect" | "fragment"
    objects list[DimTag]      (captured Objects slot)
    tools   list[DimTag]      (captured Tools slot)
    opts    {label: str, remove_object: bool, remove_tool: bool,
             cleanup_free: bool}
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class BooleanPanel:
    """Tab widget for the OCC boolean operations."""

    def __init__(
        self,
        *,
        get_selection: Callable[[], list[tuple[int, int]]],
        on_apply: Callable[
            [str, list[tuple[int, int]], list[tuple[int, int]], dict], None
        ],
    ) -> None:
        QtWidgets, _ = _qt()
        self._get_selection = get_selection
        self._on_apply = on_apply
        self._objects: list[tuple[int, int]] = []
        self._tools: list[tuple[int, int]] = []

        self.widget = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.widget)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        lay.addWidget(self._slot_group("Objects", "_objects"))
        lay.addWidget(self._slot_group("Tools", "_tools"))

        # ── Options ─────────────────────────────────────────────────
        opt = QtWidgets.QGroupBox("Options")
        ol = QtWidgets.QVBoxLayout(opt)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Result label:"))
        self._ed_label = QtWidgets.QLineEdit()
        self._ed_label.setPlaceholderText("(optional — names the result)")
        row.addWidget(self._ed_label, 1)
        ol.addLayout(row)
        self._cb_rm_obj = QtWidgets.QCheckBox("Consume objects")
        self._cb_rm_obj.setChecked(True)
        self._cb_rm_tool = QtWidgets.QCheckBox("Consume tools")
        self._cb_rm_tool.setChecked(True)
        self._cb_cleanup = QtWidgets.QCheckBox(
            "Fragment: drop free surfaces"
        )
        self._cb_cleanup.setChecked(False)
        for w in (self._cb_rm_obj, self._cb_rm_tool, self._cb_cleanup):
            ol.addWidget(w)
        lay.addWidget(opt)

        # ── Operation buttons ───────────────────────────────────────
        grid = QtWidgets.QGridLayout()
        for i, op in enumerate(("fuse", "cut", "intersect", "fragment")):
            b = QtWidgets.QPushButton(op.capitalize())
            b.clicked.connect(lambda _=False, o=op: self._apply(o))
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)

        self._hint = QtWidgets.QLabel("")
        self._hint.setObjectName("DiagramSettingsEmptyHint")
        self._hint.setWordWrap(True)
        lay.addWidget(self._hint)
        lay.addStretch(1)
        self._refresh_counts()

    # ------------------------------------------------------------------

    def _slot_group(self, title: str, attr: str) -> Any:
        QtWidgets, _ = _qt()
        box = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(box)
        lbl = QtWidgets.QLabel("0 entities")
        setattr(self, f"_lbl{attr}", lbl)
        v.addWidget(lbl)
        row = QtWidgets.QHBoxLayout()
        b_set = QtWidgets.QPushButton("Set from selection")
        b_set.clicked.connect(lambda: self._capture(attr))
        b_clr = QtWidgets.QPushButton("Clear")
        b_clr.clicked.connect(lambda: self._clear(attr))
        row.addWidget(b_set, 1)
        row.addWidget(b_clr)
        v.addLayout(row)
        return box

    def _capture(self, attr: str) -> None:
        picks = list(self._get_selection() or [])
        setattr(self, attr, picks)
        self._refresh_counts()

    def _clear(self, attr: str) -> None:
        setattr(self, attr, [])
        self._refresh_counts()

    def _refresh_counts(self) -> None:
        self._lbl_objects.setText(f"{len(self._objects)} entities")
        self._lbl_tools.setText(f"{len(self._tools)} entities")

    def set_hint(self, text: str) -> None:
        """Surface the last op's status / error inline."""
        self._hint.setText(text)

    def clear_operands(self) -> None:
        """Drop captured Objects/Tools — OCC renumbers after a
        boolean, so the old tags are stale; force a re-capture."""
        self._objects = []
        self._tools = []
        self._refresh_counts()

    def _apply(self, op: str) -> None:
        opts = {
            "label": self._ed_label.text().strip(),
            "remove_object": self._cb_rm_obj.isChecked(),
            "remove_tool": self._cb_rm_tool.isChecked(),
            "cleanup_free": self._cb_cleanup.isChecked(),
        }
        self._on_apply(op, list(self._objects), list(self._tools), opts)
