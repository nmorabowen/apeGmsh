"""MassesPanel — declare ``g.masses`` from model.viewer (pre-mesh).

Same shape as :class:`LoadsPanel` minus the pattern bar (masses are
not pattern-grouped). 4 types: point / line / surface / volume.

Callbacks
---------
get_target()   -> (kind, name) | None
on_apply(mass_type, target, params)
on_remove(key)
list_records() -> list[dict]   {key, type, target, name, params}
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ._decl_helpers import (
    qt, dsb, vec3, scalar, advanced_box, target_row, declared_tree,
)

# (mass_type, required target dim)
MASS_TYPES = (("point", 0), ("line", 1), ("surface", 2), ("volume", 3))


class MassesPanel:
    def __init__(
        self,
        *,
        get_target: Callable[[], Optional[tuple[str, str]]],
        on_apply: Callable[[str, tuple[str, str], dict], None],
        on_remove: Callable[[Any], None],
        list_records: Callable[[], list[dict]],
    ) -> None:
        QtWidgets, _ = qt()
        self._get_target = get_target
        self._on_apply = on_apply
        self._on_remove = on_remove
        self._list_records = list_records
        self._target: Optional[tuple[str, str]] = None
        self._fields: dict[str, dict[str, Any]] = {}

        self.widget = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.widget)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        tw, self._lbl_target = target_row(self._set_target,
                                          self._clear_target)
        lay.addWidget(tw)

        tr = QtWidgets.QHBoxLayout()
        tr.addWidget(QtWidgets.QLabel("Mass:"))
        self._combo_type = QtWidgets.QComboBox()
        self._combo_type.addItems([t for t, _ in MASS_TYPES])
        self._combo_type.currentIndexChanged.connect(
            lambda i: self._stack.setCurrentIndex(i)
        )
        tr.addWidget(self._combo_type, 1)
        lay.addLayout(tr)

        self._stack = QtWidgets.QStackedWidget()
        for t, _ in MASS_TYPES:
            self._stack.addWidget(self._build_form(t))
        lay.addWidget(self._stack)

        b_decl = QtWidgets.QPushButton("Declare")
        b_decl.clicked.connect(self._declare)
        lay.addWidget(b_decl)

        self._tree = declared_tree()
        lay.addWidget(self._tree, 1)
        lr = QtWidgets.QHBoxLayout()
        b_edit = QtWidgets.QPushButton("Edit")
        b_edit.clicked.connect(self._edit)
        b_rm = QtWidgets.QPushButton("Remove")
        b_rm.clicked.connect(self._remove)
        lr.addWidget(b_edit)
        lr.addWidget(b_rm)
        lay.addLayout(lr)

        self._hint = QtWidgets.QLabel("")
        self._hint.setObjectName("DiagramSettingsEmptyHint")
        self._hint.setWordWrap(True)
        lay.addWidget(self._hint)

        self.refresh_list()

    # ------------------------------------------------------------------

    def _build_form(self, t: str) -> Any:
        QtWidgets, _ = qt()
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        core = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(core)
        g.setContentsMargins(0, 0, 0, 0)
        f: dict[str, Any] = {}
        r = 0
        if t == "point":
            f["mass"] = scalar(g, r, "mass", "mass", dsb()); r += 1
        elif t == "line":
            f["linear_density"] = scalar(
                g, r, "linear_density", "linear density",
                dsb()); r += 1
        elif t == "surface":
            f["areal_density"] = scalar(
                g, r, "areal_density", "areal density",
                dsb()); r += 1
            c = QtWidgets.QCheckBox()
            f["derive_rotational"] = scalar(
                g, r, "derive_rotational", "derive rotational",
                c); r += 1
        elif t == "volume":
            f["density"] = scalar(g, r, "density", "density",
                                  dsb(2400.0)); r += 1
            c = QtWidgets.QCheckBox()
            f["derive_rotational"] = scalar(
                g, r, "derive_rotational", "derive rotational",
                c); r += 1
        v.addWidget(core)

        box, ag, _inner = advanced_box()
        f["name"] = scalar(ag, 0, "name", "name",
                           QtWidgets.QLineEdit())
        f["reduction"] = scalar(ag, 1, "reduction", "reduction",
                                QtWidgets.QLineEdit())
        v.addWidget(box)
        v.addStretch(1)
        self._fields[t] = f
        return w

    # ------------------------------------------------------------------

    def _set_target(self) -> None:
        tgt = self._get_target()
        if not tgt:
            self.set_hint("No Physical Group / Label selected in the "
                          "Outline.")
            return
        self._target = tgt
        self._lbl_target.setText(f"{tgt[0]}: {tgt[1]}")

    def _clear_target(self) -> None:
        self._target = None
        self._lbl_target.setText("— (select a Physical Group / Label)")

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)

    def _collect(self, t: str) -> dict:
        params: dict[str, Any] = {}
        for key, wgt in self._fields[t].items():
            if isinstance(wgt, dict):
                params[key] = [float(wgt[a].value()) for a in "xyz"]
            else:
                cls = type(wgt).__name__
                if cls == "QCheckBox":
                    params[key] = bool(wgt.isChecked())
                elif cls == "QLineEdit":
                    s = wgt.text().strip()
                    if s:
                        params[key] = s
                else:
                    params[key] = float(wgt.value())
        return params

    def _declare(self) -> None:
        if self._target is None:
            self.set_hint("Set a target (Physical Group / Label) first.")
            return
        t = self._combo_type.currentText()
        self._on_apply(t, self._target, self._collect(t))

    def refresh_list(self) -> None:
        QtWidgets, _ = qt()
        self._tree.clear()
        self._records = list(self._list_records() or [])
        for rec in self._records:
            label = rec.get("name") or rec.get("type", "?")
            it = QtWidgets.QTreeWidgetItem(
                [f"{rec.get('type','?')}  {label}",
                 str(rec.get("target", ""))]
            )
            it.setData(0, 0x101, rec)
            self._tree.addTopLevelItem(it)

    def _selected_rec(self) -> Optional[dict]:
        it = self._tree.currentItem()
        return None if it is None else it.data(0, 0x101)

    def _remove(self) -> None:
        rec = self._selected_rec()
        if rec is None:
            self.set_hint("Select a declared mass to remove.")
            return
        self._on_remove(rec.get("key"))

    def _edit(self) -> None:
        rec = self._selected_rec()
        if rec is None:
            self.set_hint("Select a declared mass to edit.")
            return
        t = rec.get("type")
        if t in [x for x, _ in MASS_TYPES]:
            self._combo_type.setCurrentText(t)
        tgt = rec.get("target_tuple")
        if tgt:
            self._target = tuple(tgt)
            self._lbl_target.setText(f"{tgt[0]}: {tgt[1]}")
        self._prefill(t, rec.get("params") or {})
        self._on_remove(rec.get("key"))
        self.set_hint("Loaded into the form — adjust and press Declare.")

    def _prefill(self, t: str, params: dict) -> None:
        for key, wgt in self._fields.get(t, {}).items():
            if key not in params:
                continue
            val = params[key]
            if isinstance(wgt, dict):
                for i, a in enumerate("xyz"):
                    try:
                        wgt[a].setValue(float(val[i]))
                    except Exception:
                        pass
            else:
                cls = type(wgt).__name__
                try:
                    if cls == "QCheckBox":
                        wgt.setChecked(bool(val))
                    elif cls == "QLineEdit":
                        wgt.setText(str(val))
                    else:
                        wgt.setValue(float(val))
                except Exception:
                    pass
