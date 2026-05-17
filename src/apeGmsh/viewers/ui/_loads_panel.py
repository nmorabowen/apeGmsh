"""LoadsPanel — declare ``g.loads`` from model.viewer (pre-mesh).

Pure UI: pattern bar + type combo + stacked per-type forms (core
fields + a collapsible Advanced box) + a target captured from the
outline selection + a declared-list with Remove/Edit. ``model_viewer``
owns the library call (``with g.loads.pattern(p): g.loads.<type>(…)``)
and the target dim-validation, mirroring the Boolean/Transform split.

Callbacks
---------
get_target()      -> (kind, name) | None   current outline PG/label
get_patterns()    -> list[str]             g.loads.patterns()
on_apply(load_type, pattern, target, params)
on_remove(key)
list_records()    -> list[dict]            {key,pattern,type,
                                            target,name,params}
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ._decl_helpers import (
    qt, dsb, vec3, scalar, advanced_box, target_row, declared_tree,
)

# (load_type, required target dim) — validated by model_viewer.
LOAD_TYPES = (
    ("point", 0), ("line", 1), ("surface", 2), ("gravity", 3),
    ("body", 3), ("face_load", 2), ("face_sp", 2),
)


class LoadsPanel:
    def __init__(
        self,
        *,
        get_target: Callable[[], Optional[tuple[str, str]]],
        get_patterns: Callable[[], list[str]],
        on_apply: Callable[[str, str, tuple[str, str], dict], None],
        on_remove: Callable[[Any], None],
        list_records: Callable[[], list[dict]],
    ) -> None:
        QtWidgets, _ = qt()
        self._get_target = get_target
        self._get_patterns = get_patterns
        self._on_apply = on_apply
        self._on_remove = on_remove
        self._list_records = list_records
        self._target: Optional[tuple[str, str]] = None
        self._fields: dict[str, dict[str, Any]] = {}

        self.widget = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.widget)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # ── Pattern bar ─────────────────────────────────────────────
        pr = QtWidgets.QHBoxLayout()
        pr.addWidget(QtWidgets.QLabel("Pattern:"))
        self._combo_pat = QtWidgets.QComboBox()
        self._combo_pat.setEditable(False)
        pr.addWidget(self._combo_pat, 1)
        b_new = QtWidgets.QPushButton("New…")
        b_new.clicked.connect(self._new_pattern)
        pr.addWidget(b_new)
        lay.addLayout(pr)

        # ── Target ──────────────────────────────────────────────────
        tw, self._lbl_target = target_row(self._set_target,
                                          self._clear_target)
        lay.addWidget(tw)

        # ── Type + stacked forms ────────────────────────────────────
        tr = QtWidgets.QHBoxLayout()
        tr.addWidget(QtWidgets.QLabel("Load:"))
        self._combo_type = QtWidgets.QComboBox()
        self._combo_type.addItems([t for t, _ in LOAD_TYPES])
        self._combo_type.currentIndexChanged.connect(
            lambda i: self._stack.setCurrentIndex(i)
        )
        tr.addWidget(self._combo_type, 1)
        lay.addLayout(tr)

        self._stack = QtWidgets.QStackedWidget()
        for t, _ in LOAD_TYPES:
            self._stack.addWidget(self._build_form(t))
        lay.addWidget(self._stack)

        b_decl = QtWidgets.QPushButton("Declare")
        b_decl.clicked.connect(self._declare)
        lay.addWidget(b_decl)

        # ── Declared list ───────────────────────────────────────────
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

        self.refresh_patterns()
        self.refresh_list()

    # ------------------------------------------------------------------
    # Forms
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
            f["force_xyz"] = vec3(g, r, "force xyz"); r += 1
            f["moment_xyz"] = vec3(g, r, "moment xyz"); r += 1
        elif t == "line":
            f["magnitude"] = scalar(g, r, "magnitude", "magnitude",
                                    dsb()); r += 1
            f["direction"] = vec3(g, r, "direction", (0, 0, -1)); r += 1
            cb = QtWidgets.QCheckBox()
            f["normal"] = scalar(g, r, "normal", "normal", cb); r += 1
        elif t == "surface":
            f["magnitude"] = scalar(g, r, "magnitude", "magnitude",
                                    dsb()); r += 1
            cb = QtWidgets.QCheckBox(); cb.setChecked(True)
            f["normal"] = scalar(g, r, "normal", "normal", cb); r += 1
            f["direction"] = vec3(g, r, "direction (if !normal)"); r += 1
        elif t == "gravity":
            f["g"] = vec3(g, r, "g xyz", (0, 0, -9.81)); r += 1
            f["density"] = scalar(g, r, "density", "density",
                                  dsb(2400.0)); r += 1
        elif t == "body":
            f["force_per_volume"] = vec3(g, r, "force / volume"); r += 1
        elif t == "face_load":
            f["force_xyz"] = vec3(g, r, "force xyz"); r += 1
            f["moment_xyz"] = vec3(g, r, "moment xyz"); r += 1
        elif t == "face_sp":
            dof_w = QtWidgets.QWidget()
            dh = QtWidgets.QHBoxLayout(dof_w)
            dh.setContentsMargins(0, 0, 0, 0)
            dofs = []
            for nm in ("ux", "uy", "uz", "rx", "ry", "rz"):
                c = QtWidgets.QCheckBox(nm)
                dofs.append(c)
                dh.addWidget(c)
            g.addWidget(QtWidgets.QLabel("dofs"), r, 0)
            g.addWidget(dof_w, r, 1)
            f["_dofs"] = dofs
            r += 1
            f["disp_xyz"] = vec3(g, r, "disp xyz"); r += 1
            f["rot_xyz"] = vec3(g, r, "rot xyz"); r += 1
        v.addWidget(core)

        # ── Advanced (collapsed) ────────────────────────────────────
        box, ag, _inner = advanced_box()
        ar = 0
        f["name"] = scalar(ag, ar, "name", "name",
                           QtWidgets.QLineEdit()); ar += 1
        if t in ("line", "surface", "gravity", "body"):
            f["reduction"] = scalar(
                ag, ar, "reduction", "reduction",
                QtWidgets.QLineEdit()); ar += 1
            f["target_form"] = scalar(
                ag, ar, "target_form", "target_form",
                QtWidgets.QLineEdit()); ar += 1
        if t == "line":
            f["away_from"] = vec3(ag, ar, "away_from"); ar += 1
        v.addWidget(box)
        v.addStretch(1)
        self._fields[t] = f
        return w

    # ------------------------------------------------------------------
    # Pattern / target
    # ------------------------------------------------------------------

    def refresh_patterns(self) -> None:
        cur = self._combo_pat.currentText()
        self._combo_pat.blockSignals(True)
        self._combo_pat.clear()
        pats = list(dict.fromkeys(
            ["default", *(self._get_patterns() or [])]
        ))
        self._combo_pat.addItems(pats)
        if cur in pats:
            self._combo_pat.setCurrentText(cur)
        self._combo_pat.blockSignals(False)

    def _new_pattern(self) -> None:
        QtWidgets, _ = qt()
        name, ok = QtWidgets.QInputDialog.getText(
            self.widget, "New load pattern", "Pattern name:"
        )
        if ok and name.strip():
            n = name.strip()
            if self._combo_pat.findText(n) < 0:
                self._combo_pat.addItem(n)
            self._combo_pat.setCurrentText(n)

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

    # ------------------------------------------------------------------
    # Declare / list / remove / edit
    # ------------------------------------------------------------------

    def _collect(self, t: str) -> dict:
        params: dict[str, Any] = {}
        for key, w in self._fields[t].items():
            if key == "_dofs":
                params["dofs"] = [1 if c.isChecked() else 0 for c in w]
                continue
            if isinstance(w, dict):           # vec3
                params[key] = [float(w[a].value()) for a in "xyz"]
            else:
                cls = type(w).__name__
                if cls == "QCheckBox":
                    params[key] = bool(w.isChecked())
                elif cls == "QLineEdit":
                    s = w.text().strip()
                    if s:
                        params[key] = s     # blank → library default
                else:                        # QDoubleSpinBox
                    params[key] = float(w.value())
        return params

    def _declare(self) -> None:
        if self._target is None:
            self.set_hint("Set a target (Physical Group / Label) first.")
            return
        t = self._combo_type.currentText()
        self._on_apply(
            t, self._combo_pat.currentText(), self._target,
            self._collect(t),
        )

    def refresh_list(self) -> None:
        QtWidgets, _ = qt()
        self._tree.clear()
        self._records = list(self._list_records() or [])
        groups: dict[str, Any] = {}
        for rec in self._records:
            pat = rec.get("pattern", "default")
            parent = groups.get(pat)
            if parent is None:
                parent = QtWidgets.QTreeWidgetItem([pat, ""])
                self._tree.addTopLevelItem(parent)
                parent.setExpanded(True)
                groups[pat] = parent
            label = rec.get("name") or rec.get("type", "?")
            it = QtWidgets.QTreeWidgetItem(
                [f"{rec.get('type','?')}  {label}",
                 str(rec.get("target", ""))]
            )
            it.setData(0, 0x100, rec.get("key"))
            it.setData(0, 0x101, rec)
            parent.addChild(it)

    def _selected_rec(self) -> Optional[dict]:
        it = self._tree.currentItem()
        if it is None:
            return None
        return it.data(0, 0x101)

    def _remove(self) -> None:
        rec = self._selected_rec()
        if rec is None:
            self.set_hint("Select a declared load to remove.")
            return
        self._on_remove(rec.get("key"))

    def _edit(self) -> None:
        rec = self._selected_rec()
        if rec is None:
            self.set_hint("Select a declared load to edit.")
            return
        t = rec.get("type")
        if t in [x for x, _ in LOAD_TYPES]:
            self._combo_type.setCurrentText(t)
        pat = rec.get("pattern", "default")
        if self._combo_pat.findText(pat) < 0:
            self._combo_pat.addItem(pat)
        self._combo_pat.setCurrentText(pat)
        tgt = rec.get("target_tuple")
        if tgt:
            self._target = tuple(tgt)
            self._lbl_target.setText(f"{tgt[0]}: {tgt[1]}")
        self._prefill(t, rec.get("params") or {})
        # Edit = remove the old record; Declare re-adds the edited one.
        self._on_remove(rec.get("key"))
        self.set_hint(
            "Loaded into the form — adjust and press Declare."
        )

    def _prefill(self, t: str, params: dict) -> None:
        for key, w in self._fields.get(t, {}).items():
            if key == "_dofs":
                vals = params.get("dofs") or []
                for i, c in enumerate(w):
                    c.setChecked(bool(vals[i]) if i < len(vals) else False)
                continue
            if key not in params:
                continue
            val = params[key]
            if isinstance(w, dict):
                for i, a in enumerate("xyz"):
                    try:
                        w[a].setValue(float(val[i]))
                    except Exception:
                        pass
            else:
                cls = type(w).__name__
                try:
                    if cls == "QCheckBox":
                        w.setChecked(bool(val))
                    elif cls == "QLineEdit":
                        w.setText(str(val))
                    else:
                        w.setValue(float(val))
                except Exception:
                    pass
