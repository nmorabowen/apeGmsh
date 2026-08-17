"""Mesh-view inspector page — the §9 Add / change / clear loop.

ADR 0098 S2: the inspector restates onto pane rows (amended 0088 D2).
For one selected mesh view it shows the pose (deform) and the seven
§4 slot rows on ONE row mechanism: an empty slot is an **Add**
action; an occupied slot is its editor (change quantity) plus
**Clear**. Every gesture writes the session record — the same
``view.contour = Contour(...)`` a script would run — and the
reconciler repaints; the page holds no picture state.

Timebox (per the plan): contour + deform carry the full editors this
slice was sized for; the other five categories ride the same row
mechanism with a single-token picker. Pickers offer only quantities
the realized stage records, so the discrete loop cannot ask for a
picture that refuses; a category with nothing recorded shows a
disabled Add — the §4 inapplicable-is-empty/disabled law at the
picker level, never an error. An occupied slot whose kind drew
nothing (empty scope, no surviving nodes) shows "(nothing to draw)"
from the last realization — same law, post-emission.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from qtpy import QtWidgets

from .._failures import safe_slot
from ._realize import _resolve_instant
from ._specs import _AXIS_SUFFIXES, _TENSOR_SUFFIXES

#: §4 line slot, v1 component vocabulary — the force half of the
#: beam-geometry axis table (``COMPONENT_TO_LOCAL_AXIS``; the strain
#: conjugates are a later widening).
_LINE_COMPONENTS = (
    "axial_force", "shear_y", "shear_z",
    "torsion", "bending_moment_y", "bending_moment_z",
)

_DEFORM_FIELDS = ("displacement", "velocity", "acceleration")


def _token_suffix(token: str) -> str:
    return token.rsplit("_", 1)[-1] if "_" in token else ""


class _SlotRow:
    """One §4 category as an inspector row.

    Empty → title + [Add] (disabled with a reason when the stage
    records nothing this category can draw). Occupied → editor +
    [Clear] + a status label the page feeds from the last
    realization.
    """

    def __init__(
        self,
        category: str,
        title: str,
        *,
        editor: "Optional[QtWidgets.QWidget]",
        sync_editor: Callable[[Any], None],
        make_default: Callable[[], Any],
        can_add: bool,
        write: Callable[[Optional[Any]], None],
    ) -> None:
        self.category = category
        self._sync_editor = sync_editor
        self._make_default = make_default
        self._write = write

        self.widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.widget)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)

        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(title)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        header.addWidget(label)
        self._status = QtWidgets.QLabel("")
        self._status.setEnabled(False)  # dim — informational, never an error
        header.addWidget(self._status)
        header.addStretch(1)
        self._button = QtWidgets.QPushButton("Add")
        self._button.setObjectName(f"session_slot_{category}_button")
        self._button.clicked.connect(safe_slot(self._on_button))
        header.addWidget(self._button)
        outer.addLayout(header)

        self._editor = editor
        if editor is not None:
            outer.addWidget(editor)

        if not can_add:
            self._button.setEnabled(False)
            self._button.setToolTip(
                "Nothing recorded that this slot can draw.",
            )
        self._occupied = False
        self.sync(None)

    def sync(self, record: Optional[Any]) -> None:
        """Project the session record onto the row (never writes)."""
        self._occupied = record is not None
        self._button.setText("Clear" if self._occupied else "Add")
        if self._occupied:
            self._button.setEnabled(True)
        if self._editor is not None:
            self._editor.setVisible(self._occupied)
            if self._occupied:
                self._sync_editor(record)
        if not self._occupied:
            self._status.setText("")

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def _on_button(self) -> None:
        if self._occupied:
            self._write(None)
        else:
            self._write(self._make_default())


class MeshInspectorPage:
    """The inspector page for ONE mesh view (§9)."""

    def __init__(self, session: Any, view: Any) -> None:
        self._session = session
        self._view = view
        self._syncing = False
        self._nodal, self._gauss = self._recorded_components()
        self._patterns = self._load_patterns()

        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName(f"session_inspector_{view.id}")
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(self._build_deform_row())
        self._rows: dict[str, _SlotRow] = {}
        for row in self._build_slot_rows():
            self._rows[row.category] = row
            layout.addWidget(row.widget)
        layout.addStretch(1)

        session.subscribe(self._on_session_tick)
        self.refresh()

    def dispose(self) -> None:
        self._session.unsubscribe(self._on_session_tick)

    # -- projection ----------------------------------------------------

    def refresh(self) -> None:
        """Sync every row from the view — the page is a projection, so
        a Python-side edit shows up here exactly like a click would."""
        if self._syncing:
            return
        self._syncing = True
        try:
            view = self._view
            deform = view.deform
            self._deform_check.setChecked(deform is not None)
            self._deform_field.setEnabled(deform is not None)
            self._deform_scale.setEnabled(deform is not None)
            if deform is not None:
                self._deform_field.setCurrentText(deform.field)
                self._deform_scale.setValue(
                    0.0 if deform.scale is None else float(deform.scale),
                )
            for category, row in self._rows.items():
                row.sync(view.slots.get(category))
        finally:
            self._syncing = False

    def set_realized(self, realized: Optional[Any]) -> None:
        """Feed the last realization back into the rows (ADR §4): an
        occupied slot that emitted no layer is shown as empty /
        "(nothing to draw)" — never as an error."""
        view = self._view
        if realized is None or realized.pane_id != view.id:
            for row in self._rows.values():
                row.set_status("")
            return
        emitted = {layer.key.split(":", 1)[1].split(".", 1)[0]
                   for layer in realized.layers}
        for category, row in self._rows.items():
            occupied = category in view.slots
            drew = category in emitted
            row.set_status(
                "(nothing to draw)" if occupied and not drew else "",
            )

    # -- deform row (the pose, §3 — never a slot) ----------------------

    def _build_deform_row(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 2, 0, 2)
        self._deform_check = QtWidgets.QCheckBox("Deform")
        font = self._deform_check.font()
        font.setBold(True)
        self._deform_check.setFont(font)
        self._deform_check.setObjectName("session_deform_check")
        self._deform_check.toggled.connect(safe_slot(self._on_deform_changed))
        row.addWidget(self._deform_check)
        self._deform_field = QtWidgets.QComboBox()
        self._deform_field.setObjectName("session_deform_field")
        self._deform_field.addItems(list(_DEFORM_FIELDS))
        self._deform_field.currentTextChanged.connect(
            safe_slot(self._on_deform_changed),
        )
        row.addWidget(self._deform_field)
        self._deform_scale = QtWidgets.QDoubleSpinBox()
        self._deform_scale.setObjectName("session_deform_scale")
        self._deform_scale.setRange(0.0, 1e12)
        self._deform_scale.setDecimals(3)
        self._deform_scale.setSpecialValueText("auto")
        self._deform_scale.valueChanged.connect(
            safe_slot(self._on_deform_changed),
        )
        row.addWidget(self._deform_scale)
        row.addStretch(1)
        return box

    def _on_deform_changed(self, *_args: Any) -> None:
        if self._syncing:
            return
        from apeGmsh.results.session import Deform

        if not self._deform_check.isChecked():
            self._view.deform = None
            return
        scale = self._deform_scale.value()
        self._view.deform = Deform(
            field=self._deform_field.currentText(),
            scale=None if scale == 0.0 else scale,
        )

    # -- slot rows -----------------------------------------------------

    def _build_slot_rows(self) -> "list[_SlotRow]":
        from apeGmsh.results.session import (
            Contour, Gauss, Line, Loads, Reactions, Sand, Vector,
        )

        contour_tokens = sorted(self._nodal | self._gauss)
        vector_tokens = sorted(
            {t for t in self._nodal if _token_suffix(t) in _AXIS_SUFFIXES}
            | {t for t in self._gauss
               if _token_suffix(t) in _TENSOR_SUFFIXES}
        )
        gauss_tokens = sorted(self._gauss)
        sand_tokens = sorted(self._nodal)
        line_tokens = [
            t for t in _LINE_COMPONENTS
            if t in self._nodal or t in self._gauss
        ]

        rows = [
            self._quantity_row(
                "contour", "Contour", contour_tokens, Contour,
                extra=("averaged", "unaveraged"),
            ),
            self._quantity_row("vector", "Vector", vector_tokens, Vector),
            self._quantity_row("gauss", "Gauss", gauss_tokens, Gauss),
            self._quantity_row(
                "line", "Line", line_tokens,
                lambda token: Line(component=token),
            ),
            self._quantity_row("sand", "Sand", sand_tokens, Sand),
            self._quantity_row(
                "loads", "Loads", list(self._patterns),
                lambda token: Loads(pattern=token),
            ),
        ]
        rows.append(_SlotRow(
            "reactions", "Reactions",
            editor=None,
            sync_editor=lambda record: None,
            make_default=Reactions,
            can_add=True,
            write=self._writer("reactions"),
        ))
        return rows

    def _quantity_row(
        self,
        category: str,
        title: str,
        tokens: "list[str]",
        make: Callable[[str], Any],
        *,
        extra: "Optional[tuple[str, ...]]" = None,
    ) -> _SlotRow:
        """One row whose editor is a token combo (+ an optional second
        combo — the contour's averaging). The §9 mechanism every
        category rides."""
        editor = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(editor)
        lay.setContentsMargins(12, 0, 0, 0)
        combo = QtWidgets.QComboBox()
        combo.setObjectName(f"session_slot_{category}_quantity")
        combo.addItems(tokens)
        lay.addWidget(combo, stretch=1)
        second: Optional[QtWidgets.QComboBox] = None
        if extra is not None:
            second = QtWidgets.QComboBox()
            second.setObjectName(f"session_slot_{category}_mode")
            second.addItems(list(extra))
            lay.addWidget(second)

        write = self._writer(category)

        def build() -> Any:
            record = make(combo.currentText())
            if second is not None:
                from dataclasses import replace
                record = replace(record, averaging=second.currentText())
            return record

        def on_changed(*_args: Any) -> None:
            if self._syncing:
                return
            write(build())

        combo.currentTextChanged.connect(safe_slot(on_changed))
        if second is not None:
            second.currentTextChanged.connect(safe_slot(on_changed))

        def sync_editor(record: Any) -> None:
            token = getattr(record, "quantity", None)
            if token is None:
                token = getattr(record, "component", None)
            if token is None:
                token = getattr(record, "pattern", None)
            if token is not None:
                combo.setCurrentText(str(token))
            if second is not None:
                second.setCurrentText(record.averaging)

        def make_default() -> Any:
            return build() if combo.count() else None

        return _SlotRow(
            category, title,
            editor=editor,
            sync_editor=sync_editor,
            make_default=make_default,
            can_add=bool(tokens),
            write=write,
        )

    def _writer(self, category: str) -> Callable[[Optional[Any]], None]:
        def write(record: Optional[Any]) -> None:
            setattr(self._view, category, record)
        return write

    # -- data ----------------------------------------------------------

    def _recorded_components(self) -> "tuple[set[str], set[str]]":
        """What the realized stage records — the pickers' vocabulary.

        The same instant law realize applies (§7), so the picker and
        the picture cannot disagree about which stage they mean.
        """
        results = self._session.results
        if results is None:
            return set(), set()
        try:
            stage_id, _step = _resolve_instant(
                self._session, self._view, results,
            )
            components = results.inspect.components(stage=stage_id)
        except Exception:
            return set(), set()
        return (
            set(components.get("nodes", ())),
            set(components.get("gauss", ())),
        )

    def _load_patterns(self) -> "tuple[str, ...]":
        results = self._session.results
        if results is None or results.fem is None:
            return ()
        try:
            from ..data import ViewerData

            data = ViewerData.from_fem(results.fem)
            return tuple(data.nodes.loads.patterns())
        except Exception:
            return ()

    def _on_session_tick(self) -> None:
        self.refresh()


class PanePlaceholderPage:
    """Inspector page for panes S2 does not edit yet (plot panes —
    S4-2). A named refusal, not a crash."""

    def __init__(self, text: str) -> None:
        self.widget = QtWidgets.QLabel(text)
        self.widget.setWordWrap(True)
        self.widget.setEnabled(False)

    def dispose(self) -> None:
        return None

    def set_realized(self, realized: Optional[Any]) -> None:
        return None


__all__ = ["MeshInspectorPage", "PanePlaceholderPage"]
