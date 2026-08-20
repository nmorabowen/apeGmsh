"""Mesh-view inspector page — the §9 Add / change / clear loop.

ADR 0098 S2: the inspector restates onto pane rows (amended 0088 D2).
For one selected mesh view it shows the pose (deform) and the seven
§4 slot rows on ONE row mechanism: an empty slot is an **Add**
action; an occupied slot is its editor (change quantity) plus
**Clear**. Every gesture writes the session record — the same
``view.contour = Contour(...)`` a script would run — and the
reconciler repaints; the page holds no picture state.

A plot pane gets :class:`PlotInspectorPage` instead: no slots, but
the same discipline — its SERIES are the occupants, each row clears
one, and adding happens where the selection actions live (the
outline).

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
from ._realize import recorded_components, recorded_line_components
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
        self._line = recorded_line_components(
            self._session, self._view)
        self._patterns = self._load_patterns()

        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName(f"session_inspector_{view.id}")
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(self._build_deform_row())
        layout.addWidget(self._build_time_row())
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
            self._sync_time_row()
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
        # Filtered against the line_stations family, NOT nodal/gauss:
        # a line component never appears in either, so the old test was
        # always false and the row offered nothing. _LINE_COMPONENTS
        # supplies the display ORDER; what is recorded supplies the set.
        line_tokens = [t for t in _LINE_COMPONENTS if t in self._line]

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

    def _build_time_row(self) -> QtWidgets.QWidget:
        """§9's "pane time — set it here; the link ignores it".

        The control stays ENABLED while the link is on, because that
        sentence is literal: this is the instant the pane takes when
        the link comes off, and you set it here whenever. What changes
        with the link is only the note beside it, which says whether
        the value is currently the one on screen.
        """
        box = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        line = QtWidgets.QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(4)
        line.addWidget(QtWidgets.QLabel("Pane time"))

        self._time_stage = QtWidgets.QComboBox()
        self._time_stage.setObjectName("SessionInspectorPaneStage")
        for stage_id in self._time_stages():
            self._time_stage.addItem(stage_id, stage_id)
        self._time_stage.currentIndexChanged.connect(
            safe_slot(self._on_pane_time_changed),
        )
        line.addWidget(self._time_stage, stretch=1)

        self._time_step = QtWidgets.QSpinBox()
        self._time_step.setObjectName("SessionInspectorPaneStep")
        self._time_step.setMinimum(0)
        self._time_step.valueChanged.connect(
            safe_slot(self._on_pane_time_changed),
        )
        line.addWidget(self._time_step)

        self._time_clear = QtWidgets.QToolButton()
        self._time_clear.setText("Clear")
        self._time_clear.setToolTip(
            "Drop this pane's own instant (it then has none of its own)."
        )
        self._time_clear.clicked.connect(safe_slot(self._on_pane_time_clear))
        line.addWidget(self._time_clear)
        outer.addLayout(line)

        self._time_note = QtWidgets.QLabel("")
        self._time_note.setObjectName("SessionInspectorPaneTimeNote")
        self._time_note.setWordWrap(True)
        self._time_note.setEnabled(False)
        outer.addWidget(self._time_note)
        return box

    def _time_stages(self) -> "list[str]":
        results = self._session.results
        if results is None:
            return []
        try:
            return [
                str(s.id) for s in results.stages
                if getattr(s, "kind", None) != "mode"
            ]
        except Exception:
            return []

    def _sync_time_row(self) -> None:
        view = self._view
        own = view.time
        if own is not None:
            index = self._time_stage.findData(own.stage)
            if index >= 0:
                self._time_stage.setCurrentIndex(index)
            self._time_step.setValue(int(own.step))
        stage_id = self._time_stage.currentData()
        self._time_step.setMaximum(max(0, self._n_steps(stage_id) - 1))
        self._time_clear.setEnabled(own is not None)
        if view.is_mode_posed:
            note = (
                "This view is mode-posed, so it has no instant at all "
                "and is frozen under the link (§4/§7)."
            )
        elif self._session.time_linked:
            note = (
                "The time link is on, so this pane follows the "
                "scrubber and this value is ignored until you unlink."
            )
        elif own is None:
            note = "No instant of its own — this pane draws nothing timed."
        else:
            note = "This pane is on its own instant."
        self._time_note.setText(note)

    def _n_steps(self, stage_id: Any) -> int:
        results = self._session.results
        if stage_id is None or results is None:
            return 0
        try:
            return max(0, int(results.stage(str(stage_id)).n_steps))
        except Exception:
            return 0

    def _on_pane_time_changed(self, *_args: Any) -> None:
        if self._syncing:
            return
        stage_id = self._time_stage.currentData()
        if stage_id is None:
            return
        from apeGmsh.results.session import Instant

        n = self._n_steps(stage_id)
        if n <= 0:
            return
        step = max(0, min(int(self._time_step.value()), n - 1))
        self._view.time = Instant(str(stage_id), step)

    def _on_pane_time_clear(self) -> None:
        if not self._syncing:
            self._view.time = None

    def _recorded_components(self) -> "tuple[set[str], set[str]]":
        """What the realized stage records — the pickers' vocabulary."""
        return recorded_components(self._session, self._view)

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


class PlotInspectorPage:
    """Inspector page for one plot pane (§9, S4-2).

    A plot has no §4 slots, so the Add / change / clear loop reads
    differently here: the SERIES are the occupants. Each row names one
    curve and clears it; the whole set clears at once. Adding is not on
    this page — it is "New plot from selection" on the outline, because
    what a new series means is a SELECTION, and the outline is where
    the selection actions live (one control, one place).

    Like every page here it is a projection: it rebuilds from
    ``plot.series`` on the change tick and writes only through the
    record a script would assign.
    """

    def __init__(self, session: Any, plot: Any) -> None:
        self._session = session
        self._plot = plot
        self._syncing = False

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._heading = QtWidgets.QLabel()
        self._heading.setObjectName("SessionPlotInspectorHeading")
        self._heading.setWordWrap(True)
        layout.addWidget(self._heading)

        self._rows = QtWidgets.QVBoxLayout()
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(2)
        layout.addLayout(self._rows)

        self._clear_all = QtWidgets.QPushButton("Clear all series")
        self._clear_all.setObjectName("SessionPlotInspectorClearAll")
        self._clear_all.clicked.connect(safe_slot(self._on_clear_all))
        layout.addWidget(self._clear_all)

        self._status = QtWidgets.QLabel()
        self._status.setObjectName("SessionPlotInspectorStatus")
        self._status.setWordWrap(True)
        self._status.setEnabled(False)
        layout.addWidget(self._status)
        layout.addStretch(1)

        session.subscribe(self._on_tick)
        self.refresh()

    # -- projection ----------------------------------------------------

    def refresh(self) -> None:
        series = self._plot.series
        self._heading.setText(
            f"{self._plot.kind} plot — {len(series)} series"
        )
        self._syncing = True
        try:
            while self._rows.count():
                item = self._rows.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            for index, spec in enumerate(series):
                self._rows.addWidget(self._series_row(index, spec))
        finally:
            self._syncing = False
        self._clear_all.setEnabled(bool(series))

    def set_realized(self, realized: Optional[Any]) -> None:
        """What the chart last managed to draw (§4's post-emission law,
        restated for curves): a series that resolved to nothing, or a
        plot that refused, says so here rather than leaving the author
        to guess from empty axes."""
        if realized is None:
            self._status.setText(
                "(nothing drawn — see the message on the chart)"
                if self._plot.series else ""
            )
            return
        drawn = len(realized.series)
        specs = len(self._plot.series)
        self._status.setText(
            "" if drawn == specs
            else f"({drawn} curve(s) from {specs} source(s))"
        )

    def dispose(self) -> None:
        self._session.unsubscribe(self._on_tick)

    # -- internals -----------------------------------------------------

    def _series_row(self, index: int, spec: Any) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        line = QtWidgets.QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(4)
        label = QtWidgets.QLabel(_series_title(spec))
        label.setToolTip(_series_title(spec))
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        line.addWidget(label, stretch=1)
        clear = QtWidgets.QToolButton()
        clear.setObjectName("SessionPlotInspectorClear")
        clear.setText("✕")
        clear.setToolTip("Remove this series from the plot")
        clear.clicked.connect(safe_slot(lambda *_a, i=index: self._drop(i)))
        line.addWidget(clear)
        return row

    def _drop(self, index: int) -> None:
        if self._syncing:
            return
        series = list(self._plot.series)
        if not 0 <= index < len(series):
            return
        del series[index]
        # ONE write of the whole tuple: series is a value, and the
        # session ticks per assignment.
        self._plot.series = series

    def _on_clear_all(self) -> None:
        if self._plot.series:
            self._plot.series = ()

    def _on_tick(self) -> None:
        self.refresh()


def _series_title(spec: Any) -> str:
    """One series as a row label — the same words its curve carries in
    the chart legend, so the two are recognisably the same thing."""
    source = spec.source
    if source.kind == "node":
        where = f"node {source.key}"
    elif source.kind == "gauss":
        where = f"element {source.key[0]} gp {source.key[1]}"
    else:
        where = f"{source.kind.replace('_', ' ')} {source.key}"
    return f"{where} — {spec.quantity}"


class PanePlaceholderPage:
    """Inspector page for a pane with nothing to edit — the
    nothing-selected hint beside an empty host (0088 D2). A named
    statement, not a crash."""

    def __init__(self, text: str) -> None:
        self.widget = QtWidgets.QLabel(text)
        self.widget.setWordWrap(True)
        self.widget.setEnabled(False)

    def dispose(self) -> None:
        return None

    def set_realized(self, realized: Optional[Any]) -> None:
        return None


__all__ = ["MeshInspectorPage", "PanePlaceholderPage", "PlotInspectorPage"]
