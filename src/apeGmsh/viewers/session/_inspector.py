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

#: The Add-plane axis choices (ADR 0098 A7 §4.4). Deliberately the three
#: world axes and nothing else: "view normal" needs a camera, which the
#: inspector does not have and should not reach for — the gizmo's rotate
#: handle is how a plane gets an arbitrary pose.
_CLIP_AXES = (
    ("X", (1.0, 0.0, 0.0)),
    ("Y", (0.0, 1.0, 0.0)),
    ("Z", (0.0, 0.0, 1.0)),
)


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


class _ClipRow:
    """One section plane as an inspector row (ADR 0098 A7 / R1 §4.4).

    ``◈ Plane 1   [x] cut  👁  ⇄  🗑`` with the offset beside it. Every
    control writes ``MeshView.set_clip`` — the same call the gizmo drag
    makes — so the row and the handle in the viewport are two editors
    of one record rather than two copies of it.

    The row is built ONCE per ``plane_id`` and thereafter only synced.
    That is not an optimisation: a gizmo drag ticks the session on every
    mouse-move, so a page that rebuilt its rows would destroy and
    re-create the widget under the user's cursor at drag rate — the
    focus-flicker hazard ``_outline.py`` documents for pane rows, which
    is also why the plane list is not outline children.
    """

    def __init__(
        self,
        plane_id: str,
        on_change: "Callable[[str], None]",
        on_remove: "Callable[[str], None]",
    ) -> None:
        self.plane_id = plane_id
        self._on_change = on_change
        self._on_remove = on_remove
        self._syncing = False

        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName(f"SessionInspectorClip_{plane_id}")
        row = QtWidgets.QHBoxLayout(self.widget)
        row.setContentsMargins(8, 1, 0, 1)
        row.setSpacing(4)

        self.cut = QtWidgets.QCheckBox()
        self.cut.setObjectName(f"SessionInspectorClipCut_{plane_id}")
        self.cut.setToolTip("Cut with this plane (ViewClip.active).")
        self.cut.toggled.connect(safe_slot(self._changed))
        row.addWidget(self.cut)

        self.name = QtWidgets.QLabel("")
        row.addWidget(self.name, stretch=1)

        self.offset = QtWidgets.QDoubleSpinBox()
        self.offset.setObjectName(f"SessionInspectorClipOffset_{plane_id}")
        self.offset.setDecimals(3)
        self.offset.setRange(-1.0e9, 1.0e9)
        self.offset.setToolTip(
            "Signed distance from the origin along the plane normal."
        )
        self.offset.valueChanged.connect(safe_slot(self._changed))
        row.addWidget(self.offset)

        self.eye = QtWidgets.QToolButton()
        self.eye.setObjectName(f"SessionInspectorClipEye_{plane_id}")
        self.eye.setText("Gizmo")
        self.eye.setCheckable(True)
        self.eye.setToolTip("Show this plane's drag handle in the viewport.")
        self.eye.toggled.connect(safe_slot(self._changed))
        row.addWidget(self.eye)

        self.flip = QtWidgets.QToolButton()
        self.flip.setObjectName(f"SessionInspectorClipFlip_{plane_id}")
        self.flip.setText("Reverse")
        self.flip.setToolTip("Swap which half of the model survives.")
        self.flip.clicked.connect(safe_slot(self._flip))
        row.addWidget(self.flip)

        self.remove = QtWidgets.QToolButton()
        self.remove.setObjectName(f"SessionInspectorClipRemove_{plane_id}")
        self.remove.setText("Remove")
        self.remove.setToolTip("Delete this section plane.")
        self.remove.clicked.connect(safe_slot(self._remove))
        row.addWidget(self.remove)

        self._flipped = False

    # -- projection ----------------------------------------------------

    def sync(self, clip: Any, step: float) -> None:
        """Restate this row from the record. Never writes back."""
        self._syncing = True
        try:
            self._flipped = bool(clip.flipped)
            self.name.setText(clip.name)
            self.cut.setChecked(bool(clip.active))
            self.eye.setChecked(bool(clip.gizmo_visible))
            self.offset.setSingleStep(step)
            if self.offset.value() != float(clip.offset):
                self.offset.setValue(float(clip.offset))
            self.flip.setText("Reversed" if clip.flipped else "Reverse")
        finally:
            self._syncing = False

    def set_offset_range(self, lo: float, hi: float) -> None:
        self._syncing = True
        try:
            self.offset.setRange(lo, hi)
        finally:
            self._syncing = False

    # -- gestures ------------------------------------------------------

    def _changed(self, *_args: Any) -> None:
        if self._syncing:
            return
        self._on_change(self.plane_id)

    def _flip(self) -> None:
        if self._syncing:
            return
        self._flipped = not self._flipped
        self._on_change(self.plane_id)

    def _remove(self) -> None:
        self._on_remove(self.plane_id)

    @property
    def flipped(self) -> bool:
        return self._flipped


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
        # A7 (R1) — §3 clips are view state, not a §4 slot, so the
        # section sits below the slot rows rather than among them.
        self._clip_rows: dict[str, _ClipRow] = {}
        self._clip_bounds: Optional[tuple] = None
        layout.addWidget(self._build_clips_section())
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
            self._sync_clips()
            self._sync_time_row()
        finally:
            self._syncing = False

    def set_realized(self, realized: Optional[Any]) -> None:
        """Feed the last realization back into the rows (ADR §4): an
        occupied slot that emitted no layer is shown as empty /
        "(nothing to draw)" — never as an error."""
        view = self._view
        # The extent the offset spin-boxes are scaled against — the
        # SAME bounds the gizmo quad is sized from, so the slider and
        # the handle always talk about one geometry (0083 Part 1).
        self._clip_bounds = getattr(realized, "reference_bounds", None)
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

    # -- section planes (§3 clips, ADR 0083 machinery — never a slot) --

    def _build_clips_section(self) -> QtWidgets.QWidget:
        """The Section planes section (ADR 0098 A7 / R1 §4.4).

        R1's *reach* half. The gizmo restored the direct-manipulation
        gesture, but a plane that only ``view.add_clip(...)`` can create
        is the A6.1 "works from Python, unreachable in the window" row
        the parity gate exists to end — the old viewer's clip dock has
        exactly one caller, ``results_viewer.py:975``, so the session
        window had no way to make a plane at all.

        0087 INV-2 is why the offset controls live on the plane ROWS
        rather than in a fixed block: with no planes there is no offset
        field to disable, because there is nothing for it to be about.
        Empty renders the Add action and nothing else.
        """
        box = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(box)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(2)

        title = QtWidgets.QLabel("Section planes")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        outer.addWidget(title)

        self._clip_rows_host = QtWidgets.QWidget()
        self._clip_rows_layout = QtWidgets.QVBoxLayout(self._clip_rows_host)
        self._clip_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._clip_rows_layout.setSpacing(1)
        outer.addWidget(self._clip_rows_host)

        add = QtWidgets.QHBoxLayout()
        add.setContentsMargins(8, 0, 0, 0)
        add.setSpacing(4)
        self._clip_add = QtWidgets.QToolButton()
        self._clip_add.setObjectName("SessionInspectorClipAdd")
        self._clip_add.setText("Add plane")
        self._clip_add.setToolTip(
            "Cut this pane with a new section plane through the model."
        )
        self._clip_add.clicked.connect(safe_slot(self._on_clip_add))
        add.addWidget(self._clip_add)

        self._clip_axis = QtWidgets.QComboBox()
        self._clip_axis.setObjectName("SessionInspectorClipAxis")
        for label, normal in _CLIP_AXES:
            self._clip_axis.addItem(label, normal)
        self._clip_axis.setToolTip("Normal of the new plane.")
        add.addWidget(self._clip_axis)
        add.addStretch(1)
        outer.addLayout(add)
        return box

    def _sync_clips(self) -> None:
        """Reconcile the plane rows against the record, BY PLANE ID.

        Rows are created and destroyed only when the set of plane ids
        changes — never on an ordinary edit. A gizmo drag ticks the
        session on every mouse-move, and rebuilding the rows at that
        rate would pull the widget out from under the cursor.
        """
        clips = self._view.clips
        live = {clip.plane_id for clip in clips}

        for plane_id in [p for p in self._clip_rows if p not in live]:
            row = self._clip_rows.pop(plane_id)
            self._clip_rows_layout.removeWidget(row.widget)
            row.widget.setParent(None)
            row.widget.deleteLater()

        step = self._clip_step()
        for index, clip in enumerate(clips):
            row = self._clip_rows.get(clip.plane_id)
            if row is None:
                row = _ClipRow(
                    clip.plane_id, self._on_clip_changed, self._on_clip_remove,
                )
                self._clip_rows[clip.plane_id] = row
                self._clip_rows_layout.insertWidget(index, row.widget)
                lo, hi = self._clip_offset_range()
                row.set_offset_range(lo, hi)
            row.sync(clip, step)

        self._clip_rows_host.setVisible(bool(clips))

    def _clip_offset_range(self) -> "tuple[float, float]":
        """Offset limits from the pane's reference extent.

        A plane can only usefully sit within the model it cuts, and the
        bounds realize already resolved for the gizmo quad are the same
        extent. With no realization yet, stay wide open rather than
        clamp a scripted value the user can see in the record.
        """
        bounds = self._clip_bounds
        if bounds is None:
            return (-1.0e9, 1.0e9)
        lo = min(bounds[0], bounds[1], bounds[2])
        hi = max(bounds[3], bounds[4], bounds[5])
        span = max(hi - lo, 1.0)
        return (lo - span, hi + span)

    def _clip_step(self) -> float:
        bounds = self._clip_bounds
        if bounds is None:
            return 0.1
        span = max(
            bounds[3] - bounds[0], bounds[4] - bounds[1],
            bounds[5] - bounds[2], 1.0e-9,
        )
        return max(span / 100.0, 1.0e-6)

    def _on_clip_add(self) -> None:
        """§9: an empty thing is an Add action."""
        normal = self._clip_axis.currentData()
        bounds = self._clip_bounds
        if bounds is None:
            offset = 0.0
        else:
            # Through the middle of the model, so the new plane cuts
            # something the moment it appears — a plane that lands
            # outside the mesh reads as "Add did nothing".
            centre = (
                (bounds[0] + bounds[3]) / 2.0,
                (bounds[1] + bounds[4]) / 2.0,
                (bounds[2] + bounds[5]) / 2.0,
            )
            offset = sum(c * n for c, n in zip(centre, normal))
        self._view.add_clip(normal, offset=offset)

    def _on_clip_changed(self, plane_id: str) -> None:
        if self._syncing:
            return
        row = self._clip_rows.get(plane_id)
        if row is None:
            return
        try:
            self._view.set_clip(
                plane_id,
                active=row.cut.isChecked(),
                gizmo_visible=row.eye.isChecked(),
                offset=float(row.offset.value()),
                flipped=row.flipped,
            )
        except KeyError:
            # The plane went away between the click and the write —
            # the next sync drops the row.
            pass

    def _on_clip_remove(self, plane_id: str) -> None:
        try:
            self._view.remove_clip(plane_id)
        except KeyError:
            pass

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
