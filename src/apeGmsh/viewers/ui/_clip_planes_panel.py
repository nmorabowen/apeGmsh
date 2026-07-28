"""ClipPlanesPanel — the "Section planes" dock (ADR 0083 Part 4).

The panel is a **projection** of :class:`ClipPlaneSetController`: it
reads the controller to draw itself and calls the controller's mutators
to change anything. It holds no plane state of its own, so the gizmo
(slice 2) and a session restore move these widgets without any
panel-to-gizmo wiring.

Layout, top to bottom:

* **Add plane** — a button plus an orientation combo (X / Y / Z /
  current view normal).
* **The plane list** — one row per plane: active checkbox, name, flip,
  gizmo eye, delete. The eye drives per-plane gizmo visibility
  (``set_gizmo_visible``; live since slice 2 draws the gizmos).
* **The selected plane's editor** — orientation chips X / Y / Z /
  Custom (Custom expands three normal fields), then the offset.
* **Footer** — the two set-level toggles, and a disabled *Contour on
  cut face* pointing at slice 3 (render-time clipping leaves solids
  hollow at the cut; the placeholder says so where the user meets it).

The offset slider maps the **reference-geometry bbox** projected on the
plane normal. The numeric field beside it is deliberately not clamped
to that range: deformation scale and per-geometry offsets (ADR 0058
S3a) legitimately push rendered geometry outside the reference bbox,
and a slider that cannot reach it would be the defect, not the value.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


#: Orientation presets for the Add button and the per-plane chips.
_AXIS_NORMALS: "dict[str, tuple[float, float, float]]" = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}

_ADD_CHOICES = ("X", "Y", "Z", "Current view normal")

_SLIDER_TICKS = 1000

#: Wide enough that the numeric offset field never stops the user
#: reaching geometry the deform scale threw far outside the bbox, while
#: still being a finite Qt spinbox range.
_UNCLAMPED = 1.0e15


class ClipPlanesPanel:
    """Qt panel driving a :class:`ClipPlaneSetController`."""

    def __init__(self) -> None:
        QtWidgets, QtCore, _ = _qt()
        self._controller: Any = None
        self._bbox: Optional[Sequence[float]] = None
        self._view_normal: Optional[Callable[[], Any]] = None
        self._selected_id: Optional[str] = None
        # Set while the panel is writing widget state from controller
        # state — every signal handler bails on it, so a refresh can
        # never be mistaken for a user edit.
        self._updating = False
        # Panel state, not derived state (S1 review finding B3): while
        # the selected plane's normal is still axis-aligned, "the user
        # clicked Custom" exists only here — deriving it from the
        # normal collapsed the fields on every unrelated refresh.
        self._custom_expanded = False

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Add plane ───────────────────────────────────────────────
        add_row = QtWidgets.QHBoxLayout()
        self._btn_add = QtWidgets.QPushButton("Add plane")
        self._btn_add.clicked.connect(self._on_add)
        add_row.addWidget(self._btn_add)
        self._combo_add = QtWidgets.QComboBox()
        self._combo_add.addItems(list(_ADD_CHOICES))
        add_row.addWidget(self._combo_add, 1)
        layout.addLayout(add_row)

        # ── Plane list ──────────────────────────────────────────────
        self._list_host = QtWidgets.QWidget()
        self._list_layout = QtWidgets.QVBoxLayout(self._list_host)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(2)
        layout.addWidget(self._list_host)

        self._lbl_empty = QtWidgets.QLabel("No section planes yet.")
        self._lbl_empty.setStyleSheet("color: gray;")
        layout.addWidget(self._lbl_empty)

        # ── Selected plane ──────────────────────────────────────────
        self._group = QtWidgets.QGroupBox("Selected plane")
        group_layout = QtWidgets.QVBoxLayout(self._group)

        chip_row = QtWidgets.QHBoxLayout()
        chip_row.addWidget(QtWidgets.QLabel("Normal:"))
        self._chips: dict[str, Any] = {}
        self._chip_group = QtWidgets.QButtonGroup(self.widget)
        self._chip_group.setExclusive(True)
        for name in ("X", "Y", "Z", "Custom"):
            chip = QtWidgets.QToolButton()
            chip.setText(name)
            chip.setCheckable(True)
            chip.clicked.connect(
                lambda _checked=False, n=name: self._on_chip(n),
            )
            self._chip_group.addButton(chip)
            chip_row.addWidget(chip)
            self._chips[name] = chip
        chip_row.addStretch(1)
        group_layout.addLayout(chip_row)

        self._custom_host = QtWidgets.QWidget()
        custom_row = QtWidgets.QHBoxLayout(self._custom_host)
        custom_row.setContentsMargins(0, 0, 0, 0)
        self._normal_fields = []
        for axis in ("nx", "ny", "nz"):
            custom_row.addWidget(QtWidgets.QLabel(axis))
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(-1.0e6, 1.0e6)
            box.setDecimals(4)
            box.setSingleStep(0.1)
            # Commit on Enter / focus-out, not per keystroke: every
            # valueChanged fires CLIP_PLANES_CHANGED, whose UI-lane
            # refresh rewrites this very field between keystrokes and
            # eats the decimal point (S1 review finding B3).
            box.setKeyboardTracking(False)
            box.valueChanged.connect(self._on_custom_normal)
            custom_row.addWidget(box, 1)
            self._normal_fields.append(box)
        self._custom_host.setVisible(False)
        group_layout.addWidget(self._custom_host)

        offset_row = QtWidgets.QHBoxLayout()
        offset_row.addWidget(QtWidgets.QLabel("Offset:"))
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(0, _SLIDER_TICKS)
        self._slider.valueChanged.connect(self._on_slider)
        offset_row.addWidget(self._slider, 1)
        self._spin_offset = QtWidgets.QDoubleSpinBox()
        self._spin_offset.setRange(-_UNCLAMPED, _UNCLAMPED)
        self._spin_offset.setDecimals(6)
        # Same reason as the normal fields: commit on Enter/focus-out
        # so the UI-lane refresh cannot rewrite mid-typing (B3).
        self._spin_offset.setKeyboardTracking(False)
        self._spin_offset.valueChanged.connect(self._on_offset_value)
        offset_row.addWidget(self._spin_offset)
        group_layout.addLayout(offset_row)

        self._lbl_range = QtWidgets.QLabel("range: —")
        self._lbl_range.setStyleSheet("color: gray;")
        group_layout.addWidget(self._lbl_range)

        layout.addWidget(self._group)

        # ── Footer ──────────────────────────────────────────────────
        self._cb_apply = QtWidgets.QCheckBox("Apply cuts")
        self._cb_apply.setChecked(True)
        self._cb_apply.toggled.connect(self._on_apply_cuts)
        layout.addWidget(self._cb_apply)

        self._cb_gizmos = QtWidgets.QCheckBox("Show plane gizmos")
        self._cb_gizmos.setChecked(True)
        self._cb_gizmos.toggled.connect(self._on_show_gizmos)
        layout.addWidget(self._cb_gizmos)

        self._cb_cut_face = QtWidgets.QCheckBox("Contour on cut face")
        self._cb_cut_face.setEnabled(False)
        self._cb_cut_face.setToolTip(
            "Not yet available — render-time clipping leaves solids "
            "hollow at the cut. Interpolating the active scalar onto "
            "the cut face is a later slice.",
        )
        layout.addWidget(self._cb_cut_face)

        layout.addStretch(1)
        self.refresh()

    # ------------------------------------------------------------------
    # External
    # ------------------------------------------------------------------

    def bind(
        self,
        controller: Any,
        *,
        bbox: Optional[Sequence[float]] = None,
        view_normal: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Point the panel at a controller (and the slider's bbox).

        ``bbox`` is ``(xmin, ymin, zmin, xmax, ymax, zmax)`` of the
        reference geometry; ``view_normal`` returns the camera's current
        view direction for the "current view normal" add choice.
        """
        self._controller = controller
        self._bbox = list(bbox) if bbox is not None else None
        self._view_normal = view_normal
        self.refresh()

    def refresh(self) -> None:
        """Redraw every widget from the controller's current state."""
        self._updating = True
        try:
            self._rebuild_rows()
            self._sync_selected()
            ctl = self._controller
            self._btn_add.setEnabled(ctl is not None)
            self._combo_add.setEnabled(ctl is not None)
            self._cb_apply.setEnabled(ctl is not None)
            self._cb_gizmos.setEnabled(ctl is not None)
            if ctl is not None:
                self._cb_apply.setChecked(bool(ctl.apply_cuts))
                self._cb_gizmos.setChecked(bool(ctl.show_gizmos))
        finally:
            self._updating = False

    @property
    def selected_id(self) -> Optional[str]:
        """The plane whose editor is showing, or ``None``."""
        return self._selected_id

    def select(self, plane_id: Optional[str]) -> None:
        if plane_id != self._selected_id:
            self._custom_expanded = False
        self._selected_id = plane_id
        self.refresh()

    # ------------------------------------------------------------------
    # Internal — rendering
    # ------------------------------------------------------------------

    def _planes(self) -> list:
        return self._controller.planes() if self._controller is not None else []

    def _selected(self) -> Any:
        if self._controller is None or self._selected_id is None:
            return None
        return self._controller.plane(self._selected_id)

    def _rebuild_rows(self) -> None:
        QtWidgets, _, _ = _qt()
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)

        planes = self._planes()
        self._lbl_empty.setVisible(not planes)
        self._group.setVisible(bool(planes))
        ids = {p.plane_id for p in planes}
        if self._selected_id not in ids:
            self._selected_id = planes[0].plane_id if planes else None

        for plane in planes:
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QtWidgets.QCheckBox()
            cb.setToolTip("Cut with this plane")
            cb.setChecked(bool(plane.active))
            cb.toggled.connect(
                lambda checked, pid=plane.plane_id: self._on_active(pid, checked),
            )
            row_layout.addWidget(cb)

            name = QtWidgets.QRadioButton(plane.name)
            name.setChecked(plane.plane_id == self._selected_id)
            name.toggled.connect(
                lambda checked, pid=plane.plane_id: (
                    self.select(pid) if checked and not self._updating else None
                ),
            )
            row_layout.addWidget(name, 1)

            flip = QtWidgets.QToolButton()
            flip.setText("⇄")
            flip.setCheckable(True)
            flip.setChecked(bool(plane.flipped))
            flip.setToolTip("Reverse — keep the other half")
            flip.toggled.connect(
                lambda checked, pid=plane.plane_id: self._on_flip(pid, checked),
            )
            row_layout.addWidget(flip)

            eye = QtWidgets.QToolButton()
            eye.setText("👁")
            eye.setCheckable(True)
            eye.setChecked(bool(plane.gizmo_visible))
            eye.setToolTip("Show this plane's gizmo in the viewport")
            eye.toggled.connect(
                lambda checked, pid=plane.plane_id: self._on_gizmo_eye(
                    pid, checked,
                ),
            )
            row_layout.addWidget(eye)

            delete = QtWidgets.QToolButton()
            delete.setText("✕")
            delete.setToolTip("Delete this plane")
            delete.clicked.connect(
                lambda _checked=False, pid=plane.plane_id: self._on_delete(pid),
            )
            row_layout.addWidget(delete)

            self._list_layout.addWidget(row)

    def _sync_selected(self) -> None:
        plane = self._selected()
        self._group.setEnabled(plane is not None)
        if plane is None:
            self._lbl_range.setText("range: —")
            return
        axis = self._axis_of(plane.normal)
        if axis is None:
            self._custom_expanded = True
        show_custom = self._custom_expanded
        chip_on = "Custom" if show_custom else (axis or "Custom")
        for name, chip in self._chips.items():
            chip.setChecked(name == chip_on)
        self._custom_host.setVisible(show_custom)
        # A focused editor is being typed into — rewriting it here
        # would destroy the in-progress input (B3). It re-syncs the
        # moment it commits (its own signal ends in a refresh).
        for box, value in zip(self._normal_fields, plane.normal):
            if not box.hasFocus():
                box.setValue(float(value))

        lo, hi = self._offset_range(plane.normal)
        self._lbl_range.setText(f"range: [{lo:.6g}, {hi:.6g}]")
        if not self._spin_offset.hasFocus():
            self._spin_offset.setValue(float(plane.offset))
        self._spin_offset.setSingleStep(max((hi - lo) / 100.0, 1e-6))
        if not self._slider.isSliderDown():
            self._slider.setValue(self._offset_to_tick(plane.offset, lo, hi))

    # ------------------------------------------------------------------
    # Internal — geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_of(normal: Sequence[float]) -> Optional[str]:
        """The axis chip ``normal`` corresponds to, or ``None``."""
        for name, axis in _AXIS_NORMALS.items():
            if all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(normal, axis)):
                return name
        return None

    def _offset_range(self, normal: Sequence[float]) -> tuple[float, float]:
        """Signed-distance range of the reference bbox along ``normal``.

        The eight corners projected on the normal — the exact interval
        in which a plane of that orientation intersects the reference
        geometry, for any orientation rather than just the axis ones.
        """
        bbox = self._bbox
        if bbox is None or len(bbox) != 6:
            return (-1.0, 1.0)
        n = [float(c) for c in normal]
        projections = []
        for i in (0, 3):
            for j in (1, 4):
                for k in (2, 5):
                    projections.append(
                        bbox[i] * n[0] + bbox[j] * n[1] + bbox[k] * n[2],
                    )
        lo, hi = min(projections), max(projections)
        return (lo, hi) if hi > lo else (lo - 1.0, lo + 1.0)

    @staticmethod
    def _offset_to_tick(offset: float, lo: float, hi: float) -> int:
        if hi <= lo:
            return 0
        t = (float(offset) - lo) / (hi - lo)
        return int(round(min(max(t, 0.0), 1.0) * _SLIDER_TICKS))

    # ------------------------------------------------------------------
    # Internal — signal handlers
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        if self._controller is None:
            return
        choice = self._combo_add.currentText()
        normal = _AXIS_NORMALS.get(choice)
        if normal is None:
            normal = self._current_view_normal()
        # Land the new plane mid-geometry, not at world offset 0: for
        # a bbox that does not straddle the origin (georeferenced or
        # all-negative coordinates), offset 0 cuts nothing — or
        # everything (S1 review finding B4).
        lo, hi = self._offset_range(normal)
        plane = self._controller.add(normal, offset=(lo + hi) / 2.0)
        self._selected_id = plane.plane_id
        self.refresh()

    def _current_view_normal(self) -> tuple[float, float, float]:
        if self._view_normal is not None:
            try:
                n = self._view_normal()
                if n is not None:
                    return (float(n[0]), float(n[1]), float(n[2]))
            except Exception:
                pass
        return _AXIS_NORMALS["X"]

    def _on_active(self, plane_id: str, checked: bool) -> None:
        if self._updating or self._controller is None:
            return
        if not self._controller.set_active(plane_id, bool(checked)):
            # Refused (the seventh cutting plane) — put the checkbox
            # back so the panel never claims a cut the scene isn't
            # making.
            self.refresh()

    def _on_flip(self, plane_id: str, checked: bool) -> None:
        if self._updating or self._controller is None:
            return
        self._controller.set_flipped(plane_id, bool(checked))

    def _on_gizmo_eye(self, plane_id: str, checked: bool) -> None:
        if self._updating or self._controller is None:
            return
        self._controller.set_gizmo_visible(plane_id, bool(checked))

    def _on_delete(self, plane_id: str) -> None:
        if self._controller is None:
            return
        self._controller.remove(plane_id)
        self.refresh()

    def _on_chip(self, name: str) -> None:
        if self._updating or self._controller is None:
            return
        plane = self._selected()
        if plane is None:
            return
        if name == "Custom":
            self._custom_expanded = True
            self._custom_host.setVisible(True)
            return
        self._custom_expanded = False
        self._controller.set_normal(plane.plane_id, _AXIS_NORMALS[name])
        self.refresh()

    def _on_custom_normal(self, _value: float) -> None:
        if self._updating or self._controller is None:
            return
        plane = self._selected()
        if plane is None:
            return
        self._controller.set_normal(
            plane.plane_id, [b.value() for b in self._normal_fields],
        )

    def _on_slider(self, tick: int) -> None:
        if self._updating or self._controller is None:
            return
        plane = self._selected()
        if plane is None:
            return
        lo, hi = self._offset_range(plane.normal)
        offset = lo + (hi - lo) * (float(tick) / _SLIDER_TICKS)
        self._controller.set_offset(plane.plane_id, offset)
        self._updating = True
        try:
            self._spin_offset.setValue(offset)
        finally:
            self._updating = False

    def _on_offset_value(self, value: float) -> None:
        if self._updating or self._controller is None:
            return
        plane = self._selected()
        if plane is None:
            return
        self._controller.set_offset(plane.plane_id, float(value))
        lo, hi = self._offset_range(plane.normal)
        self._updating = True
        try:
            self._slider.setValue(self._offset_to_tick(value, lo, hi))
        finally:
            self._updating = False

    def _on_apply_cuts(self, checked: bool) -> None:
        if self._updating or self._controller is None:
            return
        self._controller.set_apply_cuts(bool(checked))

    def _on_show_gizmos(self, checked: bool) -> None:
        if self._updating or self._controller is None:
            return
        self._controller.set_show_gizmos(bool(checked))


def make_clip_planes_dock(
    *,
    dock_id: str = "dock_results_clip_planes",
    title: str = "Section planes",
    default_area: str = "right",
    default_visible: bool = False,
    tabify_with: Optional[str] = None,
) -> "tuple[ClipPlanesPanel, Any]":
    """Construct a :class:`ClipPlanesPanel` + matching :class:`DockSpec`.

    Returns ``(panel, spec)`` so the caller keeps the panel (to
    :meth:`ClipPlanesPanel.bind` it once the backend exists) while
    handing the spec to ``ResultsWindow``'s ``extension_docks=``.
    Hidden by default — the toolbar button and the View menu are how it
    is found.
    """
    from ._dock_registry import DockSpec

    panel = ClipPlanesPanel()

    def _factory(parent: Any) -> Any:    # noqa: ARG001
        return panel.widget

    spec = DockSpec(
        dock_id=dock_id,
        title=title,
        factory=_factory,
        default_area=default_area,
        default_visible=default_visible,
        tabify_with=tabify_with,
    )
    return panel, spec


__all__ = ["ClipPlanesPanel", "make_clip_planes_dock"]
