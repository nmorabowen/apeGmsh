"""Per-diagram settings tab — styling controls for the active selection.

Reads the currently-selected diagram from the Diagrams tab and shows a
kind-specific control set:

* ``contour`` — clim min/max, auto-fit button, opacity slider, cmap combo
* ``deformed_shape`` — scale slider, show-undeformed toggle

When no diagram is selected, an empty-state message is shown.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from ..diagrams._base import Diagram

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


_CMAP_PRESETS = [
    "viridis", "plasma", "cividis", "magma", "inferno",
    "coolwarm", "RdBu", "Spectral", "turbo", "jet",
]


class DiagramSettingsTab:
    """Settings panel for the diagram selected in the Diagrams tab.

    Subscribes to the Director's ``on_diagrams_changed`` so the
    panel refreshes when the selected diagram is removed or replaced.
    The selection itself is set by ``set_selected(diagram)`` from the
    Diagrams tab when the user clicks a row.
    """

    def __init__(self, director: "ResultsDirector") -> None:
        QtWidgets, _ = _qt()
        self._director = director
        self._selected: Optional[Diagram] = None

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)

        self._title = QtWidgets.QLabel("No diagram selected.")
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        layout.addWidget(self._title)

        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 6, 0, 0)
        layout.addWidget(self._content, stretch=1)

        empty_hint = QtWidgets.QLabel(
            "Select a diagram in the Diagrams tab to edit its settings."
        )
        empty_hint.setWordWrap(True)
        empty_hint.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(empty_hint)
        self._empty_hint = empty_hint

        self._widget = widget

        director.subscribe_diagrams(self._on_diagrams_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def set_selected(self, diagram: Optional[Diagram]) -> None:
        """Switch to editing a different diagram (or none)."""
        self._selected = diagram
        self._rebuild()

    # ------------------------------------------------------------------
    # Internal — rebuild content for the selection
    # ------------------------------------------------------------------

    def _on_diagrams_changed(self) -> None:
        # If the selected diagram was removed, drop the reference.
        if self._selected is None:
            return
        active = self._director.registry.diagrams()
        if self._selected not in active:
            self._selected = None
            self._rebuild()

    def _rebuild(self) -> None:
        QtWidgets, _ = _qt()
        # Clear content layout
        while self._content_layout.count():
            child = self._content_layout.takeAt(0)
            w = child.widget() if child else None
            if w is not None:
                w.deleteLater()

        d = self._selected
        if d is None:
            self._title.setText("No diagram selected.")
            self._empty_hint.setVisible(True)
            self._content.setVisible(False)
            return

        self._title.setText(d.display_label())
        self._empty_hint.setVisible(False)
        self._content.setVisible(True)

        kind = d.kind
        if kind == "contour":
            self._build_contour_panel(d)
        elif kind == "deformed_shape":
            self._build_deformed_panel(d)
        elif kind == "line_force":
            self._build_line_force_panel(d)
        elif kind in ("fiber_section", "layer_stack", "gauss_marker"):
            self._build_color_panel(d)
        elif kind == "vector_glyph":
            self._build_vector_panel(d)
        elif kind == "spring_force":
            self._build_spring_panel(d)
        else:
            self._content_layout.addWidget(QtWidgets.QLabel(
                f"No settings UI for kind {kind!r} yet."
            ))

    # ------------------------------------------------------------------
    # Contour panel
    # ------------------------------------------------------------------

    def _build_contour_panel(self, d: Diagram) -> None:
        QtWidgets, QtCore = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # Cmap combo
        cmap_combo = QtWidgets.QComboBox()
        cmap_combo.setEditable(True)
        for name in _CMAP_PRESETS:
            cmap_combo.addItem(name)
        current_cmap = getattr(d, "_runtime_cmap", None) or d.spec.style.cmap
        cmap_combo.setCurrentText(current_cmap)
        cmap_combo.currentTextChanged.connect(
            lambda txt: self._safe_call(d.set_cmap, txt)
        )
        form.addRow("Colormap:", cmap_combo)

        # Clim
        clim = d.current_clim() if hasattr(d, "current_clim") else None
        lo_default, hi_default = clim if clim else (0.0, 1.0)

        lo_spin = QtWidgets.QDoubleSpinBox()
        lo_spin.setRange(-1e30, 1e30)
        lo_spin.setDecimals(6)
        lo_spin.setValue(float(lo_default))
        hi_spin = QtWidgets.QDoubleSpinBox()
        hi_spin.setRange(-1e30, 1e30)
        hi_spin.setDecimals(6)
        hi_spin.setValue(float(hi_default))

        def _commit_clim() -> None:
            self._safe_call(
                d.set_clim, float(lo_spin.value()), float(hi_spin.value()),
            )

        lo_spin.editingFinished.connect(_commit_clim)
        hi_spin.editingFinished.connect(_commit_clim)
        form.addRow("Clim min:", lo_spin)
        form.addRow("Clim max:", hi_spin)

        autofit_btn = QtWidgets.QPushButton("Auto-fit at current step")

        def _autofit() -> None:
            new_clim = self._safe_call(d.autofit_clim_at_current_step)
            if new_clim is not None:
                lo_spin.setValue(float(new_clim[0]))
                hi_spin.setValue(float(new_clim[1]))

        autofit_btn.clicked.connect(_autofit)
        self._content_layout.addWidget(autofit_btn)

        # Opacity slider
        opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        current_opacity = (
            getattr(d, "_runtime_opacity", None)
            if getattr(d, "_runtime_opacity", None) is not None
            else d.spec.style.opacity
        )
        opacity_slider.setValue(int(round(float(current_opacity) * 100)))
        opacity_slider.valueChanged.connect(
            lambda v: self._safe_call(d.set_opacity, v / 100.0)
        )
        form.addRow("Opacity:", opacity_slider)

    # ------------------------------------------------------------------
    # Deformed shape panel
    # ------------------------------------------------------------------

    def _build_deformed_panel(self, d: Diagram) -> None:
        QtWidgets, QtCore = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # Scale slider — log-ish via spinbox for ergonomics
        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(4)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        scale_spin.editingFinished.connect(
            lambda: self._safe_call(d.set_scale, float(scale_spin.value()))
        )
        form.addRow("Scale:", scale_spin)

        # Quick scale presets
        preset_row = QtWidgets.QHBoxLayout()
        for s in (1, 10, 100, 1000):
            btn = QtWidgets.QPushButton(f"× {s}")
            btn.setMaximumWidth(60)

            def _make_handler(value: float):
                def _handler() -> None:
                    scale_spin.setValue(value)
                    self._safe_call(d.set_scale, value)
                return _handler

            btn.clicked.connect(_make_handler(float(s)))
            preset_row.addWidget(btn)
        preset_row.addStretch(1)
        self._content_layout.addLayout(preset_row)

        # Show-undeformed checkbox
        chk = QtWidgets.QCheckBox("Show undeformed reference")
        style = d.spec.style
        current = (
            getattr(d, "_runtime_show_undeformed", None)
            if getattr(d, "_runtime_show_undeformed", None) is not None
            else getattr(style, "show_undeformed", True)
        )
        chk.setChecked(bool(current))
        chk.toggled.connect(
            lambda v: self._safe_call(d.set_show_undeformed, bool(v))
        )
        form.addRow("", chk)

    # ------------------------------------------------------------------
    # Line force panel
    # ------------------------------------------------------------------

    def _build_line_force_panel(self, d: Diagram) -> None:
        QtWidgets, QtCore = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # Scale
        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(6)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        scale_spin.editingFinished.connect(
            lambda: self._safe_call(d.set_scale, float(scale_spin.value()))
        )
        form.addRow("Scale:", scale_spin)

        # Quick scale presets
        preset_row = QtWidgets.QHBoxLayout()
        for s in (0.1, 1, 10, 100):
            btn = QtWidgets.QPushButton(f"× {s}")
            btn.setMaximumWidth(60)

            def _make_handler(value: float):
                def _handler() -> None:
                    new_scale = float(d.current_scale()) * value
                    scale_spin.setValue(new_scale)
                    self._safe_call(d.set_scale, new_scale)
                return _handler

            btn.clicked.connect(_make_handler(float(s)))
            preset_row.addWidget(btn)
        preset_row.addStretch(1)
        self._content_layout.addLayout(preset_row)

        # Fill axis
        axis_combo = QtWidgets.QComboBox()
        axis_combo.addItem("Auto (component default)", None)
        axis_combo.addItem("y_local", "y")
        axis_combo.addItem("z_local", "z")
        # Reflect runtime override
        runtime_axis = getattr(d, "_runtime_axis", None)
        for i in range(axis_combo.count()):
            if axis_combo.itemData(i) == runtime_axis:
                axis_combo.setCurrentIndex(i)
                break

        def _on_axis_change(idx: int) -> None:
            data = axis_combo.itemData(idx)
            if data is None:
                # "Auto" — clear the runtime override and re-attach so the
                # default mapping is recomputed.
                d._runtime_axis = None
                if d.is_attached and d._fem is not None:
                    plotter, scene = d._plotter, d._scene
                    d.detach()
                    d.attach(plotter, d._fem, scene)
                self._fire_render()
            else:
                self._safe_call(d.set_fill_axis, str(data))

        axis_combo.currentIndexChanged.connect(_on_axis_change)
        form.addRow("Fill axis:", axis_combo)

        # Flip sign
        flip_chk = QtWidgets.QCheckBox("Flip sign")
        runtime_flip = getattr(d, "_runtime_flip", None)
        style_flip = getattr(d.spec.style, "flip_sign", False)
        flip_chk.setChecked(bool(
            runtime_flip if runtime_flip is not None else style_flip
        ))
        flip_chk.toggled.connect(
            lambda v: self._safe_call(d.set_flip_sign, bool(v))
        )
        form.addRow("", flip_chk)

    # ------------------------------------------------------------------
    # Vector-glyph panel (scale + cmap/clim)
    # ------------------------------------------------------------------

    def _build_vector_panel(self, d: Diagram) -> None:
        QtWidgets, _ = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(6)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        scale_spin.editingFinished.connect(
            lambda: self._safe_call(d.set_scale, float(scale_spin.value()))
        )
        form.addRow("Scale:", scale_spin)

        # Color settings reuse the shared color panel
        self._build_color_panel(d)

    # ------------------------------------------------------------------
    # Spring-force panel (scale + direction)
    # ------------------------------------------------------------------

    def _build_spring_panel(self, d: Diagram) -> None:
        QtWidgets, _ = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(6)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        scale_spin.editingFinished.connect(
            lambda: self._safe_call(d.set_scale, float(scale_spin.value()))
        )
        form.addRow("Scale:", scale_spin)

        # Direction overrides — three spinboxes for (dx, dy, dz)
        dir_row = QtWidgets.QHBoxLayout()
        spins = []
        cur = getattr(d, "_direction", None)
        defaults = (
            (1.0, 0.0, 0.0) if cur is None
            else (float(cur[0]), float(cur[1]), float(cur[2]))
        )
        for axis_label, default in zip(("dx", "dy", "dz"), defaults):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-1e9, 1e9)
            sp.setDecimals(4)
            sp.setSingleStep(0.1)
            sp.setValue(default)
            spins.append(sp)
            dir_row.addWidget(QtWidgets.QLabel(axis_label))
            dir_row.addWidget(sp)

        def _commit_dir() -> None:
            self._safe_call(
                d.set_direction,
                (
                    float(spins[0].value()),
                    float(spins[1].value()),
                    float(spins[2].value()),
                ),
            )

        for sp in spins:
            sp.editingFinished.connect(_commit_dir)
        self._content_layout.addLayout(dir_row)

    # ------------------------------------------------------------------
    # Generic color/clim panel — used by fiber & layer diagrams which
    # share a common surface (cmap + clim + auto-fit).
    # ------------------------------------------------------------------

    def _build_color_panel(self, d: Diagram) -> None:
        QtWidgets, _ = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        cmap_combo = QtWidgets.QComboBox()
        cmap_combo.setEditable(True)
        for name in _CMAP_PRESETS:
            cmap_combo.addItem(name)
        current_cmap = (
            getattr(d, "_runtime_cmap", None)
            or getattr(d.spec.style, "cmap", "viridis")
        )
        cmap_combo.setCurrentText(current_cmap)
        cmap_combo.currentTextChanged.connect(
            lambda txt: self._safe_call(d.set_cmap, txt)
        )
        form.addRow("Colormap:", cmap_combo)

        clim = d.current_clim() if hasattr(d, "current_clim") else None
        lo_default, hi_default = clim if clim else (0.0, 1.0)
        lo_spin = QtWidgets.QDoubleSpinBox()
        lo_spin.setRange(-1e30, 1e30)
        lo_spin.setDecimals(6)
        lo_spin.setValue(float(lo_default))
        hi_spin = QtWidgets.QDoubleSpinBox()
        hi_spin.setRange(-1e30, 1e30)
        hi_spin.setDecimals(6)
        hi_spin.setValue(float(hi_default))

        def _commit() -> None:
            self._safe_call(
                d.set_clim, float(lo_spin.value()), float(hi_spin.value()),
            )

        lo_spin.editingFinished.connect(_commit)
        hi_spin.editingFinished.connect(_commit)
        form.addRow("Clim min:", lo_spin)
        form.addRow("Clim max:", hi_spin)

        autofit = QtWidgets.QPushButton("Auto-fit at current step")

        def _autofit() -> None:
            new_clim = self._safe_call(d.autofit_clim_at_current_step)
            if new_clim is not None:
                lo_spin.setValue(float(new_clim[0]))
                hi_spin.setValue(float(new_clim[1]))

        autofit.clicked.connect(_autofit)
        self._content_layout.addWidget(autofit)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_call(self, fn: Callable, *args, **kwargs) -> Any:
        try:
            result = fn(*args, **kwargs)
            self._fire_render()
            return result
        except Exception as exc:
            import sys
            print(
                f"[DiagramSettingsTab] {fn.__qualname__} raised: {exc}",
                file=sys.stderr,
            )
            return None

    def _fire_render(self) -> None:
        # The Director's render_callback isn't directly accessible; rely
        # on the registry's update path via a no-op step set when present.
        # For Phase 1 we just call the plotter's render via the Director's
        # registry-bound plotter.
        plotter = getattr(self._director.registry, "_plotter", None)
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass
