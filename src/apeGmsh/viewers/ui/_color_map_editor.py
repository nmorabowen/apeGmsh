"""ColorMapEditor — dock widget for editing the active layer's LUT.

Binds to one :class:`apeGmsh.viewers.core._lut_manager.LUT` at a time.
The active LUT is set by :meth:`bind_layer` — typically called from a
``ActiveObjects.activeLayerChanged`` subscription in ``ResultsViewer``.

Widgets (top-to-bottom):

* **Header** — the currently bound array name (or an empty-state hint).
* **Preset combo** — 10 curated colormaps (viridis, plasma, …, jet).
* **Range row** — vmin / vmax spinboxes + "Fit to data" button.
* **Log scale** — checkbox.
* **Show scalar bar** — checkbox.
* **Stops preview** — a horizontal gradient bar painted from the LUT's
  current preset.

When unbound, all editors are disabled and the header reads
``"No diagram selected"``. The dock is meant to stay visible at all
times; binding state changes don't show/hide the panel.

Plan 06 step 3: standalone widget + LUT binding. The viewer wiring
(extension-dock registration + ActiveObjects subscription) lands in
step 4. The widget is unit-testable in isolation against a bare LUT.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


# Whitelisted presets shown in the combo — same list as
# ``core._lut_manager.PRESETS`` (re-exported here so the UI module
# doesn't reach across to /core for a constant).
_COMBO_PRESETS: tuple[str, ...] = (
    "viridis", "plasma", "cividis", "magma", "inferno",
    "coolwarm", "RdBu", "Spectral", "turbo", "jet",
)


class _StopsPreview:
    """Small horizontal gradient bar painted from a LUT's color stops.

    Not a ``QWidget`` subclass — composes one via a custom paintEvent
    closure so the parent editor can lay it out like any normal widget.
    """

    def __init__(self, parent: Any = None) -> None:
        QtWidgets, QtCore, QtGui = _qt()
        self._stops: list = []
        self._widget = QtWidgets.QWidget(parent)
        self._widget.setMinimumHeight(18)
        self._widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self._widget.paintEvent = self._paint_event    # type: ignore[method-assign]

    @property
    def widget(self) -> Any:
        return self._widget

    def set_stops(self, stops: list) -> None:
        """Stops are a list of ``(t, (r, g, b))`` with ``t`` in [0, 1]
        and rgb channels in [0, 1]. Empty list paints a flat gray bar.
        """
        self._stops = list(stops)
        self._widget.update()

    def _paint_event(self, event: Any) -> None:    # noqa: ARG002
        _, _, QtGui = _qt()
        painter = QtGui.QPainter(self._widget)
        rect = self._widget.rect()
        if not self._stops:
            painter.fillRect(rect, QtGui.QColor(80, 80, 80))
            painter.end()
            return
        gradient = QtGui.QLinearGradient(rect.left(), 0, rect.right(), 0)
        for t, (r, g, b) in self._stops:
            color = QtGui.QColor(
                max(0, min(255, int(r * 255))),
                max(0, min(255, int(g * 255))),
                max(0, min(255, int(b * 255))),
            )
            gradient.setColorAt(float(max(0.0, min(1.0, t))), color)
        painter.fillRect(rect, gradient)
        painter.setPen(QtGui.QColor(40, 40, 40))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.end()


class ColorMapEditor:
    """Editor for the currently-active layer's lookup table.

    Compose-don't-inherit: :attr:`widget` is the QWidget ready to be
    placed in a dock. The editor does not touch the viewer; it only
    reads from and writes to a LUT instance.

    Usage::

        editor = ColorMapEditor()
        dock_widget = editor.widget
        # ...later:
        editor.bind_layer(diagram)    # binds editor → diagram.lut
        # ...or to unbind:
        editor.bind_layer(None)
    """

    def __init__(self, *, parent: Any = None) -> None:
        QtWidgets, QtCore, _ = _qt()
        self._lut: Any = None
        self._diagram: Any = None
        self._lut_conn: Any = None
        # Guard flag — set True while the editor is writing to the LUT
        # so the resulting LUT.changed signal doesn't drive a refresh
        # that fights the user's input (and resets spinbox cursors).
        self._self_setting: bool = False

        widget = QtWidgets.QWidget(parent)
        widget.setObjectName("ColorMapEditor")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Header ──────────────────────────────────────────────────
        self._header = QtWidgets.QLabel("No diagram selected.")
        font = self._header.font()
        font.setBold(True)
        self._header.setFont(font)
        layout.addWidget(self._header)

        # ── Preset row ──────────────────────────────────────────────
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.addItems(list(_COMBO_PRESETS))
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo, 1)
        layout.addLayout(preset_row)

        # ── Stops preview ───────────────────────────────────────────
        self._stops_preview = _StopsPreview(parent=widget)
        layout.addWidget(self._stops_preview.widget)

        # ── Range row ───────────────────────────────────────────────
        range_row = QtWidgets.QHBoxLayout()
        range_row.setSpacing(6)
        range_row.addWidget(QtWidgets.QLabel("Min:"))
        self._vmin_spin = QtWidgets.QDoubleSpinBox()
        self._vmin_spin.setDecimals(6)
        self._vmin_spin.setRange(-1e30, 1e30)
        self._vmin_spin.setKeyboardTracking(False)    # commit on Enter/blur
        self._vmin_spin.valueChanged.connect(self._on_range_changed)
        range_row.addWidget(self._vmin_spin, 1)
        range_row.addWidget(QtWidgets.QLabel("Max:"))
        self._vmax_spin = QtWidgets.QDoubleSpinBox()
        self._vmax_spin.setDecimals(6)
        self._vmax_spin.setRange(-1e30, 1e30)
        self._vmax_spin.setKeyboardTracking(False)
        self._vmax_spin.valueChanged.connect(self._on_range_changed)
        range_row.addWidget(self._vmax_spin, 1)
        self._fit_btn = QtWidgets.QPushButton("Fit to data")
        self._fit_btn.setToolTip(
            "Auto-fit Min/Max to the active step's value range."
        )
        self._fit_btn.clicked.connect(self._on_fit_clicked)
        range_row.addWidget(self._fit_btn)
        layout.addLayout(range_row)

        # ── Toggles ─────────────────────────────────────────────────
        self._log_cb = QtWidgets.QCheckBox("Log scale")
        self._log_cb.toggled.connect(self._on_log_toggled)
        layout.addWidget(self._log_cb)

        self._bar_cb = QtWidgets.QCheckBox("Show scalar bar")
        self._bar_cb.toggled.connect(self._on_bar_toggled)
        layout.addWidget(self._bar_cb)

        layout.addStretch(1)
        self._widget = widget
        self._set_enabled(False)

    # ──────────────────────────────────────────────────────────────────
    # Public surface
    # ──────────────────────────────────────────────────────────────────

    @property
    def widget(self) -> Any:
        return self._widget

    def bind_layer(self, diagram: Any) -> None:
        """Bind the editor to ``diagram``'s LUT.

        ``diagram=None`` clears the binding. Diagrams without a ``lut``
        attribute (or whose ``lut`` is ``None``) are treated as the
        empty case. Re-binding to the same LUT is a no-op.
        """
        new_lut = getattr(diagram, "lut", None) if diagram is not None else None
        if new_lut is self._lut:
            self._diagram = diagram
            return
        # Disconnect previous LUT's signal first.
        self._disconnect_lut()
        self._diagram = diagram
        self._lut = new_lut
        if self._lut is not None:
            try:
                self._lut_conn = self._lut.changed.connect(
                    self._refresh_from_lut,
                )
            except Exception:
                self._lut_conn = None
        self._refresh_from_lut()

    def bind_lut(self, lut: Any) -> None:
        """Bind directly to a LUT instance (no diagram). Lower-level
        path used by tests and by callers that don't have a diagram
        reference — the ``Fit to data`` button is disabled in this
        mode since there's no source for auto-fit values."""
        if lut is self._lut:
            return
        self._disconnect_lut()
        self._diagram = None
        self._lut = lut
        if self._lut is not None:
            try:
                self._lut_conn = self._lut.changed.connect(
                    self._refresh_from_lut,
                )
            except Exception:
                self._lut_conn = None
        self._refresh_from_lut()

    # ──────────────────────────────────────────────────────────────────
    # Internal — widget ↔ LUT sync
    # ──────────────────────────────────────────────────────────────────

    def _disconnect_lut(self) -> None:
        if self._lut is None or self._lut_conn is None:
            self._lut_conn = None
            return
        try:
            self._lut.changed.disconnect(self._lut_conn)
        except (TypeError, RuntimeError):
            pass
        self._lut_conn = None

    def _refresh_from_lut(self) -> None:
        """Repopulate widget values from the bound LUT.

        Called on bind and on every ``LUT.changed`` emission. Guards
        against feedback loops via ``_self_setting`` — if the editor
        itself triggered the change, the refresh is a no-op.
        """
        if self._self_setting:
            return
        if self._lut is None:
            self._header.setText("No diagram selected.")
            self._set_enabled(False)
            self._stops_preview.set_stops([])
            return
        array_name = getattr(self._lut, "array_name", "?")
        self._header.setText(f"Color mapping — {array_name}")
        self._set_enabled(True)

        with _BlockSignals(
            self._preset_combo, self._vmin_spin, self._vmax_spin,
            self._log_cb, self._bar_cb,
        ):
            preset = getattr(self._lut, "preset", "viridis")
            idx = self._preset_combo.findText(preset)
            if idx >= 0:
                self._preset_combo.setCurrentIndex(idx)
            self._vmin_spin.setValue(float(getattr(self._lut, "vmin", 0.0)))
            self._vmax_spin.setValue(float(getattr(self._lut, "vmax", 1.0)))
            self._log_cb.setChecked(bool(getattr(self._lut, "log_scale", False)))
            self._bar_cb.setChecked(
                bool(getattr(self._lut, "show_scalar_bar", True))
            )

        # Stops follow the preset; sample the LUT for the gradient bar.
        try:
            stops = self._lut.color_stops(n=32)
        except Exception:
            stops = []
        self._stops_preview.set_stops(stops)

        # Fit-to-data is only meaningful when bound through a diagram.
        self._fit_btn.setEnabled(
            self._diagram is not None
            and hasattr(self._diagram, "autofit_clim_at_current_step")
        )

    def _set_enabled(self, enabled: bool) -> None:
        for w in (
            self._preset_combo,
            self._vmin_spin, self._vmax_spin,
            self._log_cb, self._bar_cb,
            self._fit_btn,
        ):
            w.setEnabled(enabled)

    # ──────────────────────────────────────────────────────────────────
    # Slot handlers (user → LUT)
    # ──────────────────────────────────────────────────────────────────

    def _on_preset_changed(self, name: str) -> None:
        if self._lut is None:
            return
        self._with_self_setting(lambda: self._lut.set_preset(name))
        # Refresh stops preview locally — the changed signal is
        # suppressed by _self_setting, so the bar wouldn't otherwise
        # update.
        try:
            self._stops_preview.set_stops(self._lut.color_stops(n=32))
        except Exception:
            pass

    def _on_range_changed(self, _value: float) -> None:    # noqa: ARG002
        if self._lut is None:
            return
        vmin = float(self._vmin_spin.value())
        vmax = float(self._vmax_spin.value())
        self._with_self_setting(lambda: self._lut.set_range(vmin, vmax))

    def _on_log_toggled(self, on: bool) -> None:
        if self._lut is None:
            return
        self._with_self_setting(lambda: self._lut.set_log_scale(bool(on)))

    def _on_bar_toggled(self, on: bool) -> None:
        if self._lut is None:
            return
        self._with_self_setting(
            lambda: self._lut.set_show_scalar_bar(bool(on)),
        )
        # The diagram's own scalar-bar machinery (ScalarBarSupport)
        # listens via a separate runtime override. Propagate so the
        # actual bar in the scene appears / disappears.
        if self._diagram is not None and hasattr(
            self._diagram, "set_show_scalar_bar",
        ):
            try:
                self._diagram.set_show_scalar_bar(bool(on))
            except Exception:
                pass

    def _on_fit_clicked(self) -> None:
        if self._diagram is None:
            return
        fit = getattr(self._diagram, "autofit_clim_at_current_step", None)
        if fit is None:
            return
        try:
            fit()
        except Exception:
            return
        # autofit_clim_at_current_step calls set_clim which routes
        # through the LUT — the LUT.changed signal repopulates the
        # spinboxes for free.

    def _with_self_setting(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` with ``_self_setting`` pinned so the LUT's emit
        doesn't trigger a refresh that would fight focus / cursor."""
        self._self_setting = True
        try:
            fn()
        finally:
            self._self_setting = False


class _BlockSignals:
    """Context manager that blocks signals on a set of QObjects.

    Mirrors ``QSignalBlocker`` semantics — restores prior block state
    in ``__exit__``. The constructor accepts varargs so call sites can
    block several widgets at once without nesting ``with`` statements.
    """

    def __init__(self, *objects: Any) -> None:
        self._objects = objects
        self._prior: list[bool] = []

    def __enter__(self) -> "_BlockSignals":
        self._prior = [obj.blockSignals(True) for obj in self._objects]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:    # noqa: ARG002
        for obj, prior in zip(self._objects, self._prior):
            obj.blockSignals(prior)


def make_color_map_editor_dock(
    *,
    dock_id: str = "dock_color_map_editor",
    title: str = "Color Mapping",
    default_area: str = "right",
    default_visible: bool = False,
    tabify_with: Optional[str] = None,
) -> "tuple[ColorMapEditor, Any]":
    """Construct a :class:`ColorMapEditor` + matching :class:`DockSpec`.

    Returns ``(editor, spec)`` so the caller can hold a reference to
    the editor (for :meth:`ColorMapEditor.bind_layer`) while passing
    the spec to ``ResultsWindow``'s ``extension_docks=`` argument.

    Hidden by default — discoverable via the View menu toggle. Step 4
    wires it into ``ResultsViewer``; in tests, construct the editor
    directly without the dock spec.
    """
    from ._dock_registry import DockSpec

    editor = ColorMapEditor()

    def _factory(parent: Any) -> Any:    # noqa: ARG001
        return editor.widget

    spec = DockSpec(
        dock_id=dock_id,
        title=title,
        factory=_factory,
        default_area=default_area,
        default_visible=default_visible,
        tabify_with=tabify_with,
    )
    return editor, spec
