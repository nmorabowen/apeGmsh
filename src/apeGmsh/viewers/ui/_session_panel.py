"""SessionPanel — viewer-level settings dock for the post-solve viewer.

Hosts session-scoped controls that don't belong inside the model
(Outline) or per-diagram (Details) flows. Ships:

- Cosmetic global toggles (node IDs / element IDs labels).
- Sizing knobs (point size, line width).
- Theme picker.

The deformation editor previously lived here as a global modifier.
After the Geometry refactor it moved into each Geometry's details
panel, so deformation is now per-Geometry and each geometry can
carry a different warp. The same refactor moved the
``Show mesh / Show nodes / Opacity`` controls into the per-Geometry
Display section — those are also per-Geometry now.

Future additions land here as new sections (density, layout reset,
file actions, …).
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class SessionPanel:
    """Right-rail dock for viewer-level settings (visualization, theme).

    Subscribes to :data:`THEME` so the displayed theme stays in sync
    when something outside the panel switches it (e.g. an external
    keyboard shortcut, or another viewer instance updating settings).

    Visualization callbacks are optional — when omitted, the toggles
    still render but flipping them is a no-op. The
    :class:`ResultsViewer` wires them up to the substrate / node-cloud
    actors after constructing the panel.
    """

    def __init__(
        self,
        *,
        on_show_node_ids: Optional[Callable[[bool], None]] = None,
        on_show_element_ids: Optional[Callable[[bool], None]] = None,
        on_point_size: Optional[Callable[[float], None]] = None,
        on_line_width: Optional[Callable[[float], None]] = None,
        show_node_ids_initial: bool = False,
        show_element_ids_initial: bool = False,
        point_size_initial: float = 10.0,
        line_width_initial: float = 3.0,
    ) -> None:
        QtWidgets, QtCore = _qt()
        from .theme import THEME, PALETTES

        self._theme = THEME
        self._palettes = PALETTES
        self._on_show_node_ids = on_show_node_ids
        self._on_show_element_ids = on_show_element_ids
        self._on_point_size = on_point_size
        self._on_line_width = on_line_width

        widget = QtWidgets.QWidget()
        widget.setObjectName("SessionPanel")
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Labels section ─────────────────────────────────────────
        # Show-mesh / Show-nodes / Opacity moved to the per-Geometry
        # Display section. What stays here is the genuinely global
        # cosmetic state — node/element ID labels and point/line size.
        viz_label = QtWidgets.QLabel("Labels")
        viz_label.setStyleSheet("font-weight: 600;")
        outer.addWidget(viz_label)

        self._cb_show_node_ids = QtWidgets.QCheckBox("Show node IDs")
        self._cb_show_node_ids.setChecked(bool(show_node_ids_initial))
        self._cb_show_node_ids.toggled.connect(self._fire_show_node_ids)
        outer.addWidget(self._cb_show_node_ids)

        self._cb_show_element_ids = QtWidgets.QCheckBox("Show element IDs")
        self._cb_show_element_ids.setChecked(bool(show_element_ids_initial))
        self._cb_show_element_ids.toggled.connect(self._fire_show_element_ids)
        outer.addWidget(self._cb_show_element_ids)

        # Sizing knobs (mirror PreferencesTab) — hosted in their own
        # form so labels align cleanly under the show/hide toggles.
        viz_form = QtWidgets.QFormLayout()
        viz_form.setContentsMargins(0, 0, 0, 0)
        viz_form.setSpacing(6)

        self._sb_point_size = QtWidgets.QDoubleSpinBox()
        self._sb_point_size.setRange(0.1, 9999.0)
        self._sb_point_size.setSingleStep(1.0)
        self._sb_point_size.setDecimals(1)
        self._sb_point_size.setValue(float(point_size_initial))
        self._sb_point_size.valueChanged.connect(self._fire_point_size)
        viz_form.addRow("Point size", self._sb_point_size)

        self._sb_line_width = QtWidgets.QDoubleSpinBox()
        self._sb_line_width.setRange(0.1, 9999.0)
        self._sb_line_width.setSingleStep(0.5)
        self._sb_line_width.setDecimals(1)
        self._sb_line_width.setValue(float(line_width_initial))
        self._sb_line_width.valueChanged.connect(self._fire_line_width)
        viz_form.addRow("Line width", self._sb_line_width)

        outer.addLayout(viz_form)

        # Spacer between sections.
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        outer.addWidget(sep)

        # ── Theme picker ───────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self._theme_combo = QtWidgets.QComboBox()
        for name in sorted(PALETTES.keys()):
            self._theme_combo.addItem(name, name)
        # Reflect the current theme without firing the change signal.
        idx = self._theme_combo.findData(THEME.current.name)
        if idx >= 0:
            self._theme_combo.blockSignals(True)
            self._theme_combo.setCurrentIndex(idx)
            self._theme_combo.blockSignals(False)
        self._theme_combo.currentIndexChanged.connect(self._on_theme_chosen)
        form.addRow("Theme:", self._theme_combo)

        outer.addLayout(form)

        # Trailing stretch so controls pack at the top of the dock —
        # any spare vertical space ends up below, not between widgets.
        outer.addStretch(1)

        # Track external theme changes (e.g. another consumer calling
        # THEME.set_theme) so the combo stays accurate.
        self._unsub_theme = THEME.subscribe(self._on_theme_changed_externally)

        self._widget = widget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def set_point_size_callback(
        self, cb: Optional[Callable[[float], None]],
    ) -> None:
        """Late binding for the node-cloud size callback."""
        self._on_point_size = cb

    def set_line_width_callback(
        self, cb: Optional[Callable[[float], None]],
    ) -> None:
        """Late binding for the substrate edge-width callback."""
        self._on_line_width = cb

    def set_show_node_ids_callback(
        self, cb: Optional[Callable[[bool], None]],
    ) -> None:
        """Late binding for the node-IDs label toggle."""
        self._on_show_node_ids = cb

    def set_show_element_ids_callback(
        self, cb: Optional[Callable[[bool], None]],
    ) -> None:
        """Late binding for the element-IDs label toggle."""
        self._on_show_element_ids = cb

    def close(self) -> None:
        """Detach observers — call when the host window closes."""
        try:
            self._unsub_theme()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _fire_point_size(self, value: float) -> None:
        if self._on_point_size is not None:
            self._on_point_size(float(value))

    def _fire_line_width(self, value: float) -> None:
        if self._on_line_width is not None:
            self._on_line_width(float(value))

    def _fire_show_node_ids(self, checked: bool) -> None:
        if self._on_show_node_ids is not None:
            self._on_show_node_ids(bool(checked))

    def _fire_show_element_ids(self, checked: bool) -> None:
        if self._on_show_element_ids is not None:
            self._on_show_element_ids(bool(checked))

    def _on_theme_chosen(self, _idx: int) -> None:
        name = self._theme_combo.currentData()
        if name is None:
            return
        self._theme.set_theme(name)

    def _on_theme_changed_externally(self, palette) -> None:
        idx = self._theme_combo.findData(palette.name)
        if idx < 0 or idx == self._theme_combo.currentIndex():
            return
        self._theme_combo.blockSignals(True)
        self._theme_combo.setCurrentIndex(idx)
        self._theme_combo.blockSignals(False)
