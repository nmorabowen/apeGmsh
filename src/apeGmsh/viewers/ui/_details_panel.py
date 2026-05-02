"""DetailsPanel — contextual right-rail panel under the plot pane (B++ §4.4).

States:
- *Idle* — empty body, header only (+ Add layer button still useful
  via show_stack auto-create flow).
- *Geometry settings* — when the user picks a Geometry row in the
  outline. Hosts the GeometrySettingsPanel (deformation editor +
  name).
- *Stack* — when the user picks a Composition row. Hosts the
  DiagramSettingsTab in stack mode (every layer = one card).

Renders inside the same column as the plot pane (right rail) and
shares its width.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..diagrams._base import Diagram
from ._layout_metrics import LAYOUT

if TYPE_CHECKING:
    from ._diagram_settings_tab import DiagramSettingsTab
    from ._geometry_settings_panel import GeometrySettingsPanel


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class DetailsPanel:
    """Selection-driven details panel.

    Parameters
    ----------
    settings_tab
        The existing :class:`DiagramSettingsTab` whose widget is
        re-hosted as the body for composition (Diagram) selections.
    geometry_panel
        The GeometrySettingsPanel for Geometry-row selections.
    """

    def __init__(
        self,
        settings_tab: "DiagramSettingsTab",
        geometry_panel: "GeometrySettingsPanel",
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._settings_tab = settings_tab
        self._geometry_panel = geometry_panel

        widget = QtWidgets.QWidget()
        widget.setObjectName("DetailsPanel")
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ─────────────────────────────────────────────────
        header = QtWidgets.QFrame()
        header.setObjectName("DetailsHeader")
        header.setFixedHeight(LAYOUT.details_header_height)
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(10, 0, 10, 0)
        header_lay.setSpacing(6)

        title = QtWidgets.QLabel("DETAILS")
        title.setObjectName("DetailsHeaderLabel")
        header_lay.addWidget(title)

        self._btn_add = QtWidgets.QPushButton("+ Add layer")
        self._btn_add.setObjectName("DetailsAddButton")
        self._btn_add.setFlat(True)
        self._btn_add.setToolTip("Add a new diagram layer")
        self._btn_add.clicked.connect(self._on_add_clicked)
        header_lay.addWidget(self._btn_add)
        header_lay.addStretch(1)

        meta = QtWidgets.QLabel("")
        meta.setObjectName("DetailsHeaderMeta")
        header_lay.addWidget(meta)
        self._meta_label = meta

        outer.addWidget(header)

        # ── Body — stacked: stack-view for compositions, geometry
        #          panel for geometry, idle for nothing.
        self._body = QtWidgets.QStackedWidget()
        # Index 0: stack/diagram view (DiagramSettingsTab widget).
        self._body.addWidget(settings_tab.widget)
        # Index 1: geometry settings.
        self._body.addWidget(geometry_panel.widget)
        outer.addWidget(self._body, stretch=1)

        self._widget = widget
        self.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def show_diagram(self, diagram: Diagram) -> None:
        """Legacy single-layer view. v2 pivot prefers :meth:`show_stack`."""
        self._settings_tab.set_selected(diagram)
        self._meta_label.setText(diagram.display_label())
        self._body.setCurrentIndex(0)
        self._btn_add.setVisible(True)
        self._widget.setVisible(True)

    def show_stack(self) -> None:
        """Render the active composition's layers as a stacked card view."""
        try:
            self._settings_tab.show_stack()
        except Exception:
            pass
        comp_mgr = self._settings_tab._director.compositions  # noqa: SLF001
        active = comp_mgr.active if comp_mgr is not None else None
        n = len(self._settings_tab._director.registry)  # noqa: SLF001
        if active is not None:
            self._meta_label.setText(f"{active.name} — {n} layer(s)")
        else:
            self._meta_label.setText(f"Diagram — {n} layer(s)")
        self._body.setCurrentIndex(0)
        self._btn_add.setVisible(True)
        self._widget.setVisible(True)

    def show_geometry(self, geom_id: str) -> None:
        """Route the panel to the geometry editor for ``geom_id``."""
        self._geometry_panel.show_geometry(geom_id)
        geom = self._settings_tab._director.geometries.find(  # noqa: SLF001
            geom_id,
        )
        if geom is not None:
            self._meta_label.setText(f"{geom.name} — geometry")
        else:
            self._meta_label.setText("Geometry")
        # Hide + Add layer when a Geometry is selected — user must
        # pick (or create) a Diagram first.
        self._btn_add.setVisible(False)
        self._body.setCurrentIndex(1)
        self._widget.setVisible(True)

    def clear(self) -> None:
        """Idle state — keep + Add layer reachable (auto-creates a
        Diagram inside the active Geometry on click)."""
        try:
            self._settings_tab.set_idle()
        except Exception:
            pass
        self._meta_label.setText("")
        self._btn_add.setVisible(True)
        self._body.setCurrentIndex(0)
        self._widget.setVisible(True)

    def refresh_geometry(self) -> None:
        """Re-sync the geometry settings panel (e.g. after rename)."""
        try:
            self._geometry_panel.refresh()
        except Exception:
            pass

    def _on_add_clicked(self) -> None:
        """+ Add layer → ensure a composition exists in the active
        Geometry, then drop into stack mode with a pending creation
        card."""
        director = self._settings_tab._director  # noqa: SLF001
        geom_mgr = director.geometries
        geom = geom_mgr.active
        if geom is None:
            # No active geometry — bail; the outline tree should
            # always have one. Defensive only.
            return
        comp_mgr = geom.compositions
        if not comp_mgr.active_accepts_layers:
            comp_mgr.add(name="Diagram", make_active=True)
        try:
            self._settings_tab.show_stack()
            self._settings_tab.set_create_new(True)
        except Exception:
            pass
        active = comp_mgr.active
        active_name = active.name if active is not None else "Diagram"
        self._meta_label.setText(f"{active_name} — adding layer")
        self._btn_add.setVisible(True)
        self._body.setCurrentIndex(0)
        self._widget.setVisible(True)
