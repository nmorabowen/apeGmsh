"""DetailsPanel — contextual right-rail panel under the plot pane (B++ §4.4).

Hidden when nothing is selected, or when a tree group is selected.
Renders inside the same column as the plot pane (right rail) and
shares its width.

For B2 the only contextual content is per-diagram styling — the
existing :class:`DiagramSettingsTab` widget is hosted here directly.
Stage / Probe / Plot details fill in as those tree groups become
first-class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..diagrams._base import Diagram
from ._layout_metrics import LAYOUT

if TYPE_CHECKING:
    from ._diagram_settings_tab import DiagramSettingsTab


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class DetailsPanel:
    """Selection-driven details panel.

    Parameters
    ----------
    settings_tab
        The existing :class:`DiagramSettingsTab` whose widget is
        re-hosted as the body for diagram selections.
    """

    def __init__(self, settings_tab: "DiagramSettingsTab") -> None:
        QtWidgets, QtCore = _qt()
        self._settings_tab = settings_tab

        widget = QtWidgets.QWidget()
        widget.setObjectName("DetailsPanel")
        # Note: the legacy 220-px height cap (B++ spec §4.4) was removed
        # when the panel moved into its own QDockWidget — the dock now
        # owns sizing, so the inner widget is free to fill the dock.
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

        # + Add layer button — sits in the header so it's reachable
        # from any state. Click → settings panel switches to creation
        # mode (no kind / no data pre-selected).
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

        # ── Body — re-host the DiagramSettingsTab widget ───────────
        # We reparent settings_tab.widget into our outer layout. The
        # DiagramSettingsTab subscribes to director.on_diagrams_changed
        # itself; nothing needs to change about its observer wiring.
        body_holder = QtWidgets.QWidget()
        body_lay = QtWidgets.QVBoxLayout(body_holder)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.addWidget(settings_tab.widget)
        outer.addWidget(body_holder, stretch=1)

        # Theme-driven styling lives in viewers/ui/theme.py.
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
        self._widget.setVisible(True)

    def show_stack(self) -> None:
        """Render every active layer as a stacked card in the details dock.

        Called when the outline's ``Diagram 1`` row is selected. Each
        registry diagram becomes one collapsible card with its own
        Data combo + style controls + delete button.
        """
        try:
            self._settings_tab.show_stack()
        except Exception:
            pass
        n = len(self._settings_tab._director.registry)
        self._meta_label.setText(f"Diagram 1 — {n} layer(s)")
        self._widget.setVisible(True)

    def clear(self) -> None:
        """Idle state — keep the panel visible (the + Add button lives here),
        but clear the body."""
        try:
            self._settings_tab.set_idle()
        except Exception:
            pass
        self._meta_label.setText("")
        # Panel stays visible — + Add layer is always available.
        self._widget.setVisible(True)

    def _on_add_clicked(self) -> None:
        """+ Add layer → stack mode with a pending creation card.

        When the active composition is the locked Geometry view (which
        doesn't accept layers), this auto-creates a new Diagram
        composition first and makes it active — *that* gets the new
        layer. Otherwise the layer is added to the currently-active
        composition.
        """
        try:
            mgr = self._settings_tab._director.compositions   # noqa: SLF001
            if not mgr.active_accepts_layers:
                # Spawn a new composition; it auto-becomes active.
                mgr.add(name="Diagram", make_active=True)
            self._settings_tab.show_stack()
            self._settings_tab.set_create_new(True)
        except Exception:
            pass
        active = self._settings_tab._director.compositions.active   # noqa: SLF001
        active_name = active.name if active is not None else "Diagram"
        self._meta_label.setText(f"{active_name} — adding layer")
        self._widget.setVisible(True)
