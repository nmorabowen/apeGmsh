"""PickReadoutHUD — top-left HUD overlay for the 3-D viewport (B++ §4.2).

A glass card parented to the QtInteractor's widget that displays the
most recent point-probe result: nearest FEM node id, snapped coords,
and one row per active-diagram component value at the current step.

Updates on two signals:

* :class:`ProbeOverlay.on_point_result` — refresh on every new pick.
* :class:`ResultsDirector.subscribe_step` — re-read values at the new
  step so the HUD tracks the time scrubber.

Replaces the right-dock ``InspectorTab``: same data path
(``Director.read_at_pick``), but always visible inside the viewport
where the user is already looking.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector
    from ..overlays.probe_overlay import PointProbeResult, ProbeOverlay


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class PickReadoutHUD:
    """Top-left glass card showing the latest pick + live values.

    Parameters
    ----------
    viewport_widget
        The widget over which the HUD floats (the QtInteractor's
        rendering widget).
    overlay
        :class:`ProbeOverlay` whose point-probe callback drives updates.
    director
        :class:`ResultsDirector` for step-change subscription and for
        re-reading values at the current step.
    """

    _MARGIN = 12
    _MAX_WIDTH = 240

    def __init__(
        self,
        viewport_widget: Any,
        overlay: "ProbeOverlay",
        director: "ResultsDirector",
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._viewport = viewport_widget
        self._overlay = overlay
        self._director = director
        self._last_node_id: Optional[int] = None

        widget = QtWidgets.QFrame(parent=viewport_widget)
        widget.setObjectName("PickReadoutHUD")
        widget.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        widget.setMaximumWidth(self._MAX_WIDTH)
        self._widget = widget

        lay = QtWidgets.QVBoxLayout(widget)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        self._header = QtWidgets.QLabel("(no pick)")
        self._header.setObjectName("PickReadoutHeader")
        lay.addWidget(self._header)

        self._coords = QtWidgets.QLabel("")
        self._coords.setObjectName("PickReadoutCoords")
        lay.addWidget(self._coords)

        self._values = QtWidgets.QLabel("")
        self._values.setObjectName("PickReadoutValues")
        self._values.setWordWrap(False)
        self._values.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse,
        )
        lay.addWidget(self._values)

        self._hint = QtWidgets.QLabel("⇧ click → add to plot")
        self._hint.setObjectName("PickReadoutHint")
        lay.addWidget(self._hint)

        # Hide the value/coord rows until the first pick — empty state
        # is just the header + hint.
        self._coords.hide()
        self._values.hide()

        # Chain into the overlay's existing point-result callback so we
        # cooperate with whoever owned it before us (typically
        # ProbePaletteHUD's status-bar handler).
        self._chain_point: Optional[Callable] = overlay.on_point_result
        overlay.on_point_result = self._on_point_result

        # Track the time scrubber.
        director.subscribe_step(self._on_step_changed)
        director.subscribe_stage(self._on_stage_changed)

        # Reposition on viewport resize via event filter.
        from ._viewport_hud import _ResizeFilter
        self._filter = _ResizeFilter(self.reposition)
        viewport_widget.installEventFilter(self._filter)

        widget.show()
        widget.raise_()
        self.reposition()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def reposition(self) -> None:
        """Move the HUD into the viewport's top-left corner."""
        self._widget.move(self._MARGIN, self._MARGIN)
        self._widget.raise_()

    # ------------------------------------------------------------------
    # Probe + step callbacks
    # ------------------------------------------------------------------

    def _on_point_result(self, result: "PointProbeResult") -> None:
        self._last_node_id = int(result.closest_node_id)
        self._render(
            node_id=int(result.closest_node_id),
            coord=tuple(float(c) for c in result.closest_coord),
            field_values=dict(result.field_values),
        )
        if self._chain_point is not None:
            try:
                self._chain_point(result)
            except Exception:
                pass

    def _on_step_changed(self, _step_index: int) -> None:
        self._refresh_values_for_last_pick()

    def _on_stage_changed(self, _stage_id: Any) -> None:
        # Stage switch invalidates the cached pick — different topology
        # may not even contain the same node ids in the same order.
        self._last_node_id = None
        self._header.setText("(no pick)")
        self._coords.hide()
        self._values.hide()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _refresh_values_for_last_pick(self) -> None:
        if self._last_node_id is None:
            return
        components = self._collect_active_components()
        values = self._director.read_at_pick(
            self._last_node_id, components,
        ) if components else {}
        self._render_values(values)

    def _render(
        self,
        *,
        node_id: int,
        coord: tuple[float, float, float],
        field_values: dict[str, float],
    ) -> None:
        self._header.setText(f"node {node_id}")
        self._coords.setText(
            f"({coord[0]:.4g}, {coord[1]:.4g}, {coord[2]:.4g})"
        )
        self._coords.show()
        self._render_values(field_values)

    def _render_values(self, values: dict[str, float]) -> None:
        if not values:
            self._values.setText(
                "(add diagrams to see values)",
            )
        else:
            lines = [f"{name} = {val:.6g}" for name, val in values.items()]
            self._values.setText("\n".join(lines))
        self._values.show()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_active_components(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for d in self._director.registry.diagrams():
            if not d.is_attached:
                continue
            comp = d.spec.selector.component
            if comp not in seen:
                seen.add(comp)
                out.append(comp)
        return out
