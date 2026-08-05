"""InspectorPanel — outline-selection-driven context host (ADR 0088 D2).

One dock (``dock_results_inspector``) replaces the five-tab rotated
spine (Diagram / Geometry / Details / Display / Section planes): the
Outline is the single navigation spine and the Inspector is its
property editor. The panel is a stacked context host whose current
page follows what is selected:

* **empty**    — nothing selected: one hint line plus the same starter
  actions the empty-state HUD offers (ADR 0088 D3).
* **stage**    — a stage/step row: the stage readout (activation
  table), rendered from rows the owner supplies via
  ``stage_info_provider``.
* **geometry** — a geometry row: the GeometrySettingsPanel
  (deformation / threshold / scope / display sections).
* **diagram**  — a diagram row: the layer stack + New-layer card
  (DiagramSettingsTab) with the Color section (ColorMapEditor)
  dissolved beneath it — the standalone Color Mapping dock is gone.
* **plot**     — a plot row: plot info + "Show in Plots".
* **details**  — a pick / probe result: the readout content
  (DetailsPanel).

Dumb widget: it never reaches into the director. The owner
(ResultsViewer) drives ``show_*`` from selection events and supplies
the starter callbacks / stage rows / plot activation. Every context
obeys ADR 0087 INV-1 (no inner panel title — the dock's "Inspector"
title is the one name), INV-2 (controls only when they act on
something), INV-3 (sectioned forms live in the hosted panels).

Display and Section planes stay separate docks by design — they are
not properties of an outline selection (ADR 0088 D2 rationale).
"""
from __future__ import annotations

from typing import Any, Callable, Optional


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


class InspectorPanel:
    """Stacked selection-context host for the Inspector dock.

    Parameters
    ----------
    diagram_widget
        The layer-stack editor (``DiagramSettingsTab.widget``).
    color_widget
        The color-mapping editor (``ColorMapEditor.widget``) — mounted
        inside the diagram context per ADR 0088 D2 ("Color Mapping
        dissolves into the diagram context").
    geometry_widget
        ``GeometrySettingsPanel.widget``.
    details_widget
        The pick/probe readout content (``DetailsPanel.widget``).
    stage_info_provider
        ``provider(stage_id) -> list[(label, value)]`` — rows for the
        stage readout. ``None`` renders a hint-only stage page.
    on_show_plot
        Fired with the plot key when the user clicks "Show in Plots".
    on_primary / on_secondary
        Starter actions for the empty context — the SAME actions the
        empty-state HUD offers (ADR 0088 D3). Labels arrive later via
        :meth:`set_starter_labels`; an empty label hides its button.
    """

    def __init__(
        self,
        diagram_widget: Any,
        color_widget: Any,
        geometry_widget: Any,
        details_widget: Any,
        *,
        stage_info_provider: Optional[
            Callable[[str], "list[tuple[str, str]]"]
        ] = None,
        on_show_plot: Optional[Callable[[Any], None]] = None,
        on_primary: Optional[Callable[[], None]] = None,
        on_secondary: Optional[Callable[[], None]] = None,
    ) -> None:
        QtWidgets, QtCore = _qt()
        self._stage_info_provider = stage_info_provider
        self._on_show_plot = on_show_plot
        self._plot_key: Any = None

        widget = QtWidgets.QWidget()
        widget.setObjectName("InspectorPanel")
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        stack = QtWidgets.QStackedWidget()
        outer.addWidget(stack, stretch=1)
        self._stack = stack
        self._widget = widget

        # ── empty page — hint + starter actions ─────────────────────
        empty = QtWidgets.QWidget()
        empty_lay = QtWidgets.QVBoxLayout(empty)
        empty_lay.setContentsMargins(10, 10, 10, 10)
        empty_lay.setSpacing(8)
        hint = QtWidgets.QLabel("Select an item in the outline to edit it.")
        hint.setObjectName("InspectorEmptyHint")
        hint.setWordWrap(True)
        hint.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        empty_lay.addWidget(hint)
        self._btn_primary = QtWidgets.QPushButton(empty)
        self._btn_primary.setVisible(False)
        if on_primary is not None:
            self._btn_primary.clicked.connect(on_primary)
        empty_lay.addWidget(
            self._btn_primary,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )
        self._btn_secondary = QtWidgets.QPushButton(empty)
        self._btn_secondary.setVisible(False)
        if on_secondary is not None:
            self._btn_secondary.clicked.connect(on_secondary)
        empty_lay.addWidget(
            self._btn_secondary,
            alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
        )
        empty_lay.addStretch(1)
        self._page_empty = empty
        stack.addWidget(empty)

        # ── stage page — readout rows (rebuilt per show_stage) ──────
        stage = QtWidgets.QWidget()
        stage_lay = QtWidgets.QVBoxLayout(stage)
        stage_lay.setContentsMargins(10, 10, 10, 10)
        stage_lay.setSpacing(6)
        self._stage_form_host = QtWidgets.QWidget(stage)
        self._stage_form = QtWidgets.QFormLayout(self._stage_form_host)
        self._stage_form.setContentsMargins(0, 0, 0, 0)
        stage_lay.addWidget(self._stage_form_host)
        self._stage_hint = QtWidgets.QLabel(
            "No readout available for this stage.",
        )
        self._stage_hint.setObjectName("InspectorEmptyHint")
        self._stage_hint.setWordWrap(True)
        self._stage_hint.setVisible(False)
        stage_lay.addWidget(self._stage_hint)
        stage_lay.addStretch(1)
        self._page_stage = stage
        stack.addWidget(stage)

        # ── geometry page ───────────────────────────────────────────
        self._page_geometry = self._host_page(geometry_widget)
        stack.addWidget(self._page_geometry)

        # ── diagram page — layer stack + Color section ──────────────
        diagram = QtWidgets.QWidget()
        diagram_lay = QtWidgets.QVBoxLayout(diagram)
        diagram_lay.setContentsMargins(0, 0, 0, 0)
        diagram_lay.setSpacing(0)
        diagram_lay.addWidget(diagram_widget, stretch=1)
        if color_widget is not None:
            diagram_lay.addWidget(color_widget)
        self._page_diagram = diagram
        stack.addWidget(diagram)

        # ── plot page — info + "Show in Plots" ──────────────────────
        plot = QtWidgets.QWidget()
        plot_lay = QtWidgets.QVBoxLayout(plot)
        plot_lay.setContentsMargins(10, 10, 10, 10)
        plot_lay.setSpacing(8)
        self._plot_label = QtWidgets.QLabel("")
        self._plot_label.setWordWrap(True)
        plot_lay.addWidget(self._plot_label)
        self._btn_show_plot = QtWidgets.QPushButton("Show in Plots")
        self._btn_show_plot.clicked.connect(self._fire_show_plot)
        plot_lay.addWidget(self._btn_show_plot)
        plot_lay.addStretch(1)
        self._page_plot = plot
        stack.addWidget(plot)

        # ── details page — pick / probe readout ─────────────────────
        self._page_details = self._host_page(details_widget)
        stack.addWidget(self._page_details)

        self._context = "empty"
        stack.setCurrentWidget(self._page_empty)
        self._apply_page_size_policies(self._page_empty)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def widget(self) -> Any:
        return self._widget

    @property
    def current_context(self) -> str:
        """One of ``empty / stage / geometry / diagram / plot / details``."""
        return self._context

    def set_starter_labels(self, primary: str, secondary: str) -> None:
        """Labels for the empty-context starter buttons.

        Mirrors :class:`EmptyStateHUD` — an empty label hides that
        button (ADR 0087 INV-2: no dead controls)."""
        self._btn_primary.setText(primary)
        self._btn_primary.setVisible(bool(primary))
        self._btn_secondary.setText(secondary)
        self._btn_secondary.setVisible(bool(secondary))

    def show_empty(self) -> None:
        self._switch("empty", self._page_empty)

    def show_stage(self, stage_id: Any) -> None:
        """Stage readout context — rows come from ``stage_info_provider``."""
        # Rebuild the form from the provider's rows.
        while self._stage_form.rowCount():
            self._stage_form.removeRow(0)
        rows: "list[tuple[str, str]]" = []
        if self._stage_info_provider is not None:
            try:
                rows = list(self._stage_info_provider(stage_id) or [])
            except Exception:
                rows = []
        QtWidgets, _ = _qt()
        for label, value in rows:
            value_lbl = QtWidgets.QLabel(str(value))
            value_lbl.setWordWrap(True)
            self._stage_form.addRow(str(label), value_lbl)
        self._stage_form_host.setVisible(bool(rows))
        self._stage_hint.setVisible(not rows)
        self._switch("stage", self._page_stage)

    def show_geometry(self) -> None:
        self._switch("geometry", self._page_geometry)

    def show_diagram(self) -> None:
        self._switch("diagram", self._page_diagram)

    def show_plot(self, key: Any, label: str) -> None:
        self._plot_key = key
        self._plot_label.setText(label or str(key))
        self._switch("plot", self._page_plot)

    def show_details(self) -> None:
        self._switch("details", self._page_details)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _switch(self, context: str, page: Any) -> None:
        self._context = context
        self._apply_page_size_policies(page)
        self._stack.setCurrentWidget(page)

    def _apply_page_size_policies(self, current: Any) -> None:
        """Only the CURRENT page contributes to the stack's size hints.

        ``QStackedWidget`` reports the max minimum-size over ALL pages,
        so a wide hidden context (geometry sections, spin rows) would
        force a permanent horizontal scrollbar on every other context.
        The standard recipe: inactive pages get ``Ignored`` policies.
        """
        QtWidgets, _ = _qt()
        P = QtWidgets.QSizePolicy.Policy
        for i in range(self._stack.count()):
            page = self._stack.widget(i)
            if page is current:
                page.setSizePolicy(P.Preferred, P.Preferred)
            else:
                page.setSizePolicy(P.Ignored, P.Ignored)

    def _fire_show_plot(self) -> None:
        if self._on_show_plot is not None and self._plot_key is not None:
            try:
                self._on_show_plot(self._plot_key)
            except Exception:
                pass

    @staticmethod
    def _host_page(content: Any) -> Any:
        """Wrap a pre-built content widget in a zero-margin page."""
        QtWidgets, _ = _qt()
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        if content is not None:
            lay.addWidget(content, stretch=1)
        return page


__all__ = ["InspectorPanel"]
