"""SessionPaneFrame — one tiled pane: a header above pane content.

ADR 0098 Amendment 1 A1.3::

    ┌ SessionPaneFrame ─────────────────────────────────────────┐
    │ SessionPaneHeader   h = DENSITY.row_h (22 / 28)           │
    │  ▦ Mesh view 1     [M][O][N][G] │ Pick(N|G)  (time)    ✕  │
    ├───────────────────────────────────────────────────────────┤
    │  pane content — QtInteractor (mesh) | chart (plot)        │
    └───────────────────────────────────────────────────────────┘

Three rules the frame exists to hold:

* **The pane owns its single title** — 0087 INV-1's stated exception
  for a surface that renders outside a dock. It is the ONLY title:
  nothing draws a second header inside the content.
* **Style buttons live here and nowhere else** (INV-MESH-4). They are
  view chrome, not slots; routing them through the inspector would
  cost a pane selection per toggle — 0088 D2's accepted
  one-context-at-a-time price — exactly during the gesture (comparing
  two panes) that N panes exist for. The mesh-view inspector page does
  not restate them: one state, one control.

* **The pick-target radio sits beside them** (§8, S4-1) for the same
  reason and one more: the radio aims THIS pane's clicks, and two
  panes are allowed to aim differently over the one selection set, so
  it must be readable and settable without first selecting the pane
  you are about to click in. It is separated from the style group and
  carries its own checked treatment, because "Nodes" here means *what
  a click hits*, not *what is drawn* — the same two words, two
  different knobs.

Everything a button writes goes to ``session.pane(id)`` — the same
``MeshStyle`` record, or ``pick_target``, a script would assign — and
never to another pane. The frame is a projection: it re-syncs from the
session on every tick and holds no state of its own.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

from qtpy import QtCore, QtWidgets

from .._failures import safe_slot
from ..ui._icon_factory import bind_button_glyph
from ..ui._layout_metrics import LAYOUT
from ._pane import BackendFactory, MeshPane

#: The four INV-MESH-4 buttons: (MeshStyle field, glyph, tooltip).
#: Order is the ADR's — Mesh / Outlines / Nodes / Gauss.
STYLE_BUTTONS: "tuple[tuple[str, str, str], ...]" = (
    ("mesh", "mesh", "Mesh — interior element grid on the surface"),
    ("outlines", "outlines", "Outlines — element / feature boundaries"),
    ("nodes", "nodes", "Nodes — node glyphs (required to click-pick nodes)"),
    ("gauss", "gauss", "Gauss — integration-point glyphs (required to "
                       "click-pick Gauss)"),
)

#: The §8 pick-target radio: (``MeshView.pick_target`` value, glyph,
#: tooltip). Exactly two — selection is nodes XOR Gauss; element and
#: fiber pick targets are retired in this window (an element is a
#: membership query, not a hit).
PICK_BUTTONS: "tuple[tuple[str, str, str], ...]" = (
    ("nodes", "nodes", "Pick target: Nodes — clicks and windows in "
                       "this view hit nodes"),
    ("gauss", "gauss", "Pick target: Gauss — clicks and windows in "
                       "this view hit integration points"),
)

#: Kind glyphs. Both are shipped roster entries: a mesh pane draws the
#: analysis mesh, a plot pane draws a curve between two ends.
_KIND_GLYPH = {"mesh": "mesh", "plot": "probe_line"}

_ICON_PX = 16
_HIT_PX = 22


class PlotPanePlaceholder(QtWidgets.QLabel):
    """Content of a plot pane until the chart lands (S4-2).

    A plot IS a pane from S3 on — it tiles, it activates, it closes,
    it carries a header — so the host must be able to build one. What
    is missing is only the chart widget, and saying so beats an empty
    frame that looks like a failed render.
    """

    def __init__(self, view: Any, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(
            f"The Qt chart for this {view.kind} plot lands at S4-2. Its "
            f"series realize today through results.plot / "
            f"session.realize.",
            parent,
        )
        self.setObjectName("SessionPanePlotPlaceholder")
        self.setWordWrap(True)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setEnabled(False)

    def dispose(self) -> None:
        return None


class SessionPaneFrame(QtWidgets.QFrame):
    """One pane of the host — header + content, for a mesh or a plot."""

    def __init__(
        self,
        session: Any,
        view: Any,
        *,
        on_activate: Optional[Callable[[str], None]] = None,
        on_close: Optional[Callable[[str], None]] = None,
        backend_factory: Optional[BackendFactory] = None,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        from apeGmsh.results.session import MeshView

        self._session = session
        self._view = view
        self._pane_id = view.id
        self._is_mesh = isinstance(view, MeshView)
        self._on_activate = on_activate
        self._on_close = on_close
        self._syncing = False
        self._active = False

        self.setObjectName("SessionPaneFrame")
        # The pane id rides a Qt property, not the objectName: per-pane
        # objectNames would break 0087 INV-7's one-QSS-block-per-
        # component rule, so tests address panes through the host
        # (Amendment 1 caution 8).
        self.setProperty("paneId", self._pane_id)
        self.setProperty("active", "false")
        self.setMinimumWidth(LAYOUT.pane_min_width)
        self.setMinimumHeight(LAYOUT.pane_min_height)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)  # room for the accent border
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        self._pane: Optional[MeshPane] = None
        if self._is_mesh:
            self._pane = MeshPane(
                session,
                view.id,
                backend_factory=backend_factory,
                defer_fn=defer_fn,
                parent=self,
            )
            content: QtWidgets.QWidget = self._pane
        else:
            content = PlotPanePlaceholder(view, parent=self)
        self._content = content
        outer.addWidget(content, stretch=1)

        # Clicking ANYWHERE in the pane makes it active (A1.5). A GL
        # surface consumes its own mouse events, so the filter is
        # installed on the widgets a click can land on rather than
        # relying on propagation to this frame.
        for widget in (self._header, content, getattr(
            self._pane, "surface", None,
        )):
            if widget is not None:
                widget.installEventFilter(self)

        session.subscribe(self._on_session_tick)
        self.refresh()

    # -- surface -------------------------------------------------------

    @property
    def pane_id(self) -> str:
        return self._pane_id

    @property
    def paneId(self) -> str:  # noqa: N802 — mirrors the Qt property
        """The Qt ``paneId`` property, as an attribute (criteria 1/2)."""
        return str(self.property("paneId"))

    @property
    def is_mesh(self) -> bool:
        return self._is_mesh

    @property
    def pane(self) -> Optional[MeshPane]:
        """The :class:`MeshPane` for a mesh pane; ``None`` for a plot."""
        return self._pane

    @property
    def content(self) -> QtWidgets.QWidget:
        return self._content

    @property
    def backend(self) -> Any:
        """The pane's render backend (``None`` for a plot pane)."""
        return None if self._pane is None else self._pane.backend

    @property
    def plotter(self) -> Any:
        """The pane's live plotter, for the toolbar's active-pane
        routing (criterion 20). ``None`` when the pane owns no surface
        — a plot pane, or an injected backend in the offscreen lane."""
        if self._pane is None:
            return None
        return self._pane.surface

    def style_button(self, name: str) -> QtWidgets.QToolButton:
        """One of the four style buttons, by ``MeshStyle`` field name."""
        return self._style_buttons[name]

    def pick_button(self, name: str) -> QtWidgets.QToolButton:
        """One of the two pick-target buttons, by ``pick_target`` value."""
        return self._pick_buttons[name]

    @property
    def close_button(self) -> QtWidgets.QToolButton:
        return self._close

    @property
    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool) -> None:
        """Render the active state — a 1 px ``accent`` border (0087
        D1.4: accent is reserved for focus/active)."""
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        self.setProperty("active", "true" if active else "false")
        # A dynamic property change needs an explicit re-polish for the
        # QSS attribute selector to re-evaluate.
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def activate(self) -> None:
        """Make this the active pane (what a click does)."""
        if self._on_activate is not None:
            self._on_activate(self._pane_id)

    def request_reconcile(self) -> None:
        """Forced repaint (theme / density) — no-op for a plot pane."""
        if self._pane is not None:
            self._pane.request_reconcile()

    def apply_navigation(self, token: str) -> None:
        """Re-apply a navigation convention — no-op for a plot pane."""
        if self._pane is not None:
            self._pane.apply_navigation(token)

    def dispose(self) -> None:
        """Detach from the session (idempotent)."""
        try:
            self._session.unsubscribe(self._on_session_tick)
        except Exception:
            pass
        if self._pane is not None:
            self._pane.dispose()

    # -- projection ----------------------------------------------------

    def refresh(self) -> None:
        """Sync the header from the session record (never writes)."""
        view = self._view
        name = view.name or view.id
        self._title.setText(name)
        # The label can be squeezed to nothing (see _build_header), so
        # the name has to survive somewhere legible.
        self._title.setToolTip(name)
        if not self._is_mesh:
            return
        self._syncing = True
        try:
            style = view.style
            for name, button in self._style_buttons.items():
                button.setChecked(bool(getattr(style, name)))
            for name, button in self._pick_buttons.items():
                button.setChecked(name == view.pick_target)
        finally:
            self._syncing = False

    # -- Qt ------------------------------------------------------------

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            self.activate()
        return False

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 — Qt API
        self.activate()
        super().mousePressEvent(event)

    # -- internals -----------------------------------------------------

    def _build_header(self) -> QtWidgets.QWidget:
        from ..ui.density import DENSITY

        header = QtWidgets.QFrame()
        header.setObjectName("SessionPaneHeader")
        header.setFixedHeight(DENSITY.current.row_h)
        row = QtWidgets.QHBoxLayout(header)
        row.setContentsMargins(4, 0, 2, 0)
        row.setSpacing(2)

        kind = QtWidgets.QToolButton()
        kind.setObjectName("SessionPaneKind")
        kind.setEnabled(False)  # an indicator, not a control
        bind_button_glyph(
            kind, _KIND_GLYPH["mesh" if self._is_mesh else "plot"],
            size=_ICON_PX,
        )
        row.addWidget(kind)

        self._title = QtWidgets.QLabel(self._view.name or self._view.id)
        self._title.setObjectName("SessionPaneTitle")
        # The title is the one elastic thing in the header, and it has
        # to be: six buttons, a separator and a close glyph already
        # spend most of A1.4's 240 px pane floor, and a QLabel that
        # refuses to shrink below its text would push the frame's
        # minimumSizeHint past that floor — which the Add gate and
        # required_extent() still compute with, so the gate would admit
        # a column the splitter cannot actually fit. Ignored policy +
        # a zero minimum lets the pane reach the floor; the full name
        # stays available as the tooltip.
        self._title.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._title.setMinimumWidth(0)
        row.addWidget(self._title)
        if not self._is_mesh:
            kind_word = QtWidgets.QLabel(self._view.kind)
            kind_word.setObjectName("SessionPaneKindWord")
            row.addWidget(kind_word)
        row.addStretch(1)

        self._style_buttons: dict[str, QtWidgets.QToolButton] = {}
        self._pick_buttons: dict[str, QtWidgets.QToolButton] = {}
        if self._is_mesh:
            # 0087 INV-2: the buttons do not act on a plot, so a plot
            # pane does not render them disabled — it does not render
            # them at all.
            for name, glyph, tooltip in STYLE_BUTTONS:
                self._style_buttons[name] = self._build_style_button(
                    row, name, glyph, tooltip,
                )
            row.addWidget(self._build_separator())
            self._pick_group = QtWidgets.QButtonGroup(header)
            self._pick_group.setExclusive(True)
            for name, glyph, tooltip in PICK_BUTTONS:
                button = self._build_pick_button(row, name, glyph, tooltip)
                self._pick_buttons[name] = button
                self._pick_group.addButton(button)
            row.addStretch(1)

        # Reserved: the per-pane instant badge (S4-3 when the time link
        # is off / the plot cursor). Built and empty so the header's
        # geometry does not shift when it starts rendering.
        self._time_badge = QtWidgets.QLabel("")
        self._time_badge.setObjectName("SessionPaneTimeBadge")
        self._time_badge.setVisible(False)
        row.addWidget(self._time_badge)

        self._close = QtWidgets.QToolButton()
        self._close.setObjectName("SessionPaneClose")
        self._close.setToolTip("Close this pane")
        bind_button_glyph(self._close, "close", size=_ICON_PX)
        self._close.clicked.connect(safe_slot(self._on_close_clicked))
        row.addWidget(self._close)

        self._header = header
        return header

    def _build_style_button(
        self, row: Any, name: str, glyph: str, tooltip: str,
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setObjectName("SessionPaneStyleButton")
        button.setProperty("styleField", name)
        button.setCheckable(True)
        button.setToolTip(tooltip)
        button.setFixedSize(_HIT_PX, _HIT_PX)
        bind_button_glyph(button, glyph, size=_ICON_PX)
        if name == "gauss" and not self._gauss_available():
            # 0087 INV-2 — a control that cannot act does not pretend
            # to: no Gauss composite in this instant means there are no
            # integration points to place.
            button.setEnabled(False)
            button.setToolTip(
                "No Gauss values recorded in this stage — there are no "
                "integration-point locations to draw.",
            )
        button.toggled.connect(safe_slot(self._on_style_toggled))
        row.addWidget(button)
        return button

    def _build_separator(self) -> QtWidgets.QWidget:
        """The hairline between "what is drawn" and "what a click
        hits" — the two groups share glyphs, so they must not read as
        one strip of six toggles."""
        line = QtWidgets.QFrame()
        line.setObjectName("SessionPaneHeaderSeparator")
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFixedWidth(1)
        return line

    def _build_pick_button(
        self, row: Any, name: str, glyph: str, tooltip: str,
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setObjectName("SessionPanePickButton")
        button.setProperty("pickTarget", name)
        button.setCheckable(True)
        button.setToolTip(tooltip)
        button.setFixedSize(_HIT_PX, _HIT_PX)
        bind_button_glyph(button, glyph, size=_ICON_PX)
        if name == "gauss" and not self._gauss_available():
            # 0087 INV-2 — aiming clicks at a cloud that cannot exist
            # in this stage is a control that cannot act.
            button.setEnabled(False)
            button.setToolTip(
                "No Gauss values recorded in this stage — there are no "
                "integration points to pick.",
            )
        button.toggled.connect(safe_slot(self._on_pick_target_toggled))
        row.addWidget(button)
        return button

    def _on_pick_target_toggled(self, *_args: Any) -> None:
        if self._syncing:
            return
        for name, button in self._pick_buttons.items():
            if button.isChecked():
                # Aims THIS view's clicks and nothing else: it neither
                # owns nor clears the session's one selection set (§8).
                self._view.pick_target = name
                return

    def _gauss_available(self) -> bool:
        from ._realize import recorded_components

        _nodal, gauss = recorded_components(self._session, self._view)
        return bool(gauss)

    def _on_style_toggled(self, *_args: Any) -> None:
        if self._syncing:
            return
        # ONE write of the whole record: MeshStyle is a frozen value and
        # the session ticks per assignment, so a per-field write would
        # cost four ticks for one gesture.
        self._view.style = replace(
            self._view.style,
            **{
                name: bool(button.isChecked())
                for name, button in self._style_buttons.items()
            },
        )

    def _on_close_clicked(self) -> None:
        if self._on_close is not None:
            self._on_close(self._pane_id)

    def _on_session_tick(self) -> None:
        self.refresh()


__all__ = [
    "PICK_BUTTONS", "PlotPanePlaceholder", "STYLE_BUTTONS",
    "SessionPaneFrame",
]
