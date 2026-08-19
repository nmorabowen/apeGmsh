"""SessionPaneHost — N panes, one ``QDockWidget`` each.

ADR 0098 Amendment 3, which **reverses Amendment 1's A1.1**: the panes
were one ``SessionPaneHost`` splitter tree mounted as the window's
central widget, and that arrangement does not work and cannot be made
to work by sizing it. The centre is dropped on the first show (the host
paints orphaned at 640x480 on top of the docks), and forcing the host to
be truly central takes a Windows access violation on the first show —
``showMaximized()`` and a plain ``show()`` alike. One dock per pane
survives show, float, resize, re-dock and tabify (A3.1's table).

So each pane is its own dock panel::

    +---------+--------+--------+--------+-----------+
    | Outline | mesh-1 | mesh-2 | plot-3 | Inspector |
    |  dock   |  dock  |  dock  |  dock  |   dock    |
    +---------+--------+--------+--------+-----------+
    |                Time scrubber                   |
    +------------------------------------------------+

**Default placement is a row** — each new pane splits the previous one
horizontally. A1.2's ``T(N)`` tiling law retires with the splitters:
``T(N)`` existed to answer "where does pane N go" with no user input,
and Qt's dock layout answers the same question and then lets the user
drag, float or tabify it, which is what a viewer should do. The ratio
law, ``layout_ratios`` / ``apply_ratios`` / ``reset_tiling`` and the
stored ``panes/*`` ratio record retire with it — the arrangement now
rides the window's own ``saveState`` / ``restoreState``, keyed by each
dock's objectName (A3.3).

Once the user has tabbed two panes together, splitting the last one
puts the newcomer in THEIR tab group rather than in a fresh column —
Qt's rule, and the right one: the Add gesture raises the new pane
(A1.5), so it arrives in front, and View → Reset layout puts the row
back. Measured on the bench, not assumed.

A1.4's floors survive as dock minimum sizes and still gate Add; A1.5's
one-active-pane rule survives, expressed as dock focus (``raise_()``);
A1.3's pane frame survives unchanged as the dock's widget.

**Where the docks live.** The panes belong to ADR 0088 D1's partition —
outline left, panes in the middle, inspector right — so in a session
window they are added to the SHELL's ``QMainWindow`` (``dock_host``),
right of the outline (``anchor_dock``). With neither given the host is
its own dock host: it is a ``QMainWindow`` itself, which is what lets a
test drive a bare pane host with no shell around it.

**Panes are moved, never rebuilt.** A host that tore down and rebuilt
its panes would satisfy any placement claim while silently destroying
every camera — backend state, not session state, so nothing downstream
would catch it. The frame set is diffed against ``session.panes``; a
pane that survives keeps its frame, its backend and its dock.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from qtpy import QtCore, QtWidgets

from .._failures import safe_slot
from ..ui._layout_metrics import LAYOUT
from ._frame import SessionPaneFrame
from ._pane import BackendFactory

#: objectName prefix for a pane's dock (A3.3.2 — stable, derived from
#: the pane id, so ``saveState`` / ``restoreState`` can address it).
PANE_DOCK_PREFIX = "dock_session_pane_"

#: The zero-pane card's dock. It is also the placement anchor: the
#: first pane splits IT, so the panes' column keeps its seat in the
#: 0088 D1 partition across an empty session.
EMPTY_DOCK_NAME = "dock_session_panes_empty"


def _dock_title(pane: Any) -> str:
    """What a pane's dock title bar reads — its name, or its id."""
    return str(getattr(pane, "name", None) or pane.id)


def pane_dock_name(pane_id: str) -> str:
    """The stable objectName of ``pane_id``'s dock (A3.3.2).

    >>> pane_dock_name("mesh-1")
    'dock_session_pane_mesh-1'
    """
    return f"{PANE_DOCK_PREFIX}{pane_id}"


def required_extent(n: int) -> "tuple[int, int]":
    """``(width, height)`` px a row of ``n`` panes needs at the A1.4
    floors, dock separators included.

    Amendment 1 computed this over ``T(N)``'s grid. Default placement
    is now a row, so the arithmetic is restated against it — the gate
    has to measure the layout the Add actually produces.
    """
    if n <= 0:
        return 0, 0
    separator = LAYOUT.splitter_handle_width
    return (
        n * LAYOUT.pane_min_width + max(0, n - 1) * separator,
        LAYOUT.pane_min_height,
    )


class SessionPaneHostEmpty(QtWidgets.QFrame):
    """Zero panes is a valid session, so the host renders it (A1.4).

    The 0088 D3 empty-state-HUD idiom, INV-2 shape: one hint line, one
    action, zero enabled data controls.
    """

    def __init__(
        self,
        on_new_view: Optional[Callable[[], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SessionPaneHostEmpty")
        column = QtWidgets.QVBoxLayout(self)
        column.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        column.setSpacing(12)
        self._hint = QtWidgets.QLabel("No views in this session.")
        self._hint.setObjectName("SessionPaneHostEmptyHint")
        self._hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._hint)
        self._action = QtWidgets.QPushButton("New mesh view")
        self._action.setObjectName("SessionPaneHostEmptyAction")
        if on_new_view is not None:
            self._action.clicked.connect(safe_slot(on_new_view))
        column.addWidget(
            self._action, alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

    @property
    def action(self) -> QtWidgets.QPushButton:
        return self._action

    @property
    def hint(self) -> QtWidgets.QLabel:
        return self._hint


class SessionPaneHost(QtWidgets.QMainWindow):
    """The pane docks — a projection of ``session.panes`` (A3.2)."""

    def __init__(
        self,
        session: Any,
        *,
        dock_host: Optional[QtWidgets.QMainWindow] = None,
        anchor_dock: Optional[QtWidgets.QDockWidget] = None,
        backend_factory: Optional[BackendFactory] = None,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
        on_active_changed: Optional[Callable[[str], None]] = None,
        on_panes_changed: Optional[Callable[[], None]] = None,
        on_geometry_changed: Optional[Callable[[], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._backend_factory = backend_factory
        self._defer_fn = defer_fn
        self._on_active_changed = on_active_changed
        self._on_panes_changed = on_panes_changed
        self._on_geometry_changed = on_geometry_changed
        self._frames: "dict[str, SessionPaneFrame]" = {}
        self._docks: "dict[str, QtWidgets.QDockWidget]" = {}
        self._listed: "list[str]" = []
        self._active_pane_id: Optional[str] = None
        self._disposed = False

        self.setObjectName("SessionPaneHost")
        # No shell given: the host IS the dock host, so a bare pane host
        # is drivable on its own.
        self._dock_host: QtWidgets.QMainWindow = (
            dock_host if dock_host is not None else self
        )
        self._anchor = anchor_dock
        self._dock_host.installEventFilter(self)

        self._empty = SessionPaneHostEmpty(self._new_view)
        self._empty_dock = QtWidgets.QDockWidget("Views")
        self._empty_dock.setObjectName(EMPTY_DOCK_NAME)
        self._empty_dock.setWidget(self._empty)
        self._empty_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures,
        )
        self._empty_dock.setMinimumWidth(LAYOUT.pane_min_width)
        self._seat(self._empty_dock, after=self._anchor)

        session.subscribe(self._on_session_tick)
        self.refresh()

    # -- surface -------------------------------------------------------

    @property
    def pane_frames(self) -> "tuple[SessionPaneFrame, ...]":
        """Every pane frame, in ``session.panes`` order (criterion 2)."""
        return tuple(
            self._frames[p.id] for p in self._session.panes
            if p.id in self._frames
        )

    @property
    def pane_docks(self) -> "tuple[QtWidgets.QDockWidget, ...]":
        """Every pane's dock, in ``session.panes`` order.

        With zero panes this is the empty card's dock — it holds the
        panes' seat in the partition, so a caller sizing the middle
        column has something to size either way.
        """
        docks = tuple(
            self._docks[p.id] for p in self._session.panes
            if p.id in self._docks
        )
        return docks or (self._empty_dock,)

    @property
    def empty_card(self) -> SessionPaneHostEmpty:
        return self._empty

    @property
    def empty_dock(self) -> QtWidgets.QDockWidget:
        return self._empty_dock

    def frame(self, pane_id: str) -> SessionPaneFrame:
        """The frame of one pane — how tests address a pane (caution 8).

        A miss is a ``KeyError``: like ``session.pane``, a stale id
        must fail loudly rather than answer with another pane.
        """
        return self._frames[pane_id]

    def dock(self, pane_id: str) -> QtWidgets.QDockWidget:
        """The dock of one pane. A miss is a ``KeyError``, as above."""
        return self._docks[pane_id]

    @property
    def active_pane_id(self) -> Optional[str]:
        """The one active pane (A1.5) — ``None`` only with zero panes."""
        return self._active_pane_id

    def set_active(self, pane_id: Optional[str]) -> None:
        """Make ``pane_id`` active; ``None`` clears (zero panes).

        A1.5 expressed as dock focus (A3.2): the active pane's dock is
        raised, which is what brings it to the front of a tab group the
        user built by dropping one pane onto another.
        """
        if pane_id is not None and pane_id not in self._frames:
            return
        if pane_id == self._active_pane_id:
            return
        self._active_pane_id = pane_id
        for pid, frame in self._frames.items():
            frame.set_active(pid == pane_id)
        dock = None if pane_id is None else self._docks.get(pane_id)
        if dock is not None:
            dock.raise_()
        if pane_id is not None and self._on_active_changed is not None:
            self._on_active_changed(pane_id)

    # -- the Add gate (A1.4) -------------------------------------------

    def panes_extent(self) -> "tuple[int, int]":
        """``(width, height)`` the pane docks occupy together.

        Measured off the DOCKS, not off this widget: with a shell the
        docks live in the shell's window and this object is only their
        controller, so its own geometry says nothing about them.
        """
        rect: Any = None
        for dock in self.pane_docks:
            geometry = dock.geometry()
            rect = geometry if rect is None else rect.united(geometry)
        if rect is None:
            return 0, 0
        return rect.width(), rect.height()

    def can_add(self) -> "tuple[bool, str]":
        """Whether a row of ``N+1`` panes clears both floors at the
        current size.

        The gate constrains CREATION only: a session that arrives with
        more panes than fit — ``s.add_view()`` in a script, an S5
        snapshot restore — is docked anyway. The IR is truth and the
        host never refuses a pane that exists.
        """
        n = len(self._session.panes) + 1
        need_w, need_h = required_extent(n)
        have_w, have_h = self.panes_extent()
        if have_w >= need_w and have_h >= need_h:
            return True, ""
        if have_w < need_w:
            return False, (
                f"Pane {n} needs {need_w} px of viewport width "
                f"({LAYOUT.pane_min_width} px minimum per pane); the "
                f"host has {have_w}."
            )
        return False, (
            f"Pane {n} needs {need_h} px of viewport height "
            f"({LAYOUT.pane_min_height} px minimum per pane); the "
            f"host has {have_h}."
        )

    # -- placement (A3.2 / A3.3) ---------------------------------------

    def reset_placement(self) -> None:
        """Re-seat every dock at DEFAULT placement — the row, in
        ``session.panes`` order (View → Reset layout, A1.5).

        Also the seat-taking pass the shell runs after it mounts its
        outline: the anchor dock is hidden until content lands in it,
        and a split against a hidden dock is not worth trusting.
        """
        self._seat(self._empty_dock, after=self._anchor)
        previous: Optional[QtWidgets.QDockWidget] = self._empty_dock
        for pane in self._session.panes:
            dock = self._docks.get(pane.id)
            if dock is None:
                continue
            dock.setFloating(False)
            self._seat(dock, after=previous)
            dock.show()
            previous = dock
        self._sync_docks()

    def sync_docks(self) -> None:
        """Re-assert the dock chrome the session owns: the empty
        card's visibility and every dock's title.

        A ``restoreState`` carries the card's own saved visibility, and
        the card is not the user's to show or hide: it is on exactly
        when the session has no panes (A3.3.1 — the session is
        authoritative for existence, the saved arrangement is only
        advisory for placement).
        """
        self._sync_docks()

    # -- lifecycle -----------------------------------------------------

    def request_reconcile(self) -> None:
        """Forced repaint of every pane (theme / density change)."""
        for frame in self._frames.values():
            frame.request_reconcile()

    def apply_navigation(self, token: str) -> None:
        """Fan a navigation-convention change out to every pane.

        The panes own the interactors, so the window-level
        View → Navigation gesture lands here.
        """
        for frame in self._frames.values():
            frame.apply_navigation(token)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        try:
            self._session.unsubscribe(self._on_session_tick)
        except Exception:
            pass
        try:
            self._dock_host.removeEventFilter(self)
        except Exception:
            pass
        for pane_id in list(self._docks):
            self._drop_dock(pane_id)
        self._frames.clear()
        self._retire(self._empty_dock)

    # -- projection ----------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the dock set from ``session.panes``.

        Skips the whole pass when membership is unchanged — the session
        ticks on EVERY mutation (slot fills included), and re-seating on
        each would reparent live GL widgets during ordinary editing.
        """
        if self._disposed:
            return
        wanted = [p.id for p in self._session.panes]
        # Compared against the FRAME set, not ``pane_frames`` — that
        # property is already filtered by ``session.panes``, so a
        # removed pane would compare equal to itself and the frame
        # would never be torn down.
        if wanted == self._listed:
            return
        previous = self._last_seated()
        self._listed = list(wanted)

        for pane_id in [pid for pid in self._frames if pid not in wanted]:
            self._drop_dock(pane_id)
        for pane in self._session.panes:
            if pane.id in self._frames:
                previous = self._docks[pane.id]
                continue
            frame = SessionPaneFrame(
                self._session, pane,
                on_activate=self.set_active,
                on_close=self._close_pane,
                backend_factory=self._backend_factory,
                defer_fn=self._defer_fn,
            )
            self._frames[pane.id] = frame
            previous = self._add_dock(pane, frame, after=previous)
        self._sync_docks()

        if self._active_pane_id not in self._frames:
            # A1.5: one active pane at all times when N >= 1. The pane
            # that closed hands focus to the first surviving one.
            self._active_pane_id = None
            self.set_active(wanted[0] if wanted else None)
        if self._on_panes_changed is not None:
            self._on_panes_changed()

    # -- Qt ------------------------------------------------------------

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QtCore.QEvent.Type.Resize:
            # The Add gate is a function of the panes' extent (A1.4),
            # so both a window resize and a separator drag can flip it.
            if self._on_geometry_changed is not None:
                self._on_geometry_changed()
        elif kind == QtCore.QEvent.Type.Close and isinstance(
            obj, QtWidgets.QDockWidget,
        ):
            pane_id = self._pane_of(obj)
            if pane_id is not None:
                # The dock's own close button is the header glyph's
                # other face: it writes the SESSION and the dock
                # disappears on the tick, never the other way round.
                event.ignore()
                self._close_pane(pane_id)
                return True
        return super().eventFilter(obj, event)

    # -- internals -----------------------------------------------------

    def _seat(
        self,
        dock: QtWidgets.QDockWidget,
        *,
        after: Optional[QtWidgets.QDockWidget],
    ) -> None:
        """Place ``dock`` in the middle column, right of ``after``."""
        host = self._dock_host
        host.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock,
        )
        if after is not None and after is not dock:
            host.splitDockWidget(
                after, dock, QtCore.Qt.Orientation.Horizontal,
            )

    def _add_dock(
        self,
        pane: Any,
        frame: SessionPaneFrame,
        *,
        after: Optional[QtWidgets.QDockWidget],
    ) -> QtWidgets.QDockWidget:
        pane_id = pane.id
        dock = QtWidgets.QDockWidget(_dock_title(pane))
        dock.setObjectName(pane_dock_name(pane_id))
        dock.setWidget(frame)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
        QDW = QtWidgets.QDockWidget
        dock.setFeatures(
            QDW.DockWidgetFeature.DockWidgetMovable
            | QDW.DockWidgetFeature.DockWidgetFloatable
            | QDW.DockWidgetFeature.DockWidgetClosable
        )
        # A1.4's floors reach the dock as A3.2 says they should — but
        # through the FRAME, which already carries them, not through a
        # second pair of setMinimum* calls here. A dock derives its
        # minimum from its widget plus its title bar (measured: 240 x
        # 224 for a 240 x 200 frame), so restating them here would only
        # ever LOWER the real floor.
        dock.installEventFilter(self)
        self._docks[pane_id] = dock
        self._seat(dock, after=after)
        # A3.3.3, natively: ``restoreDockWidget`` claims this dock's
        # entry out of the state the window restored before any pane
        # existed. No entry — a pane the saved arrangement never saw —
        # returns False and leaves the default placement above standing.
        # An entry naming a pane THIS session does not have is claimed
        # by nobody and dropped. Neither is an error; neither prompts.
        try:
            self._dock_host.restoreDockWidget(dock)
        except Exception:
            pass
        dock.show()
        return dock

    def _drop_dock(self, pane_id: str) -> None:
        frame = self._frames.pop(pane_id, None)
        if frame is not None:
            frame.dispose()
        dock = self._docks.pop(pane_id, None)
        if dock is None:
            return
        dock.removeEventFilter(self)
        # Take the frame out FIRST: it is the dock's child, so deleting
        # the dock alone would take the pane down a path its own
        # dispose has already walked.
        dock.setWidget(None)
        if frame is not None:
            frame.setParent(None)
            frame.deleteLater()
        self._retire(dock)

    def _retire(self, dock: QtWidgets.QDockWidget) -> None:
        try:
            self._dock_host.removeDockWidget(dock)
        except Exception:
            pass
        dock.setParent(None)
        dock.deleteLater()

    def _last_seated(self) -> Optional[QtWidgets.QDockWidget]:
        """The dock a newly added pane should sit right of."""
        for pane in reversed(self._session.panes):
            dock = self._docks.get(pane.id)
            if dock is not None:
                return dock
        return self._empty_dock

    def _pane_of(self, dock: QtWidgets.QDockWidget) -> Optional[str]:
        for pane_id, known in self._docks.items():
            if known is dock:
                return pane_id
        return None

    def _sync_docks(self) -> None:
        self._empty_dock.setVisible(not self._frames)
        # The dock's title bar is the pane's drag handle, so it carries
        # the pane's name and follows a rename.
        for pane in self._session.panes:
            dock = self._docks.get(pane.id)
            if dock is not None:
                dock.setWindowTitle(_dock_title(pane))

    def _close_pane(self, pane_id: str) -> None:
        """The header's close glyph and the dock's own close button.

        The dock disappears because the SESSION lost the pane, never
        before: this writes the IR and the projection follows on the
        tick. Closing the last pane is allowed — it lands on the empty
        card.
        """
        try:
            self._session.remove_pane(pane_id)
        except KeyError:
            pass

    def _new_view(self) -> None:
        self._session.add_view()

    def _on_session_tick(self) -> None:
        self.refresh()
        if not self._disposed:
            self._sync_docks()


__all__ = [
    "EMPTY_DOCK_NAME",
    "PANE_DOCK_PREFIX",
    "SessionPaneHost",
    "SessionPaneHostEmpty",
    "pane_dock_name",
    "required_extent",
]
