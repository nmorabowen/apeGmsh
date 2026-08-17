"""SessionPaneHost — N panes, auto-tiled, as the window's centre.

ADR 0098 Amendment 1. The central widget is a root
``QSplitter(Horizontal)`` of **columns**; each column is a
``QSplitter(Vertical)`` of **rows**; each row is one
:class:`~._frame.SessionPaneFrame`. Depth is exactly two — the
geometry ``tests/viewers/test_pane_host_probe.py`` proved on both
oracles.

The host is a projection of ``session.panes`` on the S2 discipline:
the session is truth, the widget tree follows. ``session.add_view()``
from Python and "New mesh view" from the outline produce the same
window.

**The tiling law** ``T(N)`` (A1.2) is a pure function of the pane
count, never of click history::

    C = ceil(sqrt(N))                        columns
    cells filled row-major, in session.panes order
    column j holds panes j, j+C, j+2C, ...   (top to bottom)

Row-major minimises movement: adding a pane never moves the others
unless the column count changes (N = 2, 5, 10, ...). Closing re-tiles
the same way.

**The ratio law** (A1.2) is what keeps a re-tile from destroying a
hand-built arrangement::

    a splitter whose CHILD LIST changed  -> its ratios reset to equal
    every other splitter                 -> keeps its dragged ratios
    View -> Reset layout                 -> everything resets

An earlier draft reset every ratio on every Add; adversarial review
named it the likeliest first-week bug report, since the
one-large-mesh / one-narrow-plot arrangement this design exists to
support is destroyed by an Add that never touches the splitter holding
it.

**Panes are moved, never rebuilt.** A host that tore down and rebuilt
its panes would satisfy the cell arithmetic while silently destroying
every camera — backend state, not session state, so nothing downstream
would catch it. Criterion 4 asserts object identity across a re-tile
for exactly that reason, and the probe's reparent test is the GL half
of the same claim.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

from qtpy import QtCore, QtWidgets

from .._failures import safe_slot
from ..ui._layout_metrics import LAYOUT
from ._frame import SessionPaneFrame
from ._pane import BackendFactory

#: ``QSettings('apeGmsh', 'ResultsSession')`` keys (A1.5). Ratios only —
#: the SHAPE is derived from the pane count, so storing it would create
#: a second source of truth that could disagree with the session.
LAYOUT_KEY = "panes/layout"
SCHEMA_KEY = "panes/schema_version"
LAYOUT_SCHEMA_VERSION = 1


def tile_columns(pane_ids: "list[str]") -> "list[list[str]]":
    """``T(N)`` — the columns of the tiling, top-to-bottom per column.

    >>> tile_columns(["a", "b", "c"])
    [['a', 'c'], ['b']]
    """
    n = len(pane_ids)
    if n == 0:
        return []
    columns = math.ceil(math.sqrt(n))
    return [
        [pane_ids[i] for i in range(j, n, columns)]
        for j in range(columns)
    ]


def tile_shape(n: int) -> "tuple[int, int]":
    """``(columns, rows)`` of ``T(n)`` — the Add gate's arithmetic."""
    if n <= 0:
        return 0, 0
    columns = math.ceil(math.sqrt(n))
    return columns, math.ceil(n / columns)


def required_extent(n: int) -> "tuple[int, int]":
    """``(width, height)`` px a host needs to tile ``n`` panes at the
    A1.4 floors, handles included."""
    columns, rows = tile_shape(n)
    handle = LAYOUT.splitter_handle_width
    return (
        columns * LAYOUT.pane_min_width + max(0, columns - 1) * handle,
        rows * LAYOUT.pane_min_height + max(0, rows - 1) * handle,
    )


def _ratios(sizes: "list[int]") -> "list[float]":
    total = float(sum(sizes))
    if total <= 0.0 or not sizes:
        return [1.0 / len(sizes)] * len(sizes) if sizes else []
    return [s / total for s in sizes]


def _apply_ratios(splitter: Any, ratios: "list[float]") -> None:
    """Set ``splitter`` to ``ratios``.

    Passed as relative integers: ``QSplitter.setSizes`` normalises a
    size list whose sum differs from the splitter's extent, so ratios
    apply correctly even before the first layout pass — which is when
    a boot-time restore runs.
    """
    if splitter.count() != len(ratios) or not ratios:
        return
    splitter.setSizes([max(1, int(round(r * 10000))) for r in ratios])


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


class SessionPaneHost(QtWidgets.QWidget):
    """The tiled centre — a projection of ``session.panes``."""

    def __init__(
        self,
        session: Any,
        *,
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
        self._listed: "list[str]" = []
        self._columns: "list[QtWidgets.QSplitter]" = []
        self._column_ids: "list[tuple[str, ...]]" = []
        self._active_pane_id: Optional[str] = None
        self._disposed = False

        self.setObjectName("SessionPaneHost")
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        self._root = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal, self,
        )
        self._root.setObjectName("SessionPaneHostSplitter")
        self._root.setChildrenCollapsible(False)
        self._root.setHandleWidth(LAYOUT.splitter_handle_width)
        column.addWidget(self._root, stretch=1)

        self._empty = SessionPaneHostEmpty(self._new_view, parent=self)
        column.addWidget(self._empty, stretch=1)

        session.subscribe(self._on_session_tick)
        self.refresh()

    # -- surface -------------------------------------------------------

    @property
    def root_splitter(self) -> QtWidgets.QSplitter:
        """The root ``QSplitter(Horizontal)`` of columns."""
        return self._root

    @property
    def column_splitters(self) -> "tuple[QtWidgets.QSplitter, ...]":
        """The ``QSplitter(Vertical)`` columns, left to right."""
        return tuple(self._columns)

    @property
    def pane_frames(self) -> "tuple[SessionPaneFrame, ...]":
        """Every pane frame, in ``session.panes`` order (criterion 2)."""
        return tuple(
            self._frames[p.id] for p in self._session.panes
            if p.id in self._frames
        )

    @property
    def empty_card(self) -> SessionPaneHostEmpty:
        return self._empty

    def frame(self, pane_id: str) -> SessionPaneFrame:
        """The frame of one pane — how tests address a pane (caution 8).

        A miss is a ``KeyError``: like ``session.pane``, a stale id
        must fail loudly rather than answer with another pane.
        """
        return self._frames[pane_id]

    @property
    def active_pane_id(self) -> Optional[str]:
        """The one active pane (A1.5) — ``None`` only with zero panes."""
        return self._active_pane_id

    def set_active(self, pane_id: Optional[str]) -> None:
        """Make ``pane_id`` active; ``None`` clears (zero panes)."""
        if pane_id is not None and pane_id not in self._frames:
            return
        if pane_id == self._active_pane_id:
            return
        self._active_pane_id = pane_id
        for pid, frame in self._frames.items():
            frame.set_active(pid == pane_id)
        if pane_id is not None and self._on_active_changed is not None:
            self._on_active_changed(pane_id)

    # -- the Add gate (A1.4) -------------------------------------------

    def can_add(self) -> "tuple[bool, str]":
        """Whether ``T(N+1)`` clears both floors at the current size.

        The gate constrains CREATION only: a session that arrives with
        more panes than fit — ``s.add_view()`` in a script, an S5
        snapshot restore — is tiled anyway. The IR is truth and the
        host never refuses a pane that exists.
        """
        n = len(self._session.panes) + 1
        need_w, need_h = required_extent(n)
        have_w, have_h = self.width(), self.height()
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

    # -- persistence (A1.5) --------------------------------------------

    def layout_ratios(self) -> "dict[str, Any]":
        """The dragged ratios, as the stored ``panes/layout`` record."""
        return {
            "n": len(self.pane_frames),
            "cols": _ratios(self._root.sizes()),
            "rows": [_ratios(col.sizes()) for col in self._columns],
        }

    def apply_ratios(self, record: Any) -> bool:
        """Restore stored ratios; ``False`` when they do not apply.

        Applied only when ``n`` equals this session's pane count and
        the shape matches — otherwise equal ratios, SILENTLY. This is
        chrome (``_restore_layout``'s existing silent-mismatch
        behaviour), not §1's session-schema notice.
        """
        if not isinstance(record, dict):
            return False
        try:
            if int(record.get("n", -1)) != len(self.pane_frames):
                return False
            cols = [float(r) for r in record["cols"]]
            rows = [[float(r) for r in row] for row in record["rows"]]
        except (KeyError, TypeError, ValueError):
            return False
        if len(cols) != len(self._columns) or len(rows) != len(self._columns):
            return False
        if any(
            len(row) != col.count()
            for row, col in zip(rows, self._columns)
        ):
            return False
        _apply_ratios(self._root, cols)
        for col, row in zip(self._columns, rows):
            _apply_ratios(col, row)
        return True

    def reset_tiling(self) -> None:
        """Re-tile to ``T(N)`` with EQUAL ratios everywhere (View →
        Reset layout, A1.5)."""
        if self._columns:
            _apply_ratios(
                self._root, [1.0 / len(self._columns)] * len(self._columns),
            )
        for col in self._columns:
            if col.count():
                _apply_ratios(col, [1.0 / col.count()] * col.count())

    # -- lifecycle -----------------------------------------------------

    def request_reconcile(self) -> None:
        """Forced repaint of every pane (theme / density change)."""
        for frame in self._frames.values():
            frame.request_reconcile()

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        try:
            self._session.unsubscribe(self._on_session_tick)
        except Exception:
            pass
        for frame in list(self._frames.values()):
            frame.dispose()
        self._frames.clear()

    # -- projection ----------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the frame set from ``session.panes`` and re-tile.

        Skips the whole pass when membership is unchanged — the session
        ticks on EVERY mutation (slot fills included), and re-tiling on
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
        self._listed = list(wanted)

        gone = [pid for pid in self._frames if pid not in wanted]
        for pid in gone:
            frame = self._frames.pop(pid)
            frame.dispose()
            frame.setParent(None)
            frame.deleteLater()
        for pane in self._session.panes:
            if pane.id not in self._frames:
                self._frames[pane.id] = SessionPaneFrame(
                    self._session, pane,
                    on_activate=self.set_active,
                    on_close=self._close_pane,
                    backend_factory=self._backend_factory,
                    defer_fn=self._defer_fn,
                    parent=self,
                )
        self._tile(wanted)

        if self._active_pane_id not in self._frames:
            # A1.5: one active pane at all times when N >= 1. The pane
            # that closed hands focus to the first surviving one.
            self._active_pane_id = None
            self.set_active(wanted[0] if wanted else None)
        if self._on_panes_changed is not None:
            self._on_panes_changed()

    # -- Qt ------------------------------------------------------------

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 — Qt API
        super().resizeEvent(event)
        # The Add gate is a function of the host's size, so a resize
        # can enable or disable it (A1.4).
        if self._on_geometry_changed is not None:
            self._on_geometry_changed()

    # -- internals -----------------------------------------------------

    def _tile(self, order: "list[str]") -> None:
        columns = tile_columns(order)
        old_column_ids = list(self._column_ids)
        old_root = _ratios(self._root.sizes())
        old_rows = [_ratios(col.sizes()) for col in self._columns]

        while len(self._columns) < len(columns):
            col = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            col.setObjectName("SessionPaneColumn")
            col.setChildrenCollapsible(False)
            col.setHandleWidth(LAYOUT.splitter_handle_width)
            self._root.addWidget(col)
            self._columns.append(col)

        # MOVE the frames (never rebuild them — criterion 4). Every
        # frame belongs to exactly one target cell, so placing each one
        # explicitly leaves no strays behind: a frame pushed aside while
        # its column is processed is claimed by its own column's pass.
        for j, ids in enumerate(columns):
            col = self._columns[j]
            for i, pane_id in enumerate(ids):
                frame = self._frames[pane_id]
                if col.widget(i) is not frame:
                    col.insertWidget(i, frame)
                frame.show()

        for col in self._columns[len(columns):]:
            col.setParent(None)
            col.deleteLater()
        del self._columns[len(columns):]

        new_column_ids = [tuple(ids) for ids in columns]
        # The ratio law. The root's child list is its COLUMNS, and the
        # column widgets are reused, so the root changes exactly when
        # the column count does (N = 2, 5, 10, ...).
        if len(new_column_ids) == len(old_column_ids) and old_root:
            _apply_ratios(self._root, old_root)
        elif self._columns:
            _apply_ratios(
                self._root, [1.0 / len(self._columns)] * len(self._columns),
            )
        for j, ids in enumerate(new_column_ids):
            col = self._columns[j]
            unchanged = (
                j < len(old_column_ids)
                and old_column_ids[j] == ids
                and j < len(old_rows)
                and len(old_rows[j]) == col.count()
            )
            if unchanged:
                _apply_ratios(col, old_rows[j])
            elif col.count():
                _apply_ratios(col, [1.0 / col.count()] * col.count())

        self._column_ids = new_column_ids
        has_panes = bool(order)
        self._root.setVisible(has_panes)
        self._empty.setVisible(not has_panes)

    def _close_pane(self, pane_id: str) -> None:
        """The header's close glyph (A1.5).

        The widget disappears because the SESSION lost the pane, never
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


__all__ = [
    "LAYOUT_KEY",
    "LAYOUT_SCHEMA_VERSION",
    "SCHEMA_KEY",
    "SessionPaneHost",
    "SessionPaneHostEmpty",
    "required_extent",
    "tile_columns",
    "tile_shape",
]
