"""SessionWindow — the S2 Qt client of a ``ResultsSession``.

ADR 0098 S2: COMPOSES the existing :class:`ResultsWindow` shell
(verified director-free, generic mount points) — it does not build a
new window. What S2 adds is the projection: the outline's panes
group (left), the mesh-view inspector page (right), one
:class:`MeshPane` reconciling the outline-selected mesh view into
the shell's viewport, and an inert scrubber placeholder (the time
link is S4-3). N panes wait for the pane host (Amendment 1 / S3).

QSettings scope (plan decision 8): the session window must NOT share
``QSettings('apeGmsh', 'ResultsViewer')`` schema v7 with the live old
window — a saved arrangement from either would half-apply to the
other. :class:`SessionResultsWindow` overrides ONLY the persistence
identity: scope ``'ResultsSession'``, schema numbering restarted at
1. It is a subclass (not an instance patch) because
``ResultsWindow.__init__`` restores the stored layout before any
caller could re-point it. The dock objectNames are reused verbatim
inside the new scope — criterion 11 constrains the four RETIRED
names, and separate scopes share no state. At S6 the flip decides
whether this window adopts the old scope.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .._failures import register_error_handler, unregister_error_handler
from ..ui._results_window import ResultsWindow
from ._inspector import MeshInspectorPage, PanePlaceholderPage
from ._outline import SessionOutline
from ._pane import MeshPane

_SKIP_ENV_NOTICE = "[skip viewer] APEGMSH_SKIP_VIEWER set"

#: Non-blocking windows pinned until closed — same idiom as the old
#: viewer's live registry: ``show(blocking=False)`` returning into a
#: script that drops the handle must not garbage-collect the window.
_LIVE_WINDOWS: "list[SessionWindow]" = []


class SessionResultsWindow(ResultsWindow):
    """The ResultsWindow shell under the session's OWN persistence
    scope (plan decision 8). Layout machinery, docks, menus, focus
    mode: inherited unchanged."""

    # Fresh scope, fresh numbering: v1 is the S2 composition. Bumps
    # here follow this window's own dock changes, not the old
    # window's v7 history.
    _LAYOUT_SCHEMA_VERSION = 1

    @staticmethod
    def _layout_settings():
        from qtpy.QtCore import QSettings

        return QSettings("apeGmsh", "ResultsSession")


class SessionWindow:
    """One window projecting one session (§1: the window is a client).

    The central viewport is the shell's own ``QtInteractor``; the
    :class:`MeshPane` adopts it through the backend factory instead of
    constructing a second interactor, so the shell's camera / theme /
    screenshot machinery keeps operating on the one real plotter.
    """

    def __init__(
        self, session: Any, *, title: Optional[str] = None,
    ) -> None:
        results = session.results
        if results is None:
            raise RuntimeError(
                "This session has no Results bound — construct it via "
                "results.session() to show a window."
            )
        if results.fem is None:
            raise RuntimeError(
                "session.show requires a bound FEMData (construct "
                "with model= / model_h5= or call results.bind)."
            )
        self._session = session
        self._shell = SessionResultsWindow(
            title=title or "Results session",
            on_close=self._on_close,
        )
        self._closed = False
        self._camera_fit = False

        from ..backends import PyVistaQtBackend

        def _adopt_shell_plotter(_parent: Any) -> "tuple[None, Any]":
            return None, PyVistaQtBackend(self._shell.plotter)

        # Unparented and never shown: with the shell's plotter adopted
        # the pane owns no surface widget of its own — it is the
        # reconciler's home until S3's pane host mounts real ones.
        self._pane = MeshPane(
            session, backend_factory=_adopt_shell_plotter,
        )
        self._pane.reconciler.add_listener(self._on_reconciled)

        self._outline = SessionOutline(
            session, on_pane_selected=self._on_pane_selected,
        )
        self._shell.set_left_widget(self._outline.widget)

        self._pages: dict[str, Any] = {}
        self._current_page: Any = None

        from qtpy import QtWidgets

        placeholder = QtWidgets.QLabel(
            "Time scrubber lands with the session time link (S4).",
        )
        placeholder.setEnabled(False)
        self._shell.set_bottom_widget(placeholder)

        # Boot selection: the first pane, mirroring the reconciler's
        # own default projection.
        panes = session.panes
        if panes:
            self._outline.select_pane(panes[0].id)

        # A palette swap changes the substrate colours realize bakes
        # into its layers — repaint on theme change.
        self._shell.on_theme_changed(
            lambda _palette: self._pane.request_reconcile(),
        )

        # Surface swallowed reconciler failures on the status bar
        # (ADR 0084 D4 — never silent).
        self._error_handler = self._on_pump_error
        register_error_handler(self._error_handler)

    # -- surface -------------------------------------------------------

    @property
    def session(self) -> Any:
        return self._session

    @property
    def shell(self) -> ResultsWindow:
        return self._shell

    @property
    def pane(self) -> MeshPane:
        return self._pane

    @property
    def outline(self) -> SessionOutline:
        return self._outline

    def inspector_page(self, pane_id: str) -> Any:
        """The (cached) inspector page for a pane id."""
        return self._pages[pane_id]

    def show(self, *, blocking: bool = True) -> Optional[int]:
        """Present the window; run the Qt loop when ``blocking``."""
        if blocking:
            return self._shell.exec()
        _LIVE_WINDOWS.append(self)
        self._shell.present()
        return None

    def close(self) -> None:
        try:
            self._shell.window.close()
        except Exception:
            pass

    # -- internals -----------------------------------------------------

    def _on_pane_selected(self, pane_id: str) -> None:
        from apeGmsh.results.session import MeshView

        try:
            pane = self._session.pane(pane_id)
        except KeyError:
            return
        if isinstance(pane, MeshView):
            self._pane.set_pane(pane_id)
            page = self._pages.get(pane_id)
            if page is None:
                page = MeshInspectorPage(self._session, pane)
                self._pages[pane_id] = page
        else:
            page = self._pages.get(pane_id)
            if page is None:
                page = PanePlaceholderPage(
                    f"Plot pane {pane_id!r}: the Qt plot pane lands "
                    f"at S4-2. Its series realize today via "
                    f"results.plot / session.realize.",
                )
                self._pages[pane_id] = page
        self._current_page = page
        self._shell.set_inspector_widget(page.widget)

    def _on_reconciled(self, realized: Optional[Any]) -> None:
        if self._current_page is not None:
            self._current_page.set_realized(realized)
        # First non-empty paint: frame the model once. Later flushes
        # never reframe — an update is not a reason to move the
        # camera (the backend's own update path holds the same rule).
        if not self._camera_fit and realized is not None and realized.layers:
            try:
                self._pane.backend.reset_camera()
            except Exception:
                pass
            self._camera_fit = True

    def _on_pump_error(self, name: str, exc: BaseException) -> None:
        if not name.startswith("pump.session"):
            return
        try:
            self._shell.set_status(
                f"Repaint failed — {type(exc).__name__}: {exc}", 8000,
            )
        except Exception:
            pass

    def _on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        unregister_error_handler(self._error_handler)
        self._pane.dispose()
        self._outline.dispose()
        for page in self._pages.values():
            page.dispose()
        if self in _LIVE_WINDOWS:
            _LIVE_WINDOWS.remove(self)


def show_session(
    session: Any, *, blocking: bool = True, title: Optional[str] = None,
) -> Optional[SessionWindow]:
    """Open the S2 Qt client on ``session`` (``ResultsSession.show``).

    Honors ``APEGMSH_SKIP_VIEWER=1`` with the standard one-line notice
    (no window, returns ``None``) so headless runs and docs builds
    stay windowless. Returns the :class:`SessionWindow` (after the
    loop exits when ``blocking``; immediately otherwise).
    """
    if os.environ.get("APEGMSH_SKIP_VIEWER"):
        print(_SKIP_ENV_NOTICE)
        return None
    window = SessionWindow(session, title=title)
    window.show(blocking=blocking)
    return window


__all__ = ["SessionResultsWindow", "SessionWindow", "show_session"]
