"""SessionWindow — the Qt client of a ``ResultsSession``.

ADR 0098 S2 + Amendment 1 (S3). COMPOSES the existing
:class:`ResultsWindow` shell (verified director-free, generic mount
points) — it does not build a new window. What it adds is the
projection: the outline (panes + model composition, left), the
mesh-view inspector page (right), the :class:`SessionPaneHost` as the
**central widget**, and an inert scrubber placeholder (the time link
is S4-3).

**The shell builds no central interactor on this path** (Amendment 1
A1.1). Every GL context in the window belongs to a pane, which is what
lets seven of the pane-host acceptance criteria run in the default
offscreen lane with injected backends instead of paying the xvfb lane
for the whole suite — and it disposes of the orphaned-context risk
(caution 1) by construction rather than managing it. The seam is
additive and opt-in: ``results_viewer.py``'s construction path is
untouched, which is what ``central_interactor=True`` (the default)
means.

The toolbar's camera presets / fit / screenshot act on the **active
pane**, through the shell's plotter provider — one active object, one
place that resolves it.

QSettings scope (plan decision 8): the session window must NOT share
``QSettings('apeGmsh', 'ResultsViewer')`` schema v7 with the live old
window — a saved arrangement from either would half-apply to the
other. :class:`SessionResultsWindow` overrides ONLY the persistence
identity: scope ``'ResultsSession'``, schema numbering restarted at
1. It is a subclass (not an instance patch) because
``ResultsWindow.__init__`` restores the stored layout before any
caller could re-point it. The dock objectNames are reused verbatim
inside the new scope — criterion 11 constrains the four RETIRED
names, and separate scopes share no state. The pane host adds no dock,
so ``_LAYOUT_SCHEMA_VERSION`` stays 1 (A1.5); the splitter ratios ride
their own ``panes/*`` keys in the same scope. At S6 the flip decides
whether this window adopts the old scope.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from .._failures import register_error_handler, unregister_error_handler
from ..ui._results_window import ResultsWindow
from ._host import (
    LAYOUT_KEY,
    LAYOUT_SCHEMA_VERSION,
    SCHEMA_KEY,
    SessionPaneHost,
)
from ._inspector import (
    MeshInspectorPage,
    PanePlaceholderPage,
    PlotInspectorPage,
)
from ._outline import SessionOutline

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
    # window's v7 history. Amendment 1 adds no dock, so it stays 1.
    _LAYOUT_SCHEMA_VERSION = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        #: Called after ``reset_layout`` restores the 0088 D1 dock set,
        #: so View → Reset layout also re-tiles the pane host (A1.5).
        self.reset_layout_hook: Optional[Callable[[], None]] = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _layout_settings():
        from qtpy.QtCore import QSettings

        return QSettings("apeGmsh", "ResultsSession")

    def reset_layout(self) -> None:
        super().reset_layout()
        if self.reset_layout_hook is not None:
            self.reset_layout_hook()


class SessionWindow:
    """One window projecting one session (§1: the window is a client)."""

    def __init__(
        self,
        session: Any,
        *,
        title: Optional[str] = None,
        backend_factory: Optional[Any] = None,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
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
            central_interactor=False,
        )
        self._closed = False

        # The host and the outline are two renderings of one state
        # (A1.5), so each calls back into the other. The host is built
        # first — it owns the pane frames the outline's Add gate sizes
        # against — and its constructor already fires the boot
        # activation, so everything a callback touches is initialised
        # here, above it.
        self._pages: dict[str, Any] = {}
        self._current_page: Any = None
        self._nothing_selected = PanePlaceholderPage(
            "No pane selected. Add a mesh view from the outline to "
            "start a picture.",
        )
        self._outline: Optional[SessionOutline] = None
        self._host: Optional[SessionPaneHost] = None

        self._host = SessionPaneHost(
            session,
            backend_factory=backend_factory,
            defer_fn=defer_fn,
            on_active_changed=self._on_active_changed,
            on_panes_changed=self._on_panes_changed,
            on_geometry_changed=self._on_geometry_changed,
        )
        self._shell.set_central_widget(self._host)
        # The toolbar acts on ONE object: whichever pane is active.
        self._shell.set_plotter_provider(self._active_plotter)

        self._outline = SessionOutline(
            session,
            on_pane_selected=self._on_pane_selected,
            on_new_view=self._on_new_view,
            on_new_plot=self._on_new_plot,
            on_new_plot_from_selection=self._on_new_plot_from_selection,
            on_close_pane=self._on_close_pane,
            can_add=self._host.can_add,
        )
        self._shell.set_left_widget(self._outline.widget)

        from qtpy import QtWidgets

        placeholder = QtWidgets.QLabel(
            "Time scrubber lands with the session time link (S4-3). "
            "A plot's playhead already follows session.time.",
        )
        placeholder.setEnabled(False)
        self._shell.set_bottom_widget(placeholder)

        # Boot selection: the host already made the first pane active;
        # mirror it onto the outline and the inspector.
        if self._host.active_pane_id is not None:
            self._on_active_changed(self._host.active_pane_id)
        else:
            self._mount_page(None)

        # A palette swap changes the substrate colours realize bakes
        # into its layers AND the panes' own backgrounds — one palette,
        # N panes.
        self._shell.on_theme_changed(self._on_theme_changed)
        self._shell.reset_layout_hook = self._on_reset_layout
        # The panes own the interactors, so the navigation menu has to
        # reach them — one palette, one convention, N panes.
        self._shell.navigation_hook = self._host.apply_navigation

        self._restore_pane_layout()

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
    def host(self) -> SessionPaneHost:
        """The tiled centre (Amendment 1). Tests address panes through
        it — ``host.frame(pane_id)`` / ``host.pane_frames`` — never
        through per-pane objectNames (caution 8)."""
        return self._host

    @property
    def outline(self) -> SessionOutline:
        return self._outline

    def inspector_page(self, pane_id: str) -> Any:
        """The (cached) inspector page for a pane id."""
        return self._pages[pane_id]

    @property
    def current_page(self) -> Any:
        """The mounted inspector page (criterion 9)."""
        return self._current_page

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

    # -- active pane (A1.5: one state, three renderings) ----------------

    def _active_plotter(self) -> Any:
        pane_id = None if self._host is None else self._host.active_pane_id
        if pane_id is None:
            return None
        try:
            return self._host.frame(pane_id).plotter
        except KeyError:
            return None

    def _on_active_changed(self, pane_id: str) -> None:
        """The host made a pane active — mirror onto outline + inspector."""
        if self._outline is not None:
            self._outline.set_active_pane(pane_id)
        self._mount_page(pane_id)

    def _on_pane_selected(self, pane_id: str) -> None:
        """An outline row was selected — same state, other direction."""
        self._host.set_active(pane_id)
        if self._outline is not None:
            self._outline.refresh_scope()
        self._mount_page(pane_id)

    def _mount_page(self, pane_id: Optional[str]) -> None:
        if pane_id is None:
            # Zero panes is a valid session (A1.4), so the inspector
            # says so beside the host's empty card rather than
            # vanishing — 0088 D2's nothing-selected hint.
            self._current_page = self._nothing_selected
            self._shell.set_inspector_widget(self._nothing_selected.widget)
            return
        from apeGmsh.results.session import MeshView

        try:
            pane = self._session.pane(pane_id)
        except KeyError:
            return
        page = self._pages.get(pane_id)
        if page is None:
            if isinstance(pane, MeshView):
                page = MeshInspectorPage(self._session, pane)
            else:
                page = PlotInspectorPage(self._session, pane)
            self._pages[pane_id] = page
        self._current_page = page
        self._shell.set_inspector_widget(page.widget)
        page.set_realized(self._realized_for(pane_id))

    def _realized_for(self, pane_id: str) -> Any:
        """What that pane last realized — a ``RealizedPane`` for a mesh
        view, a ``RealizedPlot`` for a plot. Both content panes carry a
        reconciler, so this needs no kind test."""
        if self._host is None:
            return None
        try:
            frame = self._host.frame(pane_id)
        except KeyError:
            return None
        return frame.pane.reconciler.realized

    # -- pane lifecycle -------------------------------------------------

    def _on_panes_changed(self) -> None:
        """A pane was added or removed — evict dead inspector pages.

        Amendment 1 caution 9: ``_pages`` only ever grew, and a page
        subscribes to the session when constructed. With one pane that
        was harmless; the moment the header's close glyph exists, every
        closed pane would leak a live subscriber refreshing a dead
        view. The page dies with the pane.
        """
        if self._host is None:
            return  # the host's own boot refresh; __init__ mounts below
        live = {pane.id for pane in self._session.panes}
        orphaned = False
        for pane_id in [p for p in self._pages if p not in live]:
            page = self._pages.pop(pane_id)
            orphaned = orphaned or page is self._current_page
            page.dispose()
        if orphaned or not live:
            self._mount_page(self._host.active_pane_id)
        if self._outline is not None:
            self._outline.refresh()

    def _on_new_view(self) -> None:
        view = self._session.add_view()
        self._host.set_active(view.id)

    def _on_new_plot(self) -> None:
        plot = self._session.add_plot()
        self._host.set_active(plot.id)

    def _on_new_plot_from_selection(self, quantity: str) -> None:
        """§8's own test, as a gesture: the set IS the source.

        The membership is COPIED into concrete node / Gauss sources by
        ``add_plot_from_selection`` (§6 — a live alias is not v1), so
        the chart that opens keeps showing these curves however the
        selection moves next."""
        plot = self._session.add_plot_from_selection(quantity)
        self._host.set_active(plot.id)

    def _on_close_pane(self, pane_id: str) -> None:
        try:
            self._session.remove_pane(pane_id)
        except KeyError:
            pass

    def _on_geometry_changed(self) -> None:
        if self._outline is not None:
            self._outline.refresh_gate()

    # -- theme ---------------------------------------------------------

    def _on_theme_changed(self, palette: Any) -> None:
        """One palette, N panes: repaint every pane's background and
        force every reconciler past its signature guard (the palette is
        not session state, so an equality check would skip it)."""
        from ..scene.background import apply_background

        for frame in self._host.pane_frames:
            plotter = frame.plotter
            if plotter is None:
                continue
            try:
                apply_background(plotter, palette)
            except Exception:
                try:
                    plotter.set_background(
                        palette.bg_top, top=palette.bg_bottom,
                    )
                except Exception:
                    pass
        self._host.request_reconcile()

    # -- layout persistence (A1.5) --------------------------------------

    def _on_reset_layout(self) -> None:
        """View → Reset layout: the 0088 D1 dock set (the shell's job,
        already done) **and** ``T(N)`` with equal ratios."""
        self._host.reset_tiling()

    def _restore_pane_layout(self) -> None:
        """Apply the stored ratios, or equal ones — silently.

        Chrome, not session state: a mismatched ``n``, a bad version or
        corrupt JSON falls back to equal ratios with no notice and no
        raise (``_restore_layout``'s existing behaviour). Before S5 the
        stored ratios are nearly inert — a fresh session boots at N = 1,
        so the ``n`` guard rejects most restores; their real consumers
        are the S5 snapshot restore and same-shape scripted sessions.
        """
        settings = self._shell._layout_settings()  # noqa: SLF001
        try:
            if int(settings.value(SCHEMA_KEY, 0)) != LAYOUT_SCHEMA_VERSION:
                return
            raw = settings.value(LAYOUT_KEY)
            if not raw:
                return
            self._host.apply_ratios(json.loads(str(raw)))
        except (TypeError, ValueError):
            return

    def _save_pane_layout(self) -> None:
        settings = self._shell._layout_settings()  # noqa: SLF001
        try:
            settings.setValue(SCHEMA_KEY, LAYOUT_SCHEMA_VERSION)
            settings.setValue(
                LAYOUT_KEY, json.dumps(self._host.layout_ratios()),
            )
        except Exception:
            pass

    # -- internals -----------------------------------------------------

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
        self._save_pane_layout()
        unregister_error_handler(self._error_handler)
        self._host.dispose()
        if self._outline is not None:
            self._outline.dispose()
        for page in self._pages.values():
            page.dispose()
        self._pages.clear()
        if self in _LIVE_WINDOWS:
            _LIVE_WINDOWS.remove(self)


def show_session(
    session: Any, *, blocking: bool = True, title: Optional[str] = None,
) -> Optional[SessionWindow]:
    """Open the Qt client on ``session`` (``ResultsSession.show``).

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
