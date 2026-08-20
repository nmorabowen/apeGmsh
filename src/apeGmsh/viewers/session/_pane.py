"""MeshPane — the Qt widget projecting one mesh view of a session.

ADR 0098 S2, generalized to N panes by Amendment 1. Backend/interactor
injection is a HARD design requirement (plan decision 7): the pane
takes a ``backend_factory`` so headless tests inject a
``RecordingBackend`` and the pane-host oracle — widget event → session
diff → repaint — runs in the DEFAULT offscreen CI lane, not only under
xvfb. The default factory builds a real ``QtInteractor`` +
``PyVistaQtBackend``, one per pane (A1.1: the shell builds none).

The pane owns its :class:`~._reconciler.SessionReconciler` and its own
first-fit camera. Both are per-pane on purpose: S2 held one
``_camera_fit`` bool on the WINDOW (Amendment 1 caution 11), which
would leave pane 2 and beyond booting unframed.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from qtpy import QtWidgets

from ._legend_bind import PaneLegendBinding
from ._pick import PanePick
from ._reconciler import SessionReconciler

#: ``factory(parent_widget) -> (surface_widget | None, backend)``.
#: ``surface_widget`` (when not None) is mounted as the pane's child —
#: the render surface. A factory adopting an external surface (a test's
#: RecordingBackend) returns ``None`` there.
BackendFactory = Callable[[QtWidgets.QWidget], "tuple[Any, Any]"]


def _pane_theme() -> Any:
    """An explicit pyvista theme for a pane-owned interactor.

    Amendment 1 caution 2: ``viewer_window.py`` mutates pyvista's
    GLOBAL theme (``set_plot_theme("dark")``, white font) at window
    construction, so anything pyvista-defaulted inside a pane built
    later inherits ``dark`` regardless of the live palette. The
    amendment records legends as safe on the grounds that 0090 D1
    routes their colours through specs — **it is not so**:
    ``ScalarBarSpec`` carries geometry and font SIZES, no colour, so a
    bar's title and labels take ``plotter.theme.font.color``. On
    ``paper`` that meant white text on a light pane. Constructing each
    pane with a theme carrying the palette's font colour is what
    caution 2's "or construct panes with an explicit theme" buys, and
    it closes the defect without reordering the old window.

    ``None`` when pyvista's theme machinery is unavailable — the
    interactor then falls back to the global theme, exactly as S2 did.
    """
    try:
        import copy

        import pyvista as pv

        from ..ui.theme import THEME

        # ``deepcopy``, not ``.copy()``: pyvista's theme object has no
        # ``copy`` method on the shipped version, and an AttributeError
        # swallowed here would silently leave the pane on the global
        # (white-font) theme — which is exactly the bug.
        theme = copy.deepcopy(pv.global_theme)
        theme.font.color = THEME.current.text
        return theme
    except Exception:
        return None


def default_backend_factory(
    parent: QtWidgets.QWidget,
) -> "tuple[Any, Any]":
    """A pane-owned ``QtInteractor`` wrapped in a ``PyVistaQtBackend``."""
    from pyvistaqt import QtInteractor

    from ..backends import PyVistaQtBackend

    theme = _pane_theme()
    interactor = (
        QtInteractor(parent=parent) if theme is None
        else QtInteractor(parent=parent, theme=theme)
    )
    try:
        # Order-independent transparency, per pane — the shell used to
        # set this on its one central interactor (viewer_window.py).
        renderer = interactor.renderer
        renderer.SetUseDepthPeeling(True)
        renderer.SetMaximumNumberOfPeels(4)
        renderer.SetOcclusionRatio(0.0)
    except Exception:
        pass
    try:
        interactor.enable_parallel_projection()
    except Exception:
        pass
    apply_pane_navigation(interactor)
    # Theme the pane's background NOW, not only on the next palette
    # change: a pane is constructed after the shell has already applied
    # its palette, so waiting for ``on_theme_changed`` leaves a fresh
    # pane on VTK's own dark default — visibly wrong under `paper`.
    try:
        from ..scene.background import apply_background
        from ..ui.theme import THEME

        apply_background(interactor, THEME.current)
    except Exception:
        pass
    return interactor, PyVistaQtBackend(interactor)


def apply_pane_navigation(
    plotter: Any, token: "Optional[str]" = None,
) -> None:
    """Apply the mouse-navigation convention to ONE pane's interactor.

    ``ViewerWindow.set_navigation_style`` applies the preference to
    ``self._qt_interactor`` — which is ``None`` on the session path
    (A1.1: the shell builds no central interactor), so without this
    every pane would keep VTK's stock trackball, where LEFT orbits.
    That matters more here than anywhere else: §8's pick installs at
    interactor priority 10 and ABORTS the plain LMB chain, so a stock
    pane would answer a left drag with a rubber-band and have no orbit
    gesture left at all. ``apecad`` — the shipped default — is the
    mapping that makes the two coexist: LEFT is selection, MIDDLE
    orbits, RIGHT pans (``_navigation``'s own docstring names this
    slice as the reason).

    ``token=None`` reads the live preference. Failures are swallowed:
    a pane that cannot install a style still renders, and the stock
    trackball is a working camera, just not the configured one.
    """
    from ..ui._navigation import apply_navigation_style

    try:
        if token is None:
            from ..ui.preferences_manager import PREFERENCES
            token = PREFERENCES.current.mouse_navigation
        apply_navigation_style(plotter, token)
    except Exception:
        pass


def _supports_picking(backend: Any) -> bool:
    """Whether ``backend`` exposes a ``PickBackend`` (ADR 0047 INV-1).

    Probed, never assumed: ``supports_picking`` is an optional
    capability. Only the PROBE is guarded — a backend that answers
    ``True`` and then cannot build a picker is a broken backend, not a
    pane that should quietly render without picking.
    """
    probe = getattr(backend, "supports_picking", None)
    return bool(probe is not None and probe())


class MeshPane(QtWidgets.QWidget):
    """One mesh view, projected. The session is truth; this widget is
    a client (§1) — it holds no picture state of its own, only the
    backend, the reconciler, the §8 pick installation, the render
    surface (when it owns one) and its own first-fit camera flag.
    """

    def __init__(
        self,
        session: Any,
        pane_id: Optional[str] = None,
        *,
        backend_factory: Optional[BackendFactory] = None,
        defer_fn: Optional[Callable[[Callable[[], None]], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        factory = backend_factory or default_backend_factory
        surface, backend = factory(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if surface is not None:
            layout.addWidget(surface)
        self._surface = surface
        self._backend = backend
        self._camera_fit = False
        # The reconciler schedules its own boot flush — the valid
        # empty picture (§3) paints without any session mutation.
        self._reconciler = SessionReconciler(
            session, backend, pane_id=pane_id, defer_fn=defer_fn,
        )
        self._reconciler.add_listener(self._on_reconciled)
        # §8 — this pane's clicks and windows, aimed by ITS radio at
        # the ONE session set. Only a backend that can ray-cast gets
        # one; a view-only backend (a plain recorder) is legal and
        # simply cannot pick (ADR 0042 INV-3 / 0047 INV-1).
        self._pick: Optional[PanePick] = None
        if _supports_picking(backend):
            self._pick = PanePick(
                session, backend.picking(),
                lambda: self._reconciler.realized,
            )
        # A5.3(3) — drag and resize on the colour scale. Bound per
        # flush, not once: realize builds a new LegendController every
        # time, so an interactor installed here would observe a dead
        # one after the first slot edit.
        self._legends = PaneLegendBinding(backend, self._reconciler)
        # Adopt a boot flush that has ALREADY happened. The reconciler
        # schedules its own, and an immediate ``defer_fn`` runs it
        # inside the constructor above — before the listener existed,
        # so nothing would bind its controller and the pane would wait
        # for a flush that only a session edit brings. Under a QTimer
        # defer this is a no-op (realized is still None).
        self._legends.bind(
            getattr(self._reconciler.realized, "legend_controller", None)
        )

    # -- surface -------------------------------------------------------

    @property
    def backend(self) -> Any:
        """The injected/constructed backend (unwrapped)."""
        return self._backend

    @property
    def surface(self) -> Optional[QtWidgets.QWidget]:
        """The pane-owned render surface widget, if the factory built
        one (``None`` when an external surface was adopted)."""
        return self._surface

    @property
    def reconciler(self) -> SessionReconciler:
        return self._reconciler

    @property
    def pick(self) -> Optional[PanePick]:
        """This pane's §8 pick controller (``None`` when the backend
        cannot ray-cast)."""
        return self._pick

    def set_pane(self, pane_id: Optional[str]) -> None:
        """Aim the projection at another mesh view.

        Vestigial under the pane host (Amendment 1 caution 3): a hosted
        pane is constructed with its own id and never re-aimed. Kept
        for the unhosted single-pane shape and for tests; no behaviour
        rides it.
        """
        self._reconciler.set_pane(pane_id)

    def apply_navigation(self, token: str) -> None:
        """Re-apply a navigation convention to this pane's surface.

        The View → Navigation switch is a window-level gesture and the
        panes own the interactors, so it has to reach every one of
        them. A pane with no surface of its own (an injected backend)
        has no interactor to style.
        """
        if self._surface is not None:
            apply_pane_navigation(self._surface, token)

    def request_reconcile(self) -> None:
        """Schedule a repaint for a backend-side change the session
        cannot see (theme palette swap, density tokens).

        FORCED past the reconciler's signature guard — the palette is
        not session state, so an equality check over the session would
        skip exactly the repaint that is needed.
        """
        self._reconciler.schedule_forced()

    def dispose(self) -> None:
        """Detach from the session AND close the pane's GL context.

        Idempotent. Closing the interactor is not optional bookkeeping:
        before Amendment 1 the shell owned the one context and
        ``ViewerWindow._MainWindow.closeEvent`` closed it. With the
        shell building none (A1.1), a pane that does not close its own
        leaves a live render window behind every time a pane is closed
        or a window shuts — caution 1's orphaned context, moved from
        the shell to the panes. Windows tolerates the leak; Mesa
        segfaults on the NEXT interactor in the process, which is what
        the xvfb lane caught.

        The pick installation dies on this same path, for the same
        reason: its observers live on the interactor this method is
        about to close.
        """
        self._reconciler.dispose()
        self._legends.dispose()
        if self._pick is not None:
            self._pick.dispose()
        if self._surface is not None:
            try:
                self._surface.close()
            except Exception:
                pass

    def closeEvent(self, event: Any) -> None:  # noqa: N802 — Qt API
        self.dispose()
        super().closeEvent(event)

    # -- internals -----------------------------------------------------

    def _on_reconciled(self, realized: Optional[Any]) -> None:
        """First non-empty paint: frame the model once, in THIS pane.

        Later flushes never reframe — an update is not a reason to move
        the camera (the backend's own update path holds the same rule).
        """
        # Re-bind FIRST and unconditionally: the camera guard below
        # returns early on most flushes, and a legend whose controller
        # was swapped by that flush must not keep the old binding.
        self._legends.bind(
            getattr(realized, "legend_controller", None)
            if realized is not None else None
        )
        if self._camera_fit or realized is None or not realized.layers:
            return
        try:
            self._backend.reset_camera()
        except Exception:
            pass
        self._camera_fit = True


__all__ = [
    "BackendFactory", "MeshPane", "apply_pane_navigation",
    "default_backend_factory",
]
