"""MeshPane — the Qt widget projecting one mesh view of a session.

ADR 0098 S2. Backend/interactor injection is a HARD design
requirement (plan decision 7): the pane takes a ``backend_factory``
so headless tests inject a ``RecordingBackend`` and the S2 oracle —
widget event → session diff → repaint — runs in the DEFAULT
offscreen CI lane, not only under xvfb. The default factory builds a
real ``QtInteractor`` + ``PyVistaQtBackend``; the composed
``SessionWindow`` injects a factory that adopts the shell's existing
plotter instead (one interactor per window — the camera / theme /
screenshot machinery keeps operating on the real one).

The pane owns its :class:`~._reconciler.SessionReconciler` — the
per-pane reconcile is the shape S3's pane host multiplies.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from qtpy import QtWidgets

from ._reconciler import SessionReconciler

#: ``factory(parent_widget) -> (surface_widget | None, backend)``.
#: ``surface_widget`` (when not None) is mounted as the pane's child —
#: the render surface. A factory adopting an external surface (the
#: shell's plotter, a test's RecordingBackend) returns ``None`` there.
BackendFactory = Callable[[QtWidgets.QWidget], "tuple[Any, Any]"]


def default_backend_factory(
    parent: QtWidgets.QWidget,
) -> "tuple[Any, Any]":
    """A pane-owned ``QtInteractor`` wrapped in a ``PyVistaQtBackend``."""
    from pyvistaqt import QtInteractor

    from ..backends import PyVistaQtBackend

    interactor = QtInteractor(parent=parent)
    return interactor, PyVistaQtBackend(interactor)


class MeshPane(QtWidgets.QWidget):
    """One mesh view, projected. The session is truth; this widget is
    a client (§1) — it holds no picture state of its own, only the
    backend, the reconciler, and the render surface (when it owns
    one).
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
        # The reconciler schedules its own boot flush — the valid
        # empty picture (§3) paints without any session mutation.
        self._reconciler = SessionReconciler(
            session, backend, pane_id=pane_id, defer_fn=defer_fn,
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

    def set_pane(self, pane_id: Optional[str]) -> None:
        """Aim the projection at another mesh view (outline click)."""
        self._reconciler.set_pane(pane_id)

    def request_reconcile(self) -> None:
        """Schedule a repaint for a backend-side change the session
        cannot see (theme palette swap)."""
        self._reconciler.schedule()

    def dispose(self) -> None:
        """Detach from the session (idempotent)."""
        self._reconciler.dispose()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 — Qt API
        self.dispose()
        super().closeEvent(event)


__all__ = ["BackendFactory", "MeshPane", "default_backend_factory"]
