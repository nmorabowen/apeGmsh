"""``TrameBackend`` — the web / Jupyter render backend (ADR 0042, R-C).

A :class:`~apeGmsh.viewers.backends.pyvista_qt.PyVistaBackend` over a
plain ``pyvista.Plotter`` that is *served* through ``pyvista.trame``
rather than hosted in a Qt window. Because the whole ``scene_ir`` →
pyvista translation is windowing-agnostic, this backend inherits it
unchanged from ``PyVistaBackend``; only two things differ from the
desktop :class:`~apeGmsh.viewers.backends.pyvista_qt.PyVistaQtBackend`:

* it owns a non-Qt ``pyvista.Plotter`` (created here when not supplied),
  which the trame shell (:mod:`apeGmsh.viewers.web_viewer`) serves;
* picking is **off** (:meth:`supports_picking` → ``False``). Web picking
  is the most VTK-bound surface and is deferred to Phase R-D behind a
  separate ``PickBackend`` Protocol, exactly as ADR 0042 planned.

The hybrid local-WebGL / remote-software render mode (resolved Q2) is a
``pyvista.trame`` *view* configuration applied at serve time by the
shell, not a backend concern — so it does not appear here.
"""
from __future__ import annotations

from typing import Any, Optional

from .pyvista_qt import PyVistaBackend


class TrameBackend(PyVistaBackend):
    """``RenderBackend`` over a ``pyvista.Plotter`` served via trame."""

    def __init__(self, plotter: Optional[Any] = None) -> None:
        if plotter is None:
            import pyvista as pv

            plotter = pv.Plotter()
        super().__init__(plotter)

    def supports_picking(self) -> bool:
        # View-only on the web for now; picking returns in R-D.
        return False


__all__ = ["TrameBackend"]
