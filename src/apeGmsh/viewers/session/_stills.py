"""``session.render`` — an offscreen still of one pane (ADR 0098 §1).

The session-side counterpart of ``render_results``: same offscreen
plotter, same theme, same camera defaults, same GL-skip discipline —
all shared verbatim through ``apeGmsh.viewers._stills`` (relocated
from ``render.py``). What differs is the picture source: the pane is
realized from the session IR (``realize_pane``), not from ``view=``
tokens; the closed picture space is the §4 slot catalog (amended ADR
0094 INV-10).

``APEGMSH_SKIP_VIEWER=1`` or no GL prints the ``[skip viewer]`` notice
and returns ``None`` — no file, no partial file.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .._stills import (
    _DEFAULT_WINDOW,
    _SKIP_ENV,
    _apply_camera,
    _apply_theme,
    _env_skips,
    _open_plotter,
    _screenshot,
)
# The camera-token rule stays in render.py (its tests patch it there);
# sharing the one definition keeps the two still clients' defaults
# identical.
from ..render import _resolve_camera
from ._realize import realize_pane

if TYPE_CHECKING:
    from apeGmsh.results.session import MeshView, ResultsSession


def resolve_pane(session: "ResultsSession", pane: Any) -> Any:
    """The pane ``render`` / ``realize`` address (§1).

    ``None`` resolves to the session's only pane — with one pane no id
    is needed; more than one (or none) refuses loudly. A string is a
    stable pane id; a pane object must be a member of this session.
    """
    if pane is None:
        panes = session.panes
        if len(panes) == 1:
            return panes[0]
        raise ValueError(
            f"This session has {len(panes)} panes — address one by id "
            f"(have: {[p.id for p in panes]})."
        )
    if isinstance(pane, str):
        return session.pane(pane)
    return pane


def render_still(
    session: "ResultsSession",
    pane: Any,
    path: "str | Path",
    *,
    camera: Optional[str] = None,
    window_size: tuple[int, int] = _DEFAULT_WINDOW,
) -> Optional[Path]:
    """Write one still of ``pane``. Returns the Path or ``None`` on skip.

    ``camera=`` defaults to ``xy`` for a planar model, ``iso``
    otherwise (ADR 0094 Amendment 3) — same rule as ``results.render``.
    """
    from apeGmsh.results.session import MeshView

    view = resolve_pane(session, pane)
    if not isinstance(view, MeshView):
        raise NotImplementedError(
            f"session.render writes stills of mesh panes; pane "
            f"{view.id!r} is a plot pane — a matplotlib still of the "
            f"same queries is results.plot (§10), and the plot pane's "
            f"own still lands with S4-2."
        )
    results = session.results
    if results is None:
        raise RuntimeError(
            "This session has no Results bound — construct it via "
            "results.session() to realize or render."
        )
    if results.fem is None:
        raise RuntimeError(
            "session.render requires a bound FEMData "
            "(construct with model= / model_h5= or call results.bind)."
        )
    camera = _resolve_camera(camera, results.fem.nodes.coords)
    if _env_skips():
        print(_SKIP_ENV)
        return None

    from ..backends import PyVistaQtBackend

    plotter = None
    try:
        plotter = _open_plotter(window_size)
        if plotter is None:
            return None
        _apply_theme(plotter)
        backend = PyVistaQtBackend(plotter)
        realize_pane(session, view, backend)
        _apply_camera(plotter, camera)
        return _screenshot(plotter, path)
    finally:
        if plotter is not None:
            try:
                plotter.close()
            except Exception:
                pass


__all__ = ["render_still", "resolve_pane"]
