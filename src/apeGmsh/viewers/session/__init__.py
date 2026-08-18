"""``apeGmsh.viewers.session`` — realize clients of the ResultsSession.

The projection side of ADR 0098: the session IR
(``apeGmsh.results.session``) is truth; this package turns it into
scene-IR layers (``realize_pane``) and offscreen stills
(``render_still``), reusing the existing per-kind emit code in place,
director-free. It lives under ``apeGmsh.viewers`` — not under the
session package — so the session IR never imports
``viewers.diagrams`` (the ``test_session_pure.py`` guard);
``ResultsSession.realize`` / ``.render`` reach in here lazily.

S4-1 adds the §8 selection surface's projection: the node / Gauss
pick targets realize puts on screen (:class:`PaneTargets`) and
:class:`PanePick`, which turns one pane's clicks and rubber-bands into
writes on the session's ONE selection set. S4-2 adds the outline's
select-all (:func:`select_all`, the §8 fourth writer) and the plot
pane — :class:`PlotChart` and its own :class:`PlotReconciler`, because
the mesh reconciler serves mesh views only.

The Qt pane client lives here as a further projection:
:class:`SessionReconciler` (session diff → backend ops → one
coalesced render), :class:`MeshPane` (injectable backend),
:class:`SessionPaneHost` (the Amendment 1 tiled centre) with its
:class:`SessionPaneFrame`\\ s, the outline / inspector widgets, and
:func:`show_session` — the ``ResultsSession.show`` entry. The
Qt-touching names are exported LAZILY (PEP 562): ``realize`` /
``render`` must keep working in environments with no qtpy, exactly as
they did before S2.
"""
from __future__ import annotations

from importlib import import_module

from ._pick import PanePick
from ._plots import RealizedPlot, RealizedSeries, realize_plot
from ._select_all import gauss_pairs, select_all
from ._realize import (
    GaussTargets,
    NodeTargets,
    PaneTargets,
    RealizedLayer,
    RealizedPane,
    realize_pane,
    recorded_components,
)
from ._reconciler import LedgerBackend, SessionReconciler
from ._scope import ScopedSet, resolve_scope
from ._stills import render_still, resolve_pane

#: name → submodule for the lazily-exported Qt client surface.
_QT_EXPORTS = {
    "MeshPane": "._pane",
    "default_backend_factory": "._pane",
    "SessionOutline": "._outline",
    "MeshInspectorPage": "._inspector",
    "PanePlaceholderPage": "._inspector",
    "PlotInspectorPage": "._inspector",
    "SessionPaneFrame": "._frame",
    "PlotChart": "._chart",
    "PlotPane": "._chart",
    "PlotReconciler": "._chart",
    "PICK_BUTTONS": "._frame",
    "STYLE_BUTTONS": "._frame",
    "SessionPaneHost": "._host",
    "SessionPaneHostEmpty": "._host",
    "required_extent": "._host",
    "tile_columns": "._host",
    "tile_shape": "._host",
    "SessionResultsWindow": "._window",
    "SessionWindow": "._window",
    "show_session": "._window",
}


def __getattr__(name: str):
    submodule = _QT_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    return getattr(import_module(submodule, __name__), name)


def __dir__() -> "list[str]":
    return sorted(set(globals()) | set(_QT_EXPORTS) | set(__all__))


__all__ = [
    "PICK_BUTTONS",
    "STYLE_BUTTONS",
    "GaussTargets",
    "LedgerBackend",
    "MeshInspectorPage",
    "MeshPane",
    "NodeTargets",
    "PanePick",
    "PaneTargets",
    "PanePlaceholderPage",
    "PlotChart",
    "PlotInspectorPage",
    "PlotPane",
    "PlotReconciler",
    "RealizedLayer",
    "RealizedPane",
    "RealizedPlot",
    "RealizedSeries",
    "ScopedSet",
    "SessionOutline",
    "SessionPaneFrame",
    "SessionPaneHost",
    "SessionPaneHostEmpty",
    "SessionReconciler",
    "SessionResultsWindow",
    "SessionWindow",
    "default_backend_factory",
    "gauss_pairs",
    "realize_pane",
    "realize_plot",
    "recorded_components",
    "render_still",
    "required_extent",
    "resolve_pane",
    "resolve_scope",
    "select_all",
    "show_session",
    "tile_columns",
    "tile_shape",
]
