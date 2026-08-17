"""``apeGmsh.viewers.session`` — realize clients of the ResultsSession.

The projection side of ADR 0098: the session IR
(``apeGmsh.results.session``) is truth; this package turns it into
scene-IR layers (``realize_pane``) and offscreen stills
(``render_still``), reusing the existing per-kind emit code in place,
director-free. It lives under ``apeGmsh.viewers`` — not under the
session package — so the session IR never imports
``viewers.diagrams`` (the ``test_session_pure.py`` guard);
``ResultsSession.realize`` / ``.render`` reach in here lazily.

The Qt pane client (S2) joins this package as a further projection.
"""
from __future__ import annotations

from ._realize import RealizedLayer, RealizedPane, realize_pane
from ._stills import render_still, resolve_pane

__all__ = [
    "RealizedLayer",
    "RealizedPane",
    "realize_pane",
    "render_still",
    "resolve_pane",
]
