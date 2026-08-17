"""``apeGmsh.results.session`` — the Results presentation library.

ADR 0098: presentation is a :class:`ResultsSession` of views, not
geometries of diagrams. ``Results`` answers "what did the solver
write?"; the session answers "what is on screen, at which time, with
which pictures, with which pick". The Qt window (S2+), stills (S1) and
the Studio MCP (S5) are clients that project this state — VTK actors,
docks and widgets are never truth.

Laws this package encodes (S0):

* §4 — slots: a closed seven-category catalog, at most one occupant
  per category per mesh view; filling replaces; deform is a pose, not
  a slot.
* §5 — legends: derived from occupied colour-mapped slots, per view;
  same field on two slots → one scale; hide is chrome, not a clear.
* §7 — time: an instant is ``(stage, step)``; linked → one instant for
  every pane; a mode pose has no instant.
* §8 — selection: nodes XOR Gauss in ADR 0045's ONE store; the kind
  follows the last writer.

Layering exception (recorded in S0's PR, plan decision 1): this
package imports ``apeGmsh.viewers.core.selection`` and
``apeGmsh.viewers.scene_ir`` — ``SelectionState`` is already the one
store (INV-5) and both are Qt/VTK-free, so a Protocol seam would add
indirection nothing needs. Nothing here may import
``apeGmsh.viewers.diagrams`` or any Qt/VTK module (enforced by
``tests/results/session/test_session_pure.py``); the old ontology is
not reused.

``realize()`` / ``render()`` (S1) delegate to the viewers-side
projection ``apeGmsh.viewers.session`` via call-time imports — the
per-kind emit reuse lives THERE, so this package keeps the guarantee
above. ``show()`` (Qt) is S2; the snapshot is S5. The old
``results.viewer()`` window is untouched until the S6 flip.
"""
from __future__ import annotations

from ._selection import SessionSelection, gauss_target, node_target
from ._session import Pane, ResultsSession
from ._slots import (
    COLOUR_MAPPED,
    SLOT_CATALOG,
    Contour,
    Gauss,
    Line,
    Loads,
    Reactions,
    Sand,
    Slot,
    Vector,
    slot_field,
)
from ._time import Instant
from ._views import (
    PICK_TARGETS,
    PLOT_KINDS,
    Deform,
    Legend,
    MeshStyle,
    MeshView,
    PlotSeries,
    PlotSource,
    PlotView,
    Scope,
    ViewClip,
)

__all__ = [
    # session
    "ResultsSession",
    "Pane",
    # time
    "Instant",
    # views
    "MeshView",
    "PlotView",
    "Deform",
    "MeshStyle",
    "Scope",
    "ViewClip",
    "Legend",
    "PICK_TARGETS",
    "PLOT_KINDS",
    "PlotSource",
    "PlotSeries",
    # slots
    "Slot",
    "Contour",
    "Vector",
    "Gauss",
    "Line",
    "Sand",
    "Loads",
    "Reactions",
    "SLOT_CATALOG",
    "COLOUR_MAPPED",
    "slot_field",
    # selection
    "SessionSelection",
    "node_target",
    "gauss_target",
]
