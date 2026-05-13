"""Read-only structural data for the viewer package.

Phase 8.7 commit 3 introduces :class:`ViewerData` as the viewer's
single point of consumption — the seam that decouples viewer code
from :class:`apeGmsh.mesh.FEMData` and lets a ``model.h5`` file alone
drive the viewer (Phase 8.7 acceptance criterion;
[ADR 0014](../../opensees/architecture/decisions/0014-viewer-is-pure-h5-consumer.md)).

Two builders:

* :meth:`ViewerData.from_fem(fem)` — live FEMData path (g.mesh.viewer).
* :meth:`ViewerData.from_h5(path)` — ``model.h5`` path (results /
  fixtures).

The row dataclasses (:class:`NodalLoadRow`, :class:`NodePairRow`,
:class:`InterpolationRow`, …) are re-exported so callers can
type-check iteration without reaching into the private
``_records.py`` module.
"""
from __future__ import annotations

from ._elements import (
    ElementLoadView,
    SurfaceConstraintView,
    ViewerElementGroup,
    ViewerElements,
    ViewerElementType,
)
from ._nodes import (
    MassView,
    NodalLoadView,
    NodeConstraintView,
    SPView,
    ViewerNodes,
)
from ._records import (
    ConstraintRow,
    ElementLoadRow,
    InterpolationRow,
    MassRow,
    NodalLoadRow,
    NodeGroupRow,
    NodePairRow,
    NodeToSurfaceRow,
    SPRow,
    SurfaceCouplingRow,
    ViewerDataDecodeError,
)
from ._viewer_data import ViewerData

__all__ = [
    "ConstraintRow",
    "ElementLoadRow",
    "ElementLoadView",
    "InterpolationRow",
    "MassRow",
    "MassView",
    "NodalLoadRow",
    "NodalLoadView",
    "NodeConstraintView",
    "NodeGroupRow",
    "NodePairRow",
    "NodeToSurfaceRow",
    "SPRow",
    "SPView",
    "SurfaceConstraintView",
    "SurfaceCouplingRow",
    "ViewerData",
    "ViewerDataDecodeError",
    "ViewerElementGroup",
    "ViewerElementType",
    "ViewerElements",
    "ViewerNodes",
]
