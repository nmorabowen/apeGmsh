"""apeGmsh.mesh._constraint_resolver — Broker-layer constraint resolution.

Converts pre-mesh :class:`~apeGmsh.core.constraints.defs.ConstraintDef`
intents into resolved
:class:`~apeGmsh.mesh.records._constraints.ConstraintRecord` objects
by attaching mesh data (node tags, coordinates, connectivity) and
running the appropriate geometric search / projection per kind.

The package exposes :class:`ConstraintResolver` plus a few
shape-function / spatial-index helpers that historically leaked
out of the old ``apeGmsh.solvers.Constraints`` umbrella.
"""

from __future__ import annotations

from ._geom import (
    SHAPE_FUNCTIONS,
    _SpatialIndex,
    _is_inside_parametric,
    _project_point_to_face,
    _shape_quad4,
    _shape_quad8,
    _shape_tri3,
    _shape_tri6,
)
from ._resolver import ConstraintResolver


__all__ = [
    "ConstraintResolver",
    "SHAPE_FUNCTIONS",
    "_SpatialIndex",
    "_project_point_to_face",
    "_is_inside_parametric",
    "_shape_tri3",
    "_shape_quad4",
    "_shape_tri6",
    "_shape_quad8",
]
