"""Deprecation shim for the Constraints umbrella relocation (Phase 8.1).

The constraint surface has been split three ways:

* :class:`ConstraintDef` and its subclasses moved to
  :mod:`apeGmsh.core.constraints.defs` (pre-mesh user-facing intent).
* :class:`ConstraintRecord` and its subclasses moved to
  :mod:`apeGmsh.mesh.records._constraints`
  (post-mesh solver-agnostic records, alongside the
  :class:`ConstraintKind` enum).
* :class:`ConstraintResolver` and the supporting geometric helpers
  moved to :mod:`apeGmsh.mesh._constraint_resolver` (broker-layer mesh
  math; pure numpy).

The :mod:`apeGmsh.mesh.records` package is now the canonical umbrella
re-export — ``import apeGmsh.mesh.records as Constraints`` gives the
same surface this module used to provide.

This shim re-exports that surface so legacy
``from apeGmsh.solvers.Constraints import …`` keeps working with a
one-shot :class:`DeprecationWarning` for one release cycle.
"""
from __future__ import annotations

import warnings

from apeGmsh.core.constraints.defs import (
    ConstraintDef,
    DistributingCouplingDef,
    EmbeddedDef,
    EqualDOFDef,
    KinematicCouplingDef,
    MortarDef,
    NodeToSurfaceDef,
    NodeToSurfaceSpringDef,
    PenaltyDef,
    RigidBodyDef,
    RigidDiaphragmDef,
    RigidLinkDef,
    TieDef,
    TiedContactDef,
)
from apeGmsh.mesh._constraint_resolver import (  # noqa: F401  (intentional re-exports of private helpers that callers used to import from this umbrella pre-split)
    SHAPE_FUNCTIONS,
    ConstraintResolver,
    _is_inside_parametric,
    _project_point_to_face,
    _shape_quad4,
    _shape_quad8,
    _shape_tri3,
    _shape_tri6,
    _SpatialIndex,
)
from apeGmsh.mesh.records._constraints import (
    ConstraintRecord,
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)

warnings.warn(
    "apeGmsh.solvers.Constraints is deprecated; import constraint defs "
    "from apeGmsh.core.constraints.defs, records from "
    "apeGmsh.mesh.records, and ConstraintResolver from "
    "apeGmsh.mesh._constraint_resolver.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Defs
    "ConstraintDef",
    "EqualDOFDef",
    "RigidLinkDef",
    "PenaltyDef",
    "RigidDiaphragmDef",
    "RigidBodyDef",
    "KinematicCouplingDef",
    "TieDef",
    "DistributingCouplingDef",
    "EmbeddedDef",
    "NodeToSurfaceDef",
    "NodeToSurfaceSpringDef",
    "TiedContactDef",
    "MortarDef",
    # Records
    "ConstraintRecord",
    "NodePairRecord",
    "NodeGroupRecord",
    "InterpolationRecord",
    "SurfaceCouplingRecord",
    "NodeToSurfaceRecord",
    # Resolver
    "ConstraintResolver",
    # Geom helpers (historically importable)
    "SHAPE_FUNCTIONS",
]
