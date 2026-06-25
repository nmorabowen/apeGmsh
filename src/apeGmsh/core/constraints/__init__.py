"""apeGmsh.core.constraints — Model-level constraint definitions.

Hosts the user-facing :class:`ConstraintDef` subclasses that describe
*intent* at the geometry/PG/part level before meshing.  The post-mesh
**resolved** records live in :mod:`apeGmsh.mesh.records._constraints`;
the machinery that translates defs into records lives in
:mod:`apeGmsh.mesh._constraint_resolver`.
"""

from __future__ import annotations

from .defs import (
    BCDef,
    ConstraintDef,
    DistributingCouplingDef,
    EmbeddedDef,
    EqualDOFDef,
    EqualDOFMixedDef,
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


__all__ = [
    "ConstraintDef",
    "BCDef",
    "EqualDOFDef",
    "EqualDOFMixedDef",
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
]
