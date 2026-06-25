"""apeGmsh.core.constraints.defs — backward-compatible import shim.

The constraint-definition dataclasses were relocated to
:mod:`apeGmsh._kernel.defs.constraints` (selection-unification-v2 P1-K,
the keystone cycle-break).  Class identity is unchanged — only the
module path moved.

This module is a thin **downward** re-export (``core`` → ``_kernel``,
the intended layering direction) so that the public
``apeGmsh.core.constraints.defs`` path and the byte-unchanged contract
tests (``test_resolution_contract.py`` imports ``ConstraintDef`` from
here) keep resolving.  Flagged as a P3/P4 internal-cleanup candidate
(sweep with the legacy surface).
"""

from __future__ import annotations

from apeGmsh._kernel.defs.constraints import (  # noqa: F401
    BCDef,
    ConstraintDef,
    DistributingCouplingDef,
    EmbeddedDef,
    EqualDOFDef,
    EqualDOFMixedDef,
    KinematicCouplingDef,
    NodeToSurfaceDef,
    NodeToSurfaceSpringDef,
    PenaltyDef,
    ReinforceDef,
    RigidBodyDef,
    RigidDiaphragmDef,
    RigidLinkDef,
    TieDef,
    TiedContactDef,
)
from apeGmsh._kernel.defs.constraints import __all__ as __all__  # noqa: F401
