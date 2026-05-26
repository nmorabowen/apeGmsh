"""apeGmsh.mesh.records — Resolved FEM records, kinds, and the
constraint umbrella.

Primary surface: the **resolved** dataclasses produced by the
resolvers after meshing (constraint, load, and mass records) plus
the :class:`ConstraintKind` / :class:`LoadKind` classifiers that
label them.

For convenience, this package also re-exports the constraint
:class:`ConstraintDef` hierarchy and :class:`ConstraintResolver` so
``import apeGmsh.mesh.records as Constraints`` provides the same
umbrella surface that the (deprecated)
:mod:`apeGmsh.solvers.Constraints` module used to.

User-facing **definition** dataclasses (``*Def``) live in
:mod:`apeGmsh.core.constraints.defs`, :mod:`apeGmsh.core.loads.defs`,
and :mod:`apeGmsh.core.masses.defs` — they describe pre-mesh intent.
The **resolvers** that translate defs into records live in
:mod:`apeGmsh.mesh._constraint_resolver`,
:mod:`apeGmsh.mesh._load_resolver`, and
:mod:`apeGmsh.mesh._mass_resolver`.
"""

from __future__ import annotations

from apeGmsh._kernel.defs.constraints import (
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

# ``SHAPE_FUNCTIONS`` / ``ConstraintResolver`` are re-exported LAZILY
# (PEP 562 ``__getattr__`` at the bottom of this module) rather than
# eagerly imported here.  Reason: ``resolvers._constraint_resolver._resolver``
# imports the resolved-record submodules from THIS package
# (``..records._constraints`` / ``._kinds``), so an eager
# ``from ..resolvers._constraint_resolver import ...`` at records-init
# time forms an init-order-dependent cycle
# (records/__init__ ⇄ _constraint_resolver/__init__).  Pre-P1-K this was
# masked only by a fragile load-order side-effect of the old
# ``apeGmsh.mesh`` package __init__ (FEMData→_record_set→records ran
# before _constraint_resolver resolved); ``_kernel`` is a clean leaf
# with no such side-effect, so the re-export is made lazy.  Umbrella
# surface (``__all__``, ``from …records import ConstraintResolver``)
# is byte-identical to callers — only the bind moment is deferred to
# first attribute access.
from ._constraints import (
    ConstraintRecord,
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from ._kinds import ConstraintKind, LoadKind
from ._loads import (
    ElementLoadRecord,
    LoadRecord,
    NodalLoadRecord,
    SPRecord,
)
from ._compose import ComposeRecord
from ._masses import MassRecord
from ._partitions import PartitionRecord


__all__ = [
    # Kind enums
    "ConstraintKind",
    "LoadKind",
    # Constraint defs (re-export from core.constraints.defs)
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
    # Constraint records
    "ConstraintRecord",
    "NodePairRecord",
    "NodeGroupRecord",
    "InterpolationRecord",
    "SurfaceCouplingRecord",
    "NodeToSurfaceRecord",
    # Constraint resolver + geom helper
    "ConstraintResolver",
    "SHAPE_FUNCTIONS",
    # Load records
    "LoadRecord",
    "NodalLoadRecord",
    "ElementLoadRecord",
    "SPRecord",
    # Mass records
    "MassRecord",
    # Partition records
    "PartitionRecord",
    # Compose records (Phase 3A.1)
    "ComposeRecord",
]


# PEP 562 lazy re-export of the constraint-resolver umbrella symbols.
# Deferred to first access so importing this package never eagerly
# pulls ``resolvers._constraint_resolver`` (which imports back into
# ``.._constraints`` / ``.._kinds``) — see the note above line ``from
# ._constraints import``.  Behaviour-identical to the previous eager
# ``from ..resolvers._constraint_resolver import SHAPE_FUNCTIONS,
# ConstraintResolver`` for every caller.
_LAZY = {"SHAPE_FUNCTIONS", "ConstraintResolver"}


def __getattr__(name: str):
    if name in _LAZY:
        from ..resolvers._constraint_resolver import (
            SHAPE_FUNCTIONS,
            ConstraintResolver,
        )
        globals()["SHAPE_FUNCTIONS"] = SHAPE_FUNCTIONS
        globals()["ConstraintResolver"] = ConstraintResolver
        return globals()[name]
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
