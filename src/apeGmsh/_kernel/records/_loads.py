"""Load records — post-mesh resolved nodal/element loads.

These dataclasses carry the concrete output of
:class:`~apeGmsh.mesh._load_resolver.LoadResolver` after meshing.
Records are solver-agnostic — any adapter (OpenSees, Abaqus,
Code_Aster, …) can consume them.

The corresponding pre-mesh :class:`LoadDef` definitions live in
:mod:`apeGmsh.core.loads.defs`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoadRecord:
    """Base class for all resolved load records."""
    kind: str
    pattern: str = "default"
    name: str | None = None


@dataclass
class NodalLoadRecord(LoadRecord):
    """Force and/or moment at a single node.

    ``force_xyz`` and ``moment_xyz`` are pure 3D spatial vectors (or
    ``None`` when absent). The record is DOF-agnostic — mapping onto
    a solver's DOF space is the caller's responsibility.
    """
    kind: str = field(init=False, default="nodal")
    node_id: int = 0
    force_xyz: tuple[float, float, float] | None = None
    moment_xyz: tuple[float, float, float] | None = None


@dataclass
class ElementLoadRecord(LoadRecord):
    """Element-level load command (e.g. ``eleLoad -beamUniform``)."""
    kind: str = field(init=False, default="element")
    element_id: int = 0
    load_type: str = ""             # "beamUniform", "surfacePressure", "bodyForce", ...
    params: dict = field(default_factory=dict)


@dataclass
class SPRecord(LoadRecord):
    """Single-point constraint: prescribed displacement or homogeneous fix.

    One record per DOF per node.  When ``is_homogeneous`` is ``True``
    the downstream emitter can use ``ops.fix()``; otherwise it must use
    ``ops.sp(node, dof, value)``.
    """
    kind: str = field(init=False, default="sp")
    node_id: int = 0
    dof: int = 1                    # 1-based DOF index
    value: float = 0.0
    is_homogeneous: bool = True


__all__ = [
    "LoadRecord",
    "NodalLoadRecord",
    "ElementLoadRecord",
    "SPRecord",
]
