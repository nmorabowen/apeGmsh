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
from typing import ClassVar


@dataclass
class LoadRecord:
    """Base class for all resolved load records."""
    kind: str
    pattern: str = "default"
    name: str | None = None

    # ADR 0038 §"Tag-reference rewrite checklist" — Phase 3B.2a
    # rewrite cover-set declaration.  Subclasses override.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": (),
        "tag_fields_array": (),
        "name_fields": (),
    }


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

    # ADR 0038 §"Tag-reference rewrite checklist" — node_id is a tag
    # reference.  ``name`` is the optional caller label; ``pattern`` is
    # the load-pattern name — both get namespace-prefixed.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("node_id",),
        "tag_fields_array": (),
        "name_fields": ("name", "pattern"),
    }


@dataclass
class ElementLoadRecord(LoadRecord):
    """Element-level load command (e.g. ``eleLoad -beamUniform``)."""
    kind: str = field(init=False, default="element")
    element_id: int = 0
    load_type: str = ""             # "beamUniform", "surfacePressure", "bodyForce", ...
    params: dict = field(default_factory=dict)

    # ADR 0038 §"Tag-reference rewrite checklist" — element_id is a
    # tag reference; pattern + name are string-keyed surfaces.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("element_id",),
        "tag_fields_array": (),
        "name_fields": ("name", "pattern"),
    }


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

    # ADR 0038 §"Tag-reference rewrite checklist" — node_id is a tag
    # reference; pattern + name are string-keyed surfaces.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("node_id",),
        "tag_fields_array": (),
        "name_fields": ("name", "pattern"),
    }


__all__ = [
    "LoadRecord",
    "NodalLoadRecord",
    "ElementLoadRecord",
    "SPRecord",
]
