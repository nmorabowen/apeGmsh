"""Mass records — post-mesh resolved nodal masses.

The :class:`MassRecord` dataclass carries the per-node 6-DOF mass
vector produced by :class:`~apeGmsh.mesh._mass_resolver.MassResolver`
after meshing.  Records are solver-agnostic — any adapter (OpenSees,
Abaqus, Code_Aster, …) can consume them.

The corresponding pre-mesh :class:`MassDef` definitions live in
:mod:`apeGmsh.core.masses.defs`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class MassRecord:
    """Resolved per-node mass entry.

    Always length 6: ``(mx, my, mz, Ixx, Iyy, Izz)``.  The OpenSees
    bridge slices to ``ndf`` when emitting commands (the rotational
    components are dropped for ``ndf<4`` models).

    Multiple :class:`MassDef` may contribute to the same node — the
    composite accumulates them so each node gets at most one
    :class:`MassRecord` in the final :class:`MassSet`.
    """
    node_id: int = 0
    mass: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    name: str | None = None

    # ADR 0038 §"Tag-reference rewrite checklist" — node_id is the
    # tag reference; ``name`` is the optional caller label.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("node_id",),
        "tag_fields_array": (),
        "name_fields": ("name",),
    }


__all__ = ["MassRecord"]
