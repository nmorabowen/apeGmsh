"""Mass records — post-mesh resolved nodal masses.

The :class:`MassRecord` dataclass carries the per-node 6-DOF mass
vector produced by :class:`~apeGmsh.mesh._mass_resolver.MassResolver`
after meshing.  Records are solver-agnostic — any adapter (OpenSees,
Abaqus, Code_Aster, …) can consume them.

The corresponding pre-mesh :class:`MassDef` definitions live in
:mod:`apeGmsh.core.masses.defs`.

Columnar note (ADR 0065 v2 / plan_emit_memory_columnar.md C1–C3)
---------------------------------------------------------------
:class:`MassRecord` is the *view / API* type only.  At LOH.1 scale
(~7M nodes) keeping one resident dataclass per node cost ~3–5 GB, so
:class:`~apeGmsh._kernel.record_sets.MassSet` now stores its masses in
parallel numpy/dict columns and *constructs a MassRecord on the fly*
when it is iterated / indexed.  Consumers therefore see the same object
surface, but the yielded records are **transient** — never rely on
identity (``is``) or in-place mutation of a yielded record; treat them
as read-only value objects (dataclass ``__eq__`` still backs
``rec in fem.nodes.masses`` membership by value).  ``slots=True`` keeps
the transient records cheap and forbids ad-hoc attributes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(slots=True)
class MassRecord:
    """Resolved per-node mass entry.

    Always length 6: ``(mx, my, mz, Ixx, Iyy, Izz)``.  The OpenSees
    bridge slices to ``ndf`` when emitting commands (the rotational
    components are dropped for ``ndf<4`` models).

    Multiple :class:`MassDef` may contribute to the same node — the
    composite accumulates them so each node gets at most one
    :class:`MassRecord` in the final :class:`MassSet`.

    This is the read-only *view* type for the columnar
    :class:`~apeGmsh._kernel.record_sets.MassSet`; instances yielded by
    iterating a ``MassSet`` are transient (see the module docstring).
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
