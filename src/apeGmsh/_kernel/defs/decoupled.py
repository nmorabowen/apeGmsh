"""Decoupled-node definitions — pre-mesh user-facing intent (ADR 0049).

A *decoupled node* is an auxiliary node that is **not** a Gmsh mesh
vertex: a spring/dashpot ground for SSI, a rigidDiaphragm master, a
control node, or a load/mass anchor.  The user declares one on the
session via ``g.decouple_node(coords=... | point=..., label=...)``; the
FEM factory appends it to the broker's node arrays at extraction time
(where the Gmsh-gathered nodes enter), assigning a deterministic tag
above every mesh node so it is dedup-immune by construction.

PR-4 scope is **identity only** — coordinates, an optional friendly
label, and the resolved tag.  It carries **no** ``ndf``/DOF count: per
the ADR 0049 redesign, DOF lives on the bridge (``ops.ndf``), never on
the session or the neutral broker.

These classes have no Gmsh dependency, no session plumbing, and no
factory methods — pure data containers consumed by the FEM factory.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class DecoupledNodeDef:
    """A single user-declared decoupled node (identity only).

    Exactly one of ``coords`` / ``point`` locates the node:

    * ``coords=(x, y, z)`` — an explicit coordinate triple.
    * ``point="label"`` — a geometry point label, resolved to its
      coordinates **at mesh-extraction time** (a snapshot, not tracked
      through later transforms).

    ``label`` is an optional friendly name for the node (distinct from
    ``point``, which is a *geometry* label used only to locate it).

    ``tag`` is ``None`` until the FEM factory resolves the model and
    assigns a deterministic tag; the factory writes it back onto this
    def so the handle returned from ``g.decouple_node(...)`` exposes the
    final tag after meshing.
    """
    coords: tuple[float, float, float] | None = None
    point: str | None = None
    label: str | None = None
    tag: int | None = None


@dataclass
class DecoupledNodeSetDef:
    """One decoupled node per node of a mesh node set (ADR 0118 D1).

    Declared by name through ``g.decouple_node_set(source, ...)`` and
    resolved by the FEM factory at extraction, when the source nodes
    exist:

    * ``source`` — a label or physical-group name; its mesh nodes, in
      ascending tag order, are the *source nodes*.
    * ``offset`` — a ``(dx, dy, dz)`` triple, or a callable taking the
      ``(n, 3)`` source coordinates and returning ``(n, 3)`` offsets.
      Each new node sits at its source node's coordinates plus offset.
    * ``label`` — an optional friendly name (informational: constraint
      verbs do not resolve it as a role).
    * ``tie_dofs`` — when set, one ``equal_dof`` record per pair
      (retained = source node, constrained = new node) on these dofs.

    ``source_ids`` and ``tags`` are ``None`` until the factory resolves
    the model; then they hold the source node tags and the new node
    tags, paired by index.  Like :class:`DecoupledNodeDef`, the set
    carries no ``ndf`` (ADR 0049).
    """
    source: str
    offset: "tuple[float, float, float] | Callable[[Any], Any]" = (0.0, 0.0, 0.0)
    label: str | None = None
    tie_dofs: tuple[int, ...] | None = None
    source_ids: tuple[int, ...] | None = None
    tags: tuple[int, ...] | None = None

    def pairs(self) -> dict[int, int]:
        """``{source node tag: new node tag}``; raises before extraction."""
        if self.source_ids is None or self.tags is None:
            raise ValueError(
                f"decoupled node set {self.label or self.source!r} has no "
                f"resolved tags — call g.mesh.queries.get_fem_data(...) first."
            )
        return dict(zip(self.source_ids, self.tags))


__all__ = ["DecoupledNodeDef", "DecoupledNodeSetDef"]
