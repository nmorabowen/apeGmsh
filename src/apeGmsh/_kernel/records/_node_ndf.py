"""apeGmsh._kernel.records._node_ndf — Frozen ``NodeNDFRecord``.

A :class:`NodeNDFRecord` is the immutable, ergonomic view of one
node's per-node ``ndf`` (number of DOFs) intent.  Lives on the
:class:`apeGmsh._kernel.record_sets.NodeNDFSet` composite exposed at
``fem.nodes.ndf_records()``.

The per-node ``ndf`` metadata is the broker-layer foundation for the
shell-to-solid coupling feature work stream (mixed-ndf modeling).
Today most apeGmsh models are uniform-ndf (every node has the same
DOF count), but shell-on-solid interfaces and 2D plane-stress models
introduce per-node variation:

- Shell elements need ``ndf=6`` at their nodes (3 trans + 3 rot)
- Solid elements need ``ndf=3`` (only translational)
- Interface nodes (shell base on solid top face) get ``ndf=6`` —
  the **max** of the incident element classes (shell rotational
  DOFs exist; solid simply ignores extras)
- 2D plane stress models would have ``ndf=2`` (all nodes uniform)

Hybrid populator (implicit + explicit)
--------------------------------------
The broker populates ``ndf`` per node via a two-stage rule:

1. **Implicit** — for each node, take the max of
   :data:`IMPLICIT_NDF_BY_DIM` over the dim of every incident element
   group.  Default table is ``{1: 6, 2: 6, 3: 3}`` (line elements
   treated as frame-in-3D; surface elements treated as shell;
   volume elements treated as solid).
2. **Explicit override** — a session-time
   ``g.model.set_node_ndf(target, ndf=K)`` declaration resolves to a
   set of node ids at extraction and overwrites the implicit value;
   such nodes carry ``source='explicit'``.

The ``source`` discriminator is preserved on the record so downstream
consumers (apeSees bridge, future ADR) can decide policy — e.g. error
if the user explicitly demanded a value the bridge cannot honour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


#: Default mapping from element-group ``dim`` to implicit per-node
#: ``ndf``.  Used by the broker extractor at FEM-build time when the
#: user has not pinned a node's ``ndf`` via the explicit session API.
#:
#: Heuristic rationale:
#:
#: - ``1`` (line elements) → ``6`` — assume frame-in-3D.  Truss-only
#:   models can override to ``3`` via ``g.model.set_node_ndf(...)``.
#: - ``2`` (surface elements) → ``6`` — assume shell.  2D plane-stress
#:   models can override to ``2``.
#: - ``3`` (volume elements) → ``3`` — solid elements never carry
#:   rotational DOFs.
#:
#: When a node is touched by multiple groups (e.g. a shell-on-solid
#: interface node belongs to both a 2D group and a 3D group), the
#: implicit value is the **max** over its incident groups — the shell
#: rotational DOFs exist at that node; the solid element merely
#: ignores them.
IMPLICIT_NDF_BY_DIM: dict[int, int] = {
    1: 6,  # line elements (frame in 3D; truss user-overrides to 3)
    2: 6,  # surface elements (assume shell; plane stress user-overrides to 2)
    3: 3,  # volume elements (solid)
}


@dataclass(frozen=True)
class NodeNDFRecord:
    """Immutable per-node ``ndf`` record.

    Parameters
    ----------
    node_id : int
        Mesh node ID.
    ndf : int
        Number of DOFs at this node (typically in ``{2, 3, 6}``).
    source : ``'implicit'`` or ``'explicit'``
        Whether this value was derived from the element-class
        heuristic (``IMPLICIT_NDF_BY_DIM`` over incident groups) or
        pinned explicitly by the user via
        ``g.model.set_node_ndf(target, ndf=K)``.
    """

    node_id: int
    ndf: int
    source: Literal["implicit", "explicit"]

    def __repr__(self) -> str:
        return (
            f"NodeNDFRecord(node_id={self.node_id}, "
            f"ndf={self.ndf}, source={self.source!r})"
        )


__all__ = ["NodeNDFRecord", "IMPLICIT_NDF_BY_DIM"]
