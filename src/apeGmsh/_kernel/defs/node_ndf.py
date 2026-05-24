"""Pre-mesh per-node ``ndf`` override definition.

Stores user intent at the geometry / PG / label / part level before
meshing.  Resolved at FEM-build time into the per-node ``ndf`` vector
on :class:`~apeGmsh.mesh.FEMData.NodeComposite` — the resolved record
counterpart is :class:`~apeGmsh._kernel.records._node_ndf.NodeNDFRecord`.

This def is the "explicit" side of the hybrid populator described in
:mod:`apeGmsh._kernel.records._node_ndf`; the implicit side is the
element-class heuristic (``IMPLICIT_NDF_BY_DIM``) that runs first.

Pure data container — no Gmsh dependency, no session plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeNDFDef:
    """User-supplied per-node ``ndf`` override.

    One def per call to ``g.model.set_node_ndf(target, ndf=K)``.
    Multiple defs may target overlapping node sets; the last
    matching def wins (consistent with the imperative ``set_*``
    semantics — see ``g.constraints.bc`` for a similar pattern).

    Parameters
    ----------
    target : object
        Label name, physical group name, part label, raw DimTag
        list, or mesh-selection name.  Resolved at FEM-build time
        via the same precedence chain used by loads/masses
        (``label -> PG -> part``).
    ndf : int
        Number of DOFs to assign to every node resolved from
        ``target``.  Typically in ``{2, 3, 6}``.
    name : str, optional
        Friendly label for debugging / inspection.
    """

    target: object
    ndf: int
    name: str | None = None


__all__ = ["NodeNDFDef"]
