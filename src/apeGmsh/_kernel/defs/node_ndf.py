"""Pre-mesh per-node ``ndf`` override definition.

Stores user intent at the label / PG / part level before meshing.
Resolved at FEM-build time into the per-node ``ndf`` vector on
:class:`~apeGmsh.mesh.FEMData.NodeComposite`.

Pure data container — no Gmsh dependency, no session plumbing.

This is the explicit-only side of the shell-to-solid coupling feature
(S1b).  apeGmsh declines to *infer* per-node DOF counts from element
class because the implicit-then-explicit hybrid the previous design
attempted (see PR #307, withdrawn) silently rewrote per-node ndf
whenever a 1D/2D label was renamed or a fragment_all() reshuffled
groups.  The user is the single source of truth here: every node
that needs an ``ndf`` must be covered by either an
``g.node_ndf.set(target, ndf=K)`` call or a
``g.node_ndf.set_default(ndf=K)`` blanket; nodes not covered raise
``LookupError`` from :meth:`NodeComposite.ndf_for`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeNDFDef:
    """User-supplied per-node ``ndf`` override.

    One def per call to :meth:`g.node_ndf.set` /
    :meth:`g.node_ndf.set_default`.  Multiple defs may target
    overlapping node sets; the last matching def wins (declaration
    order, consistent with ``g.constraints.bc`` and the loads/masses
    accumulators).  ``target`` is ``None`` for the default def.

    Parameters
    ----------
    target : object or None
        Label name, physical group name, part label, raw DimTag list,
        or mesh-selection name — resolved at FEM-build time via the
        same precedence chain used by loads/masses
        (``label -> PG -> part``).  ``None`` marks the default def
        (set via :meth:`g.node_ndf.set_default`) which applies to
        every node not covered by another def.
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
