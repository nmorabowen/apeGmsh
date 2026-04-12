"""
_Partitioning — mesh partitioning and node/element renumbering.

Accessed via ``g.mesh.partitioning``.  Groups the MPI-oriented
partition / unpartition entry points with the RCM / simple renumbering
helpers so the whole "reorganise the DOFs" surface lives in one place.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from .Mesh import Mesh


class _Partitioning:
    """Mesh partitioning plus node / element renumbering helpers."""

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # MPI-style partitioning
    # ------------------------------------------------------------------

    def partition(
        self,
        n_parts     : int,
        element_tags: list[int] | None = None,
        partitions  : list[int] | None = None,
    ) -> "_Partitioning":
        """
        Partition the mesh into ``n_parts`` sub-domains (e.g. for MPI runs).

        Parameters
        ----------
        n_parts      : number of partitions
        element_tags : elements to partition (``None`` = all)
        partitions   : pre-assigned partition IDs per element (``None`` = auto)
        """
        gmsh.model.mesh.partition(
            n_parts,
            elementTags=element_tags or [],
            partitions=partitions    or [],
        )
        self._mesh._log(f"partition(n_parts={n_parts})")
        return self

    def unpartition(self) -> "_Partitioning":
        """Remove the partition structure and restore a monolithic mesh."""
        gmsh.model.mesh.unpartition()
        self._mesh._log("unpartition()")
        return self

    # ------------------------------------------------------------------
    # Renumbering
    # ------------------------------------------------------------------

    def compute_renumbering(
        self,
        method      : str            = "RCMK",
        element_tags: list[int] | None = None,
    ) -> tuple[ndarray, ndarray]:
        """
        Compute an optimised node renumbering (e.g. RCMK bandwidth reduction).

        Returns
        -------
        (old_tags, new_tags) : two int ndarrays of equal length
        """
        old, new = gmsh.model.mesh.computeRenumbering(
            method=method, elementTags=element_tags or []
        )
        self._mesh._log(
            f"compute_renumbering(method={method!r}) -> {len(old)} nodes"
        )
        return np.array(old, dtype=np.int64), np.array(new, dtype=np.int64)

    def renumber_nodes(
        self,
        old_tags: list[int],
        new_tags: list[int],
    ) -> "_Partitioning":
        """Apply a pre-computed node renumbering."""
        gmsh.model.mesh.renumberNodes(oldTags=old_tags, newTags=new_tags)
        self._mesh._log(f"renumber_nodes({len(old_tags)} nodes)")
        return self

    def renumber_elements(
        self,
        old_tags: list[int],
        new_tags: list[int],
    ) -> "_Partitioning":
        """Apply a pre-computed element renumbering."""
        gmsh.model.mesh.renumberElements(oldTags=old_tags, newTags=new_tags)
        self._mesh._log(f"renumber_elements({len(old_tags)} elements)")
        return self

    def renumber_mesh(
        self,
        dim: int = 2,
        *,
        method: str = "simple",
        base: int = 1,
        used_only: bool = True,
    ) -> "_Partitioning":
        """
        Renumber nodes and elements in the Gmsh model to contiguous IDs.

        After this call, **all** Gmsh queries return solver-ready
        contiguous IDs directly.  This is a mutation of the Gmsh
        model — call it **once**, before extracting FEM data with
        :meth:`_Queries.get_fem_data`.

        Parameters
        ----------
        dim : int
            Element dimension used to build adjacency for RCM.
        method : ``"simple"`` or ``"rcm"``
        base : int
            Starting ID (default 1 = OpenSees/Abaqus convention).
        used_only : bool
            If True (default), only renumber nodes connected to at
            least one element.

        Example
        -------
        ::

            g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
            fem = g.mesh.queries.get_fem_data(dim=2)
        """
        from ..solvers.Numberer import Numberer
        raw = self._mesh._get_raw_fem_data(dim=dim)
        numb = Numberer(raw)
        info = numb.renumber(method=method, base=base, used_only=used_only)
        self._mesh._log(
            f"renumber_mesh(method={method!r}): "
            f"{info.n_nodes} nodes, {info.n_elems} elements, "
            f"bandwidth={info.bandwidth}"
        )
        return self
