from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gmsh
import pandas as pd

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Tag    = int
DimTag = tuple[int, int]


class Partition:
    """
    Mesh-partitioning composite attached to a ``apeGmsh`` instance as
    ``g.partition``.

    Wraps ``gmsh.model.mesh.partition`` / ``unpartition`` and the
    partition-aware model queries exposed after partitioning.

    Workflow
    --------
    ::

        with apeGmsh(model_name="Cube") as g:
            # build and mesh
            g.model.add_box(0, 0, 0, 1, 1, 1)
            g.model.sync()
            g.mesh.generate(3)

            # auto-partition into 4 parts
            g.partition.auto(4)

            # inspect
            print(g.partition.summary())
            df = g.partition.entity_table()

            # save — one combined file or one per partition
            g.partition.save("cube_part.msh")
            g.partition.save("cube", one_file_per_partition=True)

    Or with an explicit element → partition assignment::

            elem_tags  = [1, 2, 3, 4, 5, 6]
            part_ids   = [1, 1, 2, 2, 3, 4]
            g.partition.explicit(4, element_tags=elem_tags,
                                    partitions=part_ids)

    Parameters
    ----------
    parent : _SessionBase
        The owning instance.
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[Partition] {msg}")

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def auto(self, n_parts: int) -> Partition:
        """
        Partition the current mesh into *n_parts* sub-domains using
        Gmsh's built-in partitioner (Metis when available).

        Must be called after ``g.mesh.generate()``.

        Parameters
        ----------
        n_parts : number of partitions (≥ 1)

        Returns
        -------
        self — for chaining
        """
        if n_parts < 1:
            raise ValueError(f"auto(): n_parts must be ≥ 1, got {n_parts}")
        gmsh.model.mesh.partition(n_parts)
        self._log(f"auto(n_parts={n_parts})")
        return self

    def explicit(
        self,
        n_parts     : int,
        *,
        element_tags: list[int],
        partitions  : list[int],
    ) -> Partition:
        """
        Partition the mesh with an explicit per-element assignment.

        Parameters
        ----------
        n_parts      : total number of partitions declared
        element_tags : list of element tags to assign
        partitions   : parallel list of partition IDs (1-based) for each
                       element in *element_tags*

        Returns
        -------
        self — for chaining

        Example
        -------
        ::

            g.partition.explicit(
                2,
                element_tags=[1, 2, 3, 4],
                partitions  =[1, 1, 2, 2],
            )
        """
        if len(element_tags) != len(partitions):
            raise ValueError(
                f"explicit(): len(element_tags)={len(element_tags)} != "
                f"len(partitions)={len(partitions)}"
            )
        gmsh.model.mesh.partition(
            n_parts,
            elementTags=element_tags,
            partitions=partitions,
        )
        self._log(
            f"explicit(n_parts={n_parts}, "
            f"n_elements={len(element_tags)})"
        )
        return self

    def unpartition(self) -> Partition:
        """
        Remove the partition structure and restore a monolithic mesh.

        Returns
        -------
        self — for chaining
        """
        gmsh.model.mesh.unpartition()
        self._log("unpartition()")
        return self

    # ------------------------------------------------------------------
    # Renumbering (companion operations)
    # ------------------------------------------------------------------

    def renumber(
        self,
        *,
        method      : str            = "RCMK",
        element_tags: list[int] | None = None,
    ) -> Partition:
        """
        Compute and apply a node renumbering (e.g. RCMK bandwidth
        reduction) — commonly run after partitioning to improve solver
        performance.

        Parameters
        ----------
        method       : renumbering algorithm passed to
                       ``gmsh.model.mesh.computeRenumbering``.
                       Supported values: ``"RCMK"`` (bandwidth reduction),
                       ``"Hilbert"`` (space-filling curve, good cache
                       locality), ``"Metis"`` (graph-partitioner ordering).
                       ``"RCMK"`` is the most commonly used default.
        element_tags : restrict to a subset of elements (``None`` = all)

        Returns
        -------
        self — for chaining
        """
        old_tags, new_tags = gmsh.model.mesh.computeRenumbering(
            method=method,
            elementTags=element_tags or [],
        )
        gmsh.model.mesh.renumberNodes(
            oldTags=list(old_tags),
            newTags=list(new_tags),
        )
        self._log(
            f"renumber(method={method!r}, "
            f"n_nodes={len(old_tags)})"
        )
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def n_partitions(self) -> int:
        """Return the current number of partitions (0 if not partitioned)."""
        return gmsh.model.getNumberOfPartitions()

    def get_partitions(self, dim: int, tag: Tag) -> list[int]:
        """
        Return the partition IDs that contain a given model entity.

        Parameters
        ----------
        dim : entity dimension
        tag : entity tag

        Returns
        -------
        list[int]  partition IDs (empty for non-partitioned entities)
        """
        return list(gmsh.model.getPartitions(dim, tag))

    def get_parent(self, dim: int, tag: Tag) -> DimTag:
        """
        Return the ``(dim, tag)`` of the parent entity of a partitioned
        sub-entity.

        Parameters
        ----------
        dim : dimension of the partitioned entity
        tag : tag of the partitioned entity
        """
        p_dim, p_tag = gmsh.model.getParent(dim, tag)
        return (p_dim, p_tag)

    def entity_table(self, dim: int = -1) -> pd.DataFrame:
        """
        Build a DataFrame of all model entities and their partition
        membership.

        Parameters
        ----------
        dim : restrict to a single dimension (``-1`` = all)

        Returns
        -------
        pd.DataFrame  with columns:

        ``dim``         entity dimension
        ``tag``         entity tag
        ``partitions``  comma-separated partition IDs (empty string = unpartitioned)
        ``parent_dim``  parent entity dimension (``-1`` if top-level)
        ``parent_tag``  parent entity tag (``-1`` if top-level)
        """
        rows: list[dict] = []
        entities = (
            gmsh.model.getEntities(dim=dim)
            if dim != -1
            else gmsh.model.getEntities()
        )
        for ent_dim, ent_tag in entities:
            try:
                parts = list(gmsh.model.getPartitions(ent_dim, ent_tag))
            except Exception:
                parts = []
            try:
                p_dim, p_tag = gmsh.model.getParent(ent_dim, ent_tag)
            except Exception:
                p_dim, p_tag = -1, -1
            rows.append({
                'dim'        : ent_dim,
                'tag'        : ent_tag,
                'partitions' : ", ".join(str(p) for p in parts),
                'parent_dim' : p_dim,
                'parent_tag' : p_tag,
            })

        if not rows:
            return pd.DataFrame(
                columns=['dim', 'tag', 'partitions', 'parent_dim', 'parent_tag']
            )
        return pd.DataFrame(rows).set_index(['dim', 'tag'])

    def summary(self) -> str:
        """
        Return a concise text summary of the current partition state:
        number of partitions and a per-dimension entity count.
        """
        n = self.n_partitions()
        if n == 0:
            return (
                f"Partition(model={self._parent.model_name!r}): "
                f"not partitioned"
            )
        lines = [
            f"Partition(model={self._parent.model_name!r}): "
            f"{n} partition(s)",
        ]
        df = self.entity_table()
        if not df.empty:
            # count entities that belong to at least one partition
            partitioned = df[df['partitions'] != '']
            counts = (
                partitioned
                .reset_index()
                .groupby('dim')
                .size()
                .rename(index={0:'points',1:'curves',2:'surfaces',3:'volumes'})
            )
            for dim_label, count in counts.items():
                lines.append(f"  {dim_label:10s}: {count} partitioned entities")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def save(
        self,
        path                  : Path | str,
        *,
        one_file_per_partition: bool = False,
        create_topology       : bool = False,
        create_physicals      : bool = True,
    ) -> Partition:
        """
        Write the partitioned mesh to file(s).

        Parameters
        ----------
        path                   : output file path or base name.
                                 The format is inferred from the extension
                                 (``".msh"`` is the natural choice for
                                 partitioned meshes).
        one_file_per_partition : when ``True``, Gmsh writes one file per
                                 partition alongside the combined file.
                                 The per-partition files are named
                                 ``<stem>_<k><suffix>`` (e.g.
                                 ``mesh_1.msh``, ``mesh_2.msh`` …).
        create_topology        : pass to ``Mesh.PartitionCreateTopology``
        create_physicals       : pass to ``Mesh.PartitionCreatePhysicals``
                                 (keep ``True`` so solvers can find BC tags)

        Returns
        -------
        self — for chaining
        """
        path = Path(path)

        gmsh.option.setNumber(
            "Mesh.PartitionCreateTopology", int(create_topology)
        )
        gmsh.option.setNumber(
            "Mesh.PartitionCreatePhysicals", int(create_physicals)
        )
        gmsh.option.setNumber(
            "Mesh.PartitionSplitMeshFiles",  int(one_file_per_partition)
        )

        gmsh.write(str(path))
        self._log(
            f"save({path}, "
            f"one_file_per_partition={one_file_per_partition})"
        )
        return self

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self.n_partitions()
        return (
            f"Partition(model={self._parent.model_name!r}, "
            f"n_partitions={n})"
        )
