"""
_Partitioning — mesh partitioning and node/element renumbering.

Accessed via ``g.mesh.partitioning``.  The single home for:

* **Renumbering** — contiguous IDs (``simple``) or bandwidth-optimised
  orderings (``rcm``, ``hilbert``, ``metis``).  Mutates the Gmsh model
  so that ``get_fem_data()`` produces solver-ready tags.
* **Partitioning** — MPI-style domain decomposition via Gmsh/METIS.
  Partition membership is captured in ``FEMData`` and queryable via
  ``fem.nodes.get(partition=2)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gmsh
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from .Mesh import Mesh


# =====================================================================
# Output contracts
# =====================================================================

class RenumberResult:
    """Result of a mesh renumbering operation.

    Attributes
    ----------
    method : str
        Algorithm used (``"simple"``, ``"rcm"``, ``"hilbert"``, ``"metis"``).
    n_nodes : int
        Number of nodes renumbered.
    n_elements : int
        Number of elements renumbered.
    bandwidth_before : int
        Semi-bandwidth before renumbering.
    bandwidth_after : int
        Semi-bandwidth after renumbering.
    """

    __slots__ = ('method', 'n_nodes', 'n_elements',
                 'bandwidth_before', 'bandwidth_after')

    def __init__(
        self,
        method: str,
        n_nodes: int,
        n_elements: int,
        bandwidth_before: int,
        bandwidth_after: int,
    ) -> None:
        self.method = method
        self.n_nodes = n_nodes
        self.n_elements = n_elements
        self.bandwidth_before = bandwidth_before
        self.bandwidth_after = bandwidth_after

    def __repr__(self) -> str:
        if self.bandwidth_after > 0:
            ratio = self.bandwidth_before / self.bandwidth_after
            return (
                f"RenumberResult({self.method}): "
                f"{self.n_nodes} nodes, {self.n_elements} elements, "
                f"bw {self.bandwidth_before}\u2192{self.bandwidth_after} "
                f"({ratio:.1f}\u00d7)")
        return (
            f"RenumberResult({self.method}): "
            f"{self.n_nodes} nodes, {self.n_elements} elements, "
            f"bw {self.bandwidth_before}\u2192{self.bandwidth_after}")


class PartitionInfo:
    """Result of a mesh partitioning operation.

    Attributes
    ----------
    n_parts : int
        Number of partitions created.
    elements_per_partition : dict[int, int]
        ``{partition_id: element_count}``.
    """

    __slots__ = ('n_parts', 'elements_per_partition')

    def __init__(
        self,
        n_parts: int,
        elements_per_partition: dict[int, int],
    ) -> None:
        self.n_parts = n_parts
        self.elements_per_partition = elements_per_partition

    def __repr__(self) -> str:
        counts = ", ".join(
            f"P{k}:{v}"
            for k, v in sorted(self.elements_per_partition.items()))
        return f"PartitionInfo({self.n_parts} parts: {counts})"


# =====================================================================
# Gmsh method name mapping
# =====================================================================

_METHOD_MAP: dict[str, str] = {
    "rcm":     "RCMK",
    "hilbert": "Hilbert",
    "metis":   "Metis",
}


# =====================================================================
# Composite
# =====================================================================

class _Partitioning:
    """Mesh partitioning plus node / element renumbering.

    Accessed via ``g.mesh.partitioning``.
    """

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # Renumbering
    # ------------------------------------------------------------------

    def renumber(
        self,
        dim: int = 2,
        *,
        method: str = "rcm",
        base: int = 1,
    ) -> RenumberResult:
        """Renumber nodes and elements in the Gmsh model.

        After this call every Gmsh query returns solver-ready contiguous
        IDs.  Call **once**, before extracting FEM data with
        :meth:`~_Queries.get_fem_data`.

        Parameters
        ----------
        dim : int
            Element dimension used to compute bandwidth and to collect
            element tags for renumbering.
        method : ``"simple"`` | ``"rcm"`` | ``"hilbert"`` | ``"metis"``
            ``"simple"``  — contiguous IDs, no optimisation.
            ``"rcm"``     — Reverse Cuthill-McKee (bandwidth reduction).
            ``"hilbert"`` — Hilbert space-filling curve (cache locality).
            ``"metis"``   — METIS graph-partitioner ordering.
        base : int
            Starting ID (default 1 = OpenSees / Abaqus convention).

        Returns
        -------
        RenumberResult
        """
        from ._fem_extract import extract_raw
        from .FEMData import _compute_bandwidth

        # 1. Bandwidth BEFORE ────────────────────────────────────
        raw = extract_raw(dim=dim)
        bw_before = _compute_bandwidth(raw['connectivity'])
        n_nodes = len(raw['node_tags'])
        n_elems = len(raw['elem_tags'])

        # 2. Node renumbering ────────────────────────────────────
        if method == "simple":
            self._renumber_nodes_simple(base)
        elif method in _METHOD_MAP:
            old, new = gmsh.model.mesh.computeRenumbering(
                method=_METHOD_MAP[method])
            gmsh.model.mesh.renumberNodes(
                oldTags=list(old), newTags=list(new))
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"Use 'simple', 'rcm', 'hilbert', or 'metis'.")

        # 3. Element renumbering (always simple contiguous) ──────
        self._renumber_elements_simple(dim, base)

        # 4. Bandwidth AFTER ─────────────────────────────────────
        raw_after = extract_raw(dim=dim)
        bw_after = _compute_bandwidth(raw_after['connectivity'])

        result = RenumberResult(
            method=method,
            n_nodes=n_nodes,
            n_elements=n_elems,
            bandwidth_before=bw_before,
            bandwidth_after=bw_after,
        )
        self._mesh._log(
            f"renumber(method={method!r}, dim={dim}): "
            f"{n_nodes} nodes, {n_elems} elements, "
            f"bw {bw_before}\u2192{bw_after}")
        return result

    # ── internal helpers ─────────────────────────────────────────

    @staticmethod
    def _renumber_nodes_simple(base: int) -> None:
        """Assign contiguous node tags starting from *base*."""
        tags, _, _ = gmsh.model.mesh.getNodes()
        old = np.sort(np.asarray(tags, dtype=np.int64))
        new = np.arange(base, base + len(old), dtype=np.int64)
        gmsh.model.mesh.renumberNodes(
            oldTags=old.tolist(), newTags=new.tolist())

    @staticmethod
    def _renumber_elements_simple(dim: int, base: int) -> None:
        """Assign contiguous element tags for *dim* starting from *base*."""
        _, etags_list, _ = gmsh.model.mesh.getElements(dim=dim, tag=-1)
        all_tags: list[int] = []
        for etags in etags_list:
            all_tags.extend(int(t) for t in etags)
        if not all_tags:
            return
        old = np.array(sorted(all_tags), dtype=np.int64)
        new = np.arange(base, base + len(old), dtype=np.int64)
        gmsh.model.mesh.renumberElements(
            oldTags=old.tolist(), newTags=new.tolist())

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def partition(self, n_parts: int) -> PartitionInfo:
        """Partition the mesh into *n_parts* sub-domains (METIS).

        Must be called after ``g.mesh.generation.generate()``.

        Parameters
        ----------
        n_parts : int
            Number of partitions (>= 1).

        Returns
        -------
        PartitionInfo
        """
        if n_parts < 1:
            raise ValueError(f"n_parts must be >= 1, got {n_parts}")
        gmsh.model.mesh.partition(n_parts)
        info = self._gather_partition_info()
        self._mesh._log(f"partition(n_parts={n_parts})")
        return info

    def partition_explicit(
        self,
        n_parts: int,
        elem_tags: list[int],
        parts: list[int],
    ) -> PartitionInfo:
        """Partition with an explicit per-element assignment.

        Parameters
        ----------
        n_parts : int
            Total number of partitions declared.
        elem_tags : list[int]
            Element tags to assign.
        parts : list[int]
            Parallel list of 1-based partition IDs.

        Returns
        -------
        PartitionInfo
        """
        if len(elem_tags) != len(parts):
            raise ValueError(
                f"len(elem_tags)={len(elem_tags)} != "
                f"len(parts)={len(parts)}")
        gmsh.model.mesh.partition(
            n_parts, elementTags=elem_tags, partitions=parts)
        info = self._gather_partition_info()
        self._mesh._log(
            f"partition_explicit(n_parts={n_parts}, "
            f"n_elements={len(elem_tags)})")
        return info

    def unpartition(self) -> "_Partitioning":
        """Remove the partition structure and restore a monolithic mesh."""
        gmsh.model.mesh.unpartition()
        self._mesh._log("unpartition()")
        return self

    # ── internal ─────────────────────────────────────────────────

    def _gather_partition_info(self) -> PartitionInfo:
        """Query Gmsh to build :class:`PartitionInfo` after partitioning."""
        n = gmsh.model.getNumberOfPartitions()
        elems_per: dict[int, int] = {}
        for ent_dim, ent_tag in gmsh.model.getEntities():
            try:
                pparts = gmsh.model.getPartitions(ent_dim, ent_tag)
            except Exception:
                continue
            if len(pparts) == 0:
                continue
            _, etags_list, _ = gmsh.model.mesh.getElements(
                ent_dim, ent_tag)
            n_elems = sum(len(et) for et in etags_list)
            for p in pparts:
                elems_per[int(p)] = elems_per.get(int(p), 0) + n_elems
        return PartitionInfo(n_parts=n, elements_per_partition=elems_per)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def n_partitions(self) -> int:
        """Return the current number of partitions (0 if not partitioned)."""
        return gmsh.model.getNumberOfPartitions()

    def summary(self) -> str:
        """Concise text summary of the partition state."""
        n = self.n_partitions()
        model_name = getattr(
            getattr(self._mesh, '_parent', None), 'name', '?')
        if n == 0:
            return f"Partitioning(model={model_name!r}): not partitioned"
        lines = [
            f"Partitioning(model={model_name!r}): {n} partition(s)"]
        df = self.entity_table()
        if not df.empty:
            partitioned = df[df['partitions'] != '']
            counts = (
                partitioned
                .reset_index()
                .groupby('dim')
                .size()
                .rename(index={
                    0: 'points', 1: 'curves',
                    2: 'surfaces', 3: 'volumes'}))
            for dim_label, count in counts.items():
                lines.append(
                    f"  {dim_label:10s}: {count} partitioned entities")
        return "\n".join(lines)

    def entity_table(self, dim: int = -1) -> "pd.DataFrame":
        """DataFrame of all model entities and their partition membership.

        Parameters
        ----------
        dim : int
            Restrict to a single dimension (``-1`` = all).

        Returns
        -------
        pd.DataFrame
            Columns: ``dim``, ``tag``, ``partitions``,
            ``parent_dim``, ``parent_tag``.
        """
        import pandas as pd

        rows: list[dict] = []
        entities = (
            gmsh.model.getEntities(dim=dim)
            if dim != -1
            else gmsh.model.getEntities())
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
                'dim':        ent_dim,
                'tag':        ent_tag,
                'partitions': ", ".join(str(p) for p in parts),
                'parent_dim': p_dim,
                'parent_tag': p_tag,
            })

        if not rows:
            return pd.DataFrame(
                columns=['dim', 'tag', 'partitions',
                         'parent_dim', 'parent_tag'])
        return pd.DataFrame(rows).set_index(['dim', 'tag'])

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def save(
        self,
        path: Path | str,
        *,
        one_file_per_partition: bool = False,
        create_topology: bool = False,
        create_physicals: bool = True,
    ) -> "_Partitioning":
        """Write the partitioned mesh to file(s).

        Parameters
        ----------
        path : Path or str
            Output file path (format inferred from extension).
        one_file_per_partition : bool
            Write one file per partition alongside the combined file.
        create_topology : bool
            Pass to ``Mesh.PartitionCreateTopology``.
        create_physicals : bool
            Pass to ``Mesh.PartitionCreatePhysicals``.

        Returns
        -------
        self — for chaining
        """
        path = Path(path)
        gmsh.option.setNumber(
            "Mesh.PartitionCreateTopology", int(create_topology))
        gmsh.option.setNumber(
            "Mesh.PartitionCreatePhysicals", int(create_physicals))
        gmsh.option.setNumber(
            "Mesh.PartitionSplitMeshFiles", int(one_file_per_partition))
        gmsh.write(str(path))
        self._mesh._log(
            f"save({path}, "
            f"one_file_per_partition={one_file_per_partition})")
        return self
