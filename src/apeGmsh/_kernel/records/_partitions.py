"""apeGmsh._kernel.records._partitions — Frozen ``PartitionRecord``.

A :class:`PartitionRecord` is the immutable, ergonomic view of one
mesh partition that lives on the broker as part of
:class:`apeGmsh._kernel.record_sets.PartitionSet` (exposed on
``fem.partitions``).

The record is a lightweight wrapper around the ``node_ids`` /
``element_ids`` arrays already stored on
:attr:`NodeComposite._partitions` and
:attr:`ElementComposite._partitions` — the private back-stores that
power ``fem.{nodes,elements}.select(partition=N)``.  Those back-stores
are untouched by this layer; the record is constructed once at
:class:`~apeGmsh.mesh.FEMData.FEMData` init from the same dicts the
extractor produced, so there is no second source of truth.

The optional ``weight_sum`` field is reserved for P1's weighted
partitioning extension and stays ``None`` until that PR wires it
through ``extract_partitions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class PartitionRecord:
    """Immutable view of one mesh partition.

    Parameters
    ----------
    id : int
        Partition tag as assigned by Gmsh / the partitioner.
    node_ids : ndarray
        ``int64`` array of node IDs in this partition.  Should be
        sorted and de-duplicated upstream — the dataclass does not
        re-sort or copy.
    element_ids : ndarray
        ``int64`` array of element IDs in this partition.  Same
        invariants as ``node_ids``.
    weight_sum : float or None
        Sum of element weights used by the weighted partitioner.
        ``None`` when the partition was built without weights (the
        current default in this PR).  Future-proofing for P1.
    """

    id: int
    node_ids: np.ndarray
    element_ids: np.ndarray
    weight_sum: float | None = None

    # ADR 0038 §"Merge semantics" — module's own PartitionSet is
    # DISCARD: the host re-partitions per the 3-layer rank model
    # (line 168).  ``None`` sentinel makes the compose rewriter skip
    # this record kind without falling through to the default scan.
    tag_rewrite_spec: ClassVar[None] = None

    @property
    def n_nodes(self) -> int:
        """Number of nodes in this partition."""
        return int(self.node_ids.size)

    @property
    def n_elements(self) -> int:
        """Number of elements in this partition."""
        return int(self.element_ids.size)

    def __repr__(self) -> str:
        ws = (
            f", weight_sum={self.weight_sum:.3g}"
            if self.weight_sum is not None
            else ""
        )
        return (
            f"PartitionRecord(id={self.id}, "
            f"n_nodes={self.n_nodes}, "
            f"n_elements={self.n_elements}{ws})"
        )
