"""``FEMData.repartition`` — partition a snapshot as one graph (ADR 0120 D2).

The mesh partitioner (``g.mesh.partitioning``) runs inside a live Gmsh
session. A snapshot that no session holds — a composed model, the merge
of a structure and a soil box, a ``model.h5`` read back — has either no
partition or one rank per composed module. ``repartition`` splits every
element of the snapshot into ``n_parts`` ranks at once, whatever module
it came from, by recursive coordinate bisection (RCB) of the element
centroids:

* the element set is cut across its longest extent, at the weight that
  gives each side its share of the parts, and each side is cut again
  until every part is one rank;
* an element's weight is its node count unless ``weights`` says
  otherwise, so a hex8 counts twice a quad4; a weight of zero (a group
  no element spec emits, such as geometry points) still places the
  element by position but costs nothing;
* a rank holds the nodes of its elements; a node on a cut is in every
  rank that holds one of its elements (the interface nodes OpenSeesMP
  merges). A node no element references is in no partition: the
  bridge routes it (ADR 0120 D1).

RCB needs no graph library and keeps what is close together on one rank:
a structure's node embedded in a soil element is usually on that
element's rank, so the coupling needs no ghost.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from .FEMData import FEMData

__all__ = ["repartition"]

#: ``weights=``: a weight per element type name (``"hex8"``), or a callable
#: ``f(type_name, ids, centroids) -> (n,) weights`` for one element group.
PartitionWeights = Union[
    Mapping[str, float],
    Callable[[str, np.ndarray, np.ndarray], "np.ndarray"],
]


def _bisect(
    xyz: np.ndarray, w: np.ndarray, eid: np.ndarray, idx: np.ndarray,
    n_parts: int, first: int, out: np.ndarray,
) -> None:
    """Assign ``idx`` to parts ``first .. first + n_parts - 1`` by RCB."""
    if n_parts == 1:
        out[idx] = first
        return
    n_left = n_parts // 2
    n_right = n_parts - n_left
    pts = xyz[idx]
    axis = int(np.argmax(np.ptp(pts, axis=0)))
    # Sort along the axis, element id breaking ties: the same snapshot
    # always gives the same partition.
    order = idx[np.lexsort((eid[idx], pts[:, axis]))]
    cum = np.cumsum(w[order])
    target = cum[-1] * n_left / n_parts
    k = int(np.searchsorted(cum, target, side="left"))
    if k < cum.size and (cum[k] - target) <= (target - (cum[k - 1] if k else 0.0)):
        k += 1
    k = min(max(k, n_left), order.size - n_right)
    _bisect(xyz, w, eid, order[:k], n_left, first, out)
    _bisect(xyz, w, eid, order[k:], n_right, first + n_left, out)


def repartition(
    fem: "FEMData", n_parts: int, *,
    weights: "PartitionWeights | None" = None,
) -> "FEMData":
    """See :meth:`FEMData.repartition`."""
    from apeGmsh._kernel.record_sets import PartitionSet
    from apeGmsh._kernel.records._partitions import PartitionRecord

    if isinstance(n_parts, bool) or not isinstance(n_parts, (int, np.integer)):
        raise TypeError(f"repartition: n_parts must be an int, got {n_parts!r}.")
    n_parts = int(n_parts)
    if n_parts < 1:
        raise ValueError(f"repartition: n_parts must be >= 1, got {n_parts}.")

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    by_id = np.argsort(node_ids, kind="stable")
    sorted_ids = node_ids[by_id]
    coords = np.asarray(fem.nodes.coords, dtype=float)
    groups = [g for g in fem.elements if len(g)]
    if not groups:
        raise ValueError("repartition: the snapshot has no elements to partition.")

    eids, cents, ws, conns = [], [], [], []
    for g in groups:
        conn = np.asarray(g.connectivity, dtype=np.int64)
        pos = np.searchsorted(sorted_ids, conn)
        if np.any(pos >= sorted_ids.size) or np.any(
                sorted_ids[np.minimum(pos, sorted_ids.size - 1)] != conn):
            raise ValueError(
                f"repartition: a {g.type_name!r} element references a node "
                f"the snapshot does not hold.")
        rows = by_id[pos]
        cent = coords[rows].mean(axis=1)
        ids = np.asarray(g.ids, dtype=np.int64)
        if weights is None:
            w = np.full(ids.size, float(conn.shape[1]))
        elif callable(weights):
            w = np.asarray(weights(g.type_name, ids, cent), dtype=float)
            if w.shape != ids.shape:
                raise ValueError(
                    f"repartition: weights({g.type_name!r}, ...) returned shape "
                    f"{w.shape}, expected {ids.shape}.")
        else:
            if g.type_name not in weights:
                raise ValueError(
                    f"repartition: weights has no entry for element type "
                    f"{g.type_name!r} (it names {sorted(weights)}).")
            w = np.full(ids.size, float(weights[g.type_name]))
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError(
                f"repartition: element weights must be finite and >= 0 "
                f"({g.type_name!r}).")
        eids.append(ids)
        cents.append(cent)
        ws.append(w)
        conns.append(conn)

    eid = np.concatenate(eids)
    if np.unique(eid).size != eid.size:
        raise ValueError("repartition: element ids repeat across element groups.")
    if eid.size < n_parts:
        raise ValueError(
            f"repartition: {eid.size} elements cannot fill {n_parts} ranks.")
    w_all = np.concatenate(ws)
    if not w_all.sum() > 0.0:
        raise ValueError("repartition: every element weight is zero.")
    part = np.empty(eid.size, dtype=np.int64)
    _bisect(np.concatenate(cents), w_all, eid,
            np.arange(eid.size), n_parts, 1, part)

    node_parts: dict[int, dict] = {}
    elem_parts: dict[int, dict] = {}
    records: dict[int, PartitionRecord] = {}
    offsets = np.cumsum([0] + [c.shape[0] for c in conns])
    for p in range(1, n_parts + 1):
        mask = part == p
        e_ids = np.sort(eid[mask])
        n_ids = np.unique(np.concatenate([
            conns[k][mask[offsets[k]:offsets[k + 1]]].ravel()
            for k in range(len(conns))]))
        node_parts[p] = {"node_ids": n_ids, "element_ids": e_ids}
        elem_parts[p] = {"node_ids": n_ids, "element_ids": e_ids}
        records[p] = PartitionRecord(id=p, node_ids=n_ids, element_ids=e_ids)
    if n_parts == 1:
        node_parts, elem_parts, records = {}, {}, {}

    new = fem._replaced(
        nodes=fem._replaced_nodes(_partitions=node_parts),
        elements=fem._replaced_elements(_partitions=elem_parts),
    )
    new.partitions = PartitionSet(records)
    return new
