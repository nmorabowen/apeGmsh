"""Embedment resolver — node set → ``LadrunoEmbeddedNode`` ties.

The apeGmsh-owned crux of ``g.embed``: for each node of a constrained node
set, invert it into the non-matching solid host mesh
(:func:`apeGmsh._kernel.geometry._inverse_map.locate_point`) and emit one
:class:`~apeGmsh._kernel.records._constraints.EmbedTieRecord` per node. The
isotropic sibling of :mod:`apeGmsh._kernel.resolvers._reinforce` — no bar
axis, bond law, or tributary length. Records are solver-agnostic; the bridge
build step turns each into ``element LadrunoEmbeddedNode`` via the
``embedded_node_args`` builder.

Pure NumPy — no Gmsh, no OpenSees imports.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

from apeGmsh._kernel.geometry._inverse_map import locate_point
from apeGmsh._kernel.records._constraints import EmbedTieRecord


__all__ = ["resolve_embed"]


def resolve_embed(
    *,
    node_ids: list[int],
    node_coords: ndarray,
    host_node_ids: list[list[int]],
    host_node_coords: list[ndarray],
    host_kinds: list[str],
    k: float | None = None,
    k_alpha: float | None = None,
    enforce: str = "penalty",
    explicit: bool = False,
    dtcr: float | None = None,
    staged: bool = True,
    tolerance: float = 1e-6,
    snap: bool = False,
    name: str | None = None,
) -> list[EmbedTieRecord]:
    """Resolve a constrained node set into ``LadrunoEmbeddedNode`` records.

    Parameters
    ----------
    node_ids, node_coords
        The constrained mesh nodes (parallel: ids + ``(n, dim)`` coords).
    host_node_ids, host_node_coords, host_kinds
        Per host element: its node tags, node coords ``(n_nodes, dim)``, and
        kind (``"hex8"`` / ``"tet4"`` / ``"quad4"`` / ``"tri3"``). For a
        straight-sided higher-order host pass its **corner** subset + the
        corner kind (the weights then couple to the corner nodes).
    k, k_alpha, enforce, explicit, dtcr, staged
        Tie parameters (pass-through to the emit).
    tolerance, snap
        Inverse-map out-of-bounds policy: reject-by-default, opt-in snap.

    Returns one :class:`EmbedTieRecord` per constrained node.
    """
    coords = {
        int(nid): np.asarray(node_coords[i], dtype=float)
        for i, nid in enumerate(node_ids)
    }

    records: list[EmbedTieRecord] = []
    for nid in node_ids:
        res = locate_point(
            coords[nid], host_node_coords, host_kinds,
            tol=tolerance, snap=snap, label=name or "",
        )
        host = host_node_ids[res.host_index]
        if len(host) != len(res.weights):
            raise ValueError(
                f"resolve_embed: host element {res.host_index} has "
                f"{len(host)} nodes but the {host_kinds[res.host_index]} map "
                f"produced {len(res.weights)} weights"
            )
        records.append(EmbedTieRecord(
            kind="embed",
            name=name,
            node=int(nid),
            host_nodes=[int(h) for h in host],
            weights=res.weights.copy(),
            k=k,
            k_alpha=k_alpha,
            enforce=enforce,
            bipenalty=explicit,
            dtcr=dtcr,
            staged=staged,
            excess=res.excess,
            in_bounds=res.in_bounds,
        ))
    return records
