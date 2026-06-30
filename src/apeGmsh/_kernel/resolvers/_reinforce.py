"""Reinforcement resolver — rebar line PG → ``LadrunoEmbeddedRebar`` ties.

The apeGmsh-owned crux of ``g.reinforce`` (ADR 20): for each node of a
pre-meshed rebar line PG, invert it into the non-matching solid host mesh
(:func:`apeGmsh._kernel.geometry._inverse_map.locate_point`), derive the
bar axis ``d̂`` and tributary length ``L_trib`` from the rebar segments,
and emit one :class:`~apeGmsh._kernel.records._constraints.ReinforceTieRecord`
per rebar node. The records are solver-agnostic; the bridge build step
turns each into ``element LadrunoEmbeddedRebar`` via the R0
``embedded_rebar_args`` builder.

Pure NumPy — no Gmsh, no OpenSees imports.
"""
from __future__ import annotations

from math import pi

import numpy as np
from numpy import ndarray

from apeGmsh._kernel.geometry._inverse_map import (
    inverse_map_single,
    locate_point,
)
from apeGmsh._kernel.records._constraints import ReinforceTieRecord


__all__ = ["resolve_reinforce", "tributary_lengths", "node_directions"]


def _point_b_weights(
    point_a: ndarray,
    dhat: ndarray,
    host_coords: ndarray,
    host_kind: str,
) -> ndarray:
    """Shape weights ``NshapeB`` at a second point B along the bar, inside the
    rebar node's host (the ``-corot -shapeB`` path, ADR 20 §10.5).

    Point B is a SHORT step ``s = 0.05·host_radius`` along the bar axis from the
    embed point A, inverse-mapped into the SAME host element. The fork forms the
    co-rotated axis as ``normalize(Σ NshapeB·x − Σ Nshape·x)`` from current host
    node positions, so only the *direction* A→B matters — its magnitude cancels
    under the normalisation. A short in-host step keeps the inverse map well
    inside the reference element (so the weights are clean) while still
    capturing the host's local deformation gradient along the bar. The bar
    direction sign is arbitrary (the axis is a line); if the ``+d̂`` step does
    not converge (A near a host face, bar pointing outward) the ``−d̂`` step is
    tried before failing loud.
    """
    centroid = host_coords.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(host_coords - centroid, axis=1)))
    if radius < 1e-30:
        raise ValueError(
            "resolve_reinforce: degenerate host element (zero radius) — "
            "cannot place the corot point B"
        )
    s = 0.05 * radius
    for sign in (1.0, -1.0):
        point_b = point_a + sign * s * dhat
        _, weights, _, converged = inverse_map_single(
            point_b, host_coords, host_kind)
        if converged:
            return weights
    raise ValueError(
        "resolve_reinforce: could not inverse-map the corot point B into the "
        "host element (Newton failed from either bar direction) — unexpected "
        "for a short in-host step; check the host geometry"
    )


def _segment_table(
    bar_segments: list[tuple[int, int]],
    coords: dict[int, ndarray],
) -> dict[int, list[tuple[int, float, ndarray]]]:
    """node -> list of (other_node, segment_length, unit_dir_away_from_node)."""
    table: dict[int, list[tuple[int, float, ndarray]]] = {}
    for a, b in bar_segments:
        va, vb = coords[a], coords[b]
        d = vb - va
        L = float(np.linalg.norm(d))
        if L < 1e-30:
            continue
        u = d / L
        table.setdefault(a, []).append((b, L, u))
        table.setdefault(b, []).append((a, L, -u))
    return table


def tributary_lengths(
    bar_node_ids: list[int],
    bar_segments: list[tuple[int, int]],
    coords: dict[int, ndarray],
) -> dict[int, float]:
    """``L_trib`` per rebar node = ½·Σ(adjacent segment lengths).

    An endpoint (one segment) gets ½ its single segment; an interior node
    (two segments) gets ½(L₁+L₂).
    """
    table = _segment_table(bar_segments, coords)
    return {
        nid: 0.5 * sum(L for _, L, _ in table.get(nid, []))
        for nid in bar_node_ids
    }


def node_directions(
    bar_node_ids: list[int],
    bar_segments: list[tuple[int, int]],
    coords: dict[int, ndarray],
) -> dict[int, ndarray]:
    """Unit bar axis ``d̂`` per rebar node.

    Interior nodes use the **through** direction (the secant between the two
    neighbours); endpoints use their single segment's direction. The sign is
    arbitrary for the coupling (the axial split is symmetric in ``±d̂``).
    """
    table = _segment_table(bar_segments, coords)
    out: dict[int, ndarray] = {}
    for nid in bar_node_ids:
        adj = table.get(nid, [])
        if len(adj) >= 2:
            # through-direction: neighbour0 -> neighbour1
            n0 = coords[adj[0][0]]
            n1 = coords[adj[1][0]]
            d = n1 - n0
        elif len(adj) == 1:
            d = adj[0][2]  # unit dir already points from node toward neighbour
        else:
            raise ValueError(
                f"resolve_reinforce: rebar node {nid} has no adjacent bar "
                f"segment — cannot define a bar axis"
            )
        L = float(np.linalg.norm(d))
        if L < 1e-30:
            raise ValueError(
                f"resolve_reinforce: degenerate bar axis at node {nid}"
            )
        out[nid] = d / L
    return out


def resolve_reinforce(
    *,
    bar_node_ids: list[int],
    bar_node_coords: ndarray,
    bar_segments: list[tuple[int, int]],
    host_node_ids: list[list[int]],
    host_node_coords: list[ndarray],
    host_kinds: list[str],
    bond: str | None = None,
    perfect: float | None = None,
    diameter: float | None = None,
    kt: float | None = None,
    kt_alpha: float | None = None,
    enforce: str = "penalty",
    bipenalty: bool = False,
    dtcr: float | None = None,
    corot: bool = False,
    tolerance: float = 1e-6,
    snap: bool = False,
    name: str | None = None,
) -> list[ReinforceTieRecord]:
    """Resolve a rebar line PG into ``LadrunoEmbeddedRebar`` tie records.

    Parameters
    ----------
    bar_node_ids, bar_node_coords
        The rebar mesh nodes (parallel: ids + ``(n, dim)`` coords).
    bar_segments
        The rebar line-element connectivity ``[(i, j), ...]`` (node tags),
        used for the bar axis + tributary length.
    host_node_ids, host_node_coords, host_kinds
        Per host element: its node tags, node coords ``(n_nodes, dim)``,
        and kind (``"hex8"`` / ``"tet4"`` / ``"quad4"`` / ``"tri3"``). For a
        straight-sided higher-order host pass its **corner** subset + the
        corner kind (the weights then couple to the corner nodes).
    bond, perfect, diameter, kt, kt_alpha, enforce
        Tie parameters (pass-through to the emit). ``diameter`` is required
        for ``bond`` (``bondScale = π·d·L_trib``).
    corot
        ``True`` ⇒ also compute each node's point-B shape weights
        (``shape_b`` / ``-shapeB``) for the co-rotated bar axis (ADR 20
        §10.5); ``False`` (default) ⇒ ``shape_b`` is ``None`` (frozen ``-dir``).
    tolerance, snap
        Inverse-map out-of-bounds policy (ADR 20 D3): reject-by-default,
        opt-in snap.

    Returns one :class:`ReinforceTieRecord` per rebar node.
    """
    if (bond is None) == (perfect is None):
        raise ValueError(
            "resolve_reinforce: supply exactly one axial law (bond or perfect)"
        )
    if bond is not None and diameter is None:
        raise ValueError(
            "resolve_reinforce: a bond law needs `diameter` for bondScale"
        )

    coords = {
        int(nid): np.asarray(bar_node_coords[i], dtype=float)
        for i, nid in enumerate(bar_node_ids)
    }
    Ltrib = tributary_lengths(bar_node_ids, bar_segments, coords)
    dirs = node_directions(bar_node_ids, bar_segments, coords)

    records: list[ReinforceTieRecord] = []
    for nid in bar_node_ids:
        res = locate_point(
            coords[nid], host_node_coords, host_kinds,
            tol=tolerance, snap=snap, label=name or "",
        )
        host = host_node_ids[res.host_index]
        if len(host) != len(res.weights):
            raise ValueError(
                f"resolve_reinforce: host element {res.host_index} has "
                f"{len(host)} nodes but the {host_kinds[res.host_index]} map "
                f"produced {len(res.weights)} weights"
            )
        if bond is not None:
            assert diameter is not None  # guaranteed by the validation above
            bond_scale: float | None = pi * float(diameter) * Ltrib[nid]
        else:
            bond_scale = None
        # Co-rotated bar axis (ADR 20 §10.5): compute the point-B weights in
        # THIS rebar node's host so the fork can co-rotate d̂ each step.
        shape_b = (
            _point_b_weights(
                coords[nid], dirs[nid],
                np.asarray(host_node_coords[res.host_index], dtype=float),
                host_kinds[res.host_index],
            ).copy()
            if corot else None
        )
        records.append(ReinforceTieRecord(
            kind="reinforce",
            name=name,
            rebar_node=int(nid),
            host_nodes=[int(h) for h in host],
            weights=res.weights.copy(),
            direction=dirs[nid].copy(),
            corot=corot,
            shape_b=shape_b,
            bond_scale=bond_scale,
            bond=bond,
            perfect=perfect,
            kt=kt,
            kt_alpha=kt_alpha,
            enforce=enforce,
            bipenalty=bipenalty,
            dtcr=dtcr,
            excess=res.excess,
            in_bounds=res.in_bounds,
        ))
    return records
