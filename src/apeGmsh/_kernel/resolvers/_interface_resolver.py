"""Interface resolver — oriented coincident-pair zeroLength records
(ADR 0093 S4).

The geometry math behind ``g.constraints.interface()``: KD-tree
coincident pairing, per-node **outward** normals on a 2D master
polyline, tributary lengths, backing-element stamping (INV-5) and
mixed-ndf phantom minting (D4). Pure kernel — plain NumPy, no Gmsh, no
OpenSees. The composite gathers the inputs (both node pools, the master
boundary-edge connectivity, the adjacent domain elements) and this
module owns every rule, so each one is testable from bare arrays.

The single sign that matters (INV-1) is fixed here: the record's
``master_node`` is the real continuum node (the future ``iNode``) and
local-x is the master face's **outward** normal, so a separating pair
elongates and an ENT normal law carries zero force. Per-edge outward is
not guessed from a winding convention — it is fixed against the adjacent
domain element's centroid, which is what makes a curved master follow
the wall-to-crown swing (D2) instead of collapsing to one face-average
frame.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numpy import ndarray

from apeGmsh._kernel.records._constraints import (
    InterfaceRecord,
    NodePairRecord,
    NormalLaw,
    TangentialLaw,
)
from apeGmsh._kernel.records._kinds import ConstraintKind

from apeGmsh._kernel.geometry._boundary_chain import EdgeData, edge_frames

from ._constraint_resolver._geom import _SpatialIndex

__all__ = ["resolve_interface_records"]


#: Directions/lengths below this are treated as degenerate rather than
#: normalised into noise.
_ZERO_TOL = 1e-12

#: The ndf a phantom bridge node gets: the LOWER side of a mixed pair,
#: which in the 2D continuum-vs-beam case is always the continuum's 2
#: (ADR 0093 D4 — the mirror, a 3-dof phantom on the continuum side,
#: is the rejected design).
_PHANTOM_NDF = 2

#: The slave ndf values the resolver accepts. ``None`` and ``2`` both
#: mean "the slave matches the 2D continuum" (direct zeroLength, no
#: phantom); ``3`` means a beam slave and mints the phantom bridge.
_SLAVE_NDF_VALUES = (None, 2, 3)


def resolve_interface_records(
    node_tags,
    node_coords,
    *,
    master_nodes: Iterable[int],
    slave_nodes: Iterable[int],
    master_edges,
    domain_elem_tags: Sequence[int],
    domain_elem_nodes: Sequence[Sequence[int]],
    normal_law: NormalLaw,
    tangential_law: TangentialLaw,
    thickness: float,
    tolerance: float = 1e-6,
    slave_ndf: int | None = None,
    ndm: int = 2,
    phantom_tag_start: int | None = None,
    name: str | None = None,
) -> tuple[list[InterfaceRecord], int]:
    """Resolve one ``interface`` def to its per-pair records.

    Parameters
    ----------
    node_tags, node_coords
        The whole model's node pool — ``(n,)`` tags and ``(n, 3)``
        coordinates. Every master / slave / domain-element node must be
        in it.
    master_nodes, slave_nodes
        The two coincident node sets. They must be disjoint: a node in
        both would pair with itself.
    master_edges
        ``(n_edges, 2)`` node tags — the master boundary polyline's
        2-node segments. Both endpoints of every edge must be master
        nodes; a duplicated edge is refused (it would double that
        stretch's tributary share).
    domain_elem_tags, domain_elem_nodes
        The model's **highest-dimension** (2D continuum) elements —
        tags and per-element node tags (ragged is fine). Boundary
        curve elements must NOT be in here: INV-5's owner pick is exact
        only over top-dimension elements, which ``_fem_extract`` never
        replicates across partitions.
    normal_law, tangential_law
        The declarative per-area laws (D1), stored verbatim on every
        record; translated to materials only at emit (S5).
    thickness
        Out-of-plane thickness, ``> 0`` — ``A_trib = ell_trib *
        thickness`` (D3).
    tolerance
        Coincidence radius for the pairing.
    slave_ndf
        ``None`` / ``2`` — the slave matches the 2D continuum, so the
        zeroLength connects the two real nodes. ``3`` — a beam slave,
        so each pair gets a phantom bridge (D4). See the
        ``InterfaceDef`` docstring for why this is explicit and never
        inferred.
    ndm
        Model dimension. Only ``2`` is implemented (D2).
    phantom_tag_start
        First phantom node tag to mint. Defaults to ``max(node_tags) +
        1`` — **but** the MP lane's ``ConstraintResolver`` mints its own
        phantoms from that same base, so a model carrying both must pass
        that resolver's high-water mark here or the two ranges collide.
    name
        Friendly name carried onto every record.

    Returns
    -------
    (records, next_phantom_tag)
        The per-pair records, ordered by ascending **slave** node tag,
        and the next free phantom tag (unchanged from
        ``phantom_tag_start`` when no phantom was minted) so a caller
        can keep threading one high-water mark.
    """
    if int(ndm) != 2:
        raise NotImplementedError(
            f"interface: only 2D line masters are implemented (ndm=2), "
            f"got ndm={ndm}. 3D surface masters are deferred — "
            f"ADR 0093 D2.")
    if slave_ndf not in _SLAVE_NDF_VALUES:
        raise ValueError(
            f"interface: slave_ndf must be one of {_SLAVE_NDF_VALUES} "
            f"(None/2 = the slave matches the 2D continuum, direct "
            f"zeroLength; 3 = beam slave, phantom bridge per ADR 0093 "
            f"D4), got {slave_ndf!r}.")
    if not (float(thickness) > 0.0):
        raise ValueError(
            f"interface: thickness must be > 0, got {thickness!r}.")

    label = f" {name!r}" if name else ""
    tags = np.asarray(node_tags, dtype=int).ravel()
    coords = np.asarray(node_coords, dtype=float).reshape(tags.size, 3)
    xyz: dict[int, ndarray] = {
        int(t): coords[i] for i, t in enumerate(tags)
    }

    m_set = {int(t) for t in master_nodes}
    s_set = {int(t) for t in slave_nodes}
    if not m_set:
        raise ValueError(f"interface{label}: the master node set is empty.")
    if not s_set:
        raise ValueError(f"interface{label}: the slave node set is empty.")
    shared = sorted(m_set & s_set)
    if shared:
        raise ValueError(
            f"interface{label}: the master and slave node sets share "
            f"node(s) {shared[:20]} ({len(shared)} total). A shared node "
            f"would pair with itself and emit a zeroLength of a node "
            f"against itself — the two sides must be independent meshes "
            f"(duplicate the interface nodes, e.g. via a crack / a "
            f"separate part).")
    absent = sorted((m_set | s_set) - xyz.keys())
    if absent:
        raise ValueError(
            f"interface{label}: node(s) {absent[:20]} are not in the "
            f"model node pool.")

    pairs = _pair_coincident(m_set, s_set, xyz, tolerance, label)

    edge_data = edge_frames(
        master_edges, m_set, xyz,
        domain_elem_tags, domain_elem_nodes, label,
        verb="interface",
    )
    normals = _node_normals(edge_data, label)
    ell = _tributary_lengths(edge_data)
    _assert_tributary_closure(
        pairs, ell, edge_data, float(thickness), label)

    records: list[InterfaceRecord] = []
    next_tag = (
        int(phantom_tag_start) if phantom_tag_start is not None
        else int(tags.max()) + 1
    )
    for master, slave in pairs:
        n = normals[master]
        # ADR 0093 D2 / INV-1: local-x is the master face's OUTWARD
        # normal; local-y is z-hat x x-hat, which in 2D is the unique
        # right-handed in-plane completion. The tangential yield cap is
        # symmetric (|tau| <= tau_b), so the tangent's sign is a
        # determinism choice, not mechanics — but it must be a *fixed*
        # choice so two runs of the same model produce the same deck.
        orient = (
            float(n[0]), float(n[1]), 0.0,
            float(-n[1]), float(n[0]), 0.0,
        )
        backing = _backing_element(
            master, n, xyz[master], edge_data, label)

        phantom_node = None
        phantom_coords = None
        phantom_ndf = None
        equal_dofs: list[NodePairRecord] = []
        if slave_ndf == 3:
            # D4 — the beam slave cannot take the zeroLength directly
            # (ZeroLength::setDomain refuses dofNd1 != dofNd2), so the
            # pair gets a 2-dof phantom standing on the SLAVE side and
            # rigidly tied to the beam node's translations. The
            # zeroLength then runs master continuum -> phantom (S5's
            # job); ``slave_node`` below still carries the real beam
            # node for provenance and for this nested equalDOF.
            phantom_node = next_tag
            next_tag += 1
            phantom_coords = np.asarray(xyz[slave], dtype=float).copy()
            phantom_ndf = _PHANTOM_NDF
            equal_dofs = [NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                name=name,
                master_node=slave,      # retained: the real beam node
                slave_node=phantom_node,  # constrained: the phantom
                dofs=[1, 2],
            )]

        records.append(InterfaceRecord(
            kind=ConstraintKind.INTERFACE,
            name=name,
            master_node=master,
            slave_node=slave,
            backing_element=backing,
            orient=orient,
            a_trib=float(ell[master]) * float(thickness),
            normal_law=normal_law,
            tangential_law=tangential_law,
            phantom_node=phantom_node,
            phantom_coords=phantom_coords,
            phantom_ndf=phantom_ndf,
            equal_dof_records=equal_dofs,
        ))

    return records, next_tag


# ── pairing ─────────────────────────────────────────────────────────

def _pair_coincident(
    m_set: set[int],
    s_set: set[int],
    xyz: dict[int, ndarray],
    tolerance: float,
    label: str,
) -> list[tuple[int, int]]:
    """Coincident (master, slave) pairs, ordered by ascending slave tag.

    Sibling of ``ConstraintResolver._match_node_pairs`` with one added
    refusal: a slave with no master inside ``tolerance`` is an error
    here, not a silent skip. An interface that quietly springs only
    part of a face is the silent-failure class ADR 0093 exists to kill.
    """
    m_list = sorted(m_set)
    index = _SpatialIndex(np.array([xyz[t] for t in m_list], dtype=float))

    pairs: list[tuple[int, int]] = []
    claimed: dict[int, list[int]] = {}
    unmatched: list[int] = []
    for st in sorted(s_set):
        dist, idx = index.query(xyz[st])
        if float(dist) > tolerance:
            unmatched.append(st)
            continue
        mt = m_list[int(idx)]
        pairs.append((mt, st))
        claimed.setdefault(mt, []).append(st)

    if unmatched:
        raise ValueError(
            f"interface{label}: slave node(s) {unmatched[:20]} "
            f"({len(unmatched)} of {len(s_set)}) have no master node "
            f"within tolerance={tolerance}. The two sides are not "
            f"node-for-node coincident — remesh them to match, or "
            f"scope the sides with master_entities= / slave_entities=. "
            f"g.constraints.interface() has no projection lane (use "
            f"tie / contact for a non-matching mesh).")

    multi = {m: sorted(s) for m, s in claimed.items() if len(s) > 1}
    if multi:
        raise ValueError(
            f"interface{label}: co-located pairing is ambiguous — master "
            f"node(s) {sorted(multi)} each matched >1 slave within "
            f"tolerance={tolerance} ({multi}). Tighten the tolerance or "
            f"deduplicate the coincident slave nodes.")
    return pairs




def _node_normals(data: EdgeData, label: str) -> dict[int, ndarray]:
    """Per-master-node outward unit normal (D2).

    The normalized average of the node's adjacent edge normals — which
    on an open polyline's endpoint is simply that one edge's normal, and
    on a curved master swings pair-by-pair instead of collapsing to a
    single face-average frame (INV-1's curved case). Fail loud when the
    average opposes a contributor: a node where the boundary doubles
    back has no single outward direction.
    """
    out: dict[int, ndarray] = {}
    for node, e_list in data.node_edges.items():
        if not e_list:
            continue
        stack = [data.normals[e] for e in e_list]
        avg = np.sum(stack, axis=0)
        norm = float(np.linalg.norm(avg))
        if norm <= _ZERO_TOL:
            raise ValueError(
                f"interface{label}: master node {node}'s adjacent edge "
                f"normals cancel out (their average is the zero vector) "
                f"— the boundary doubles back on itself there, so the "
                f"node has no single outward direction (ADR 0093 D2).")
        avg = avg / norm
        bad = [
            (int(e), float(np.dot(avg, data.normals[e])))
            for e in e_list
            if float(np.dot(avg, data.normals[e])) <= 0.0
        ]
        if bad:
            raise ValueError(
                f"interface{label}: master node {node}'s averaged "
                f"outward normal {tuple(round(float(v), 6) for v in avg)} "
                f"opposes the normal of its own adjacent edge(s) "
                f"{[e for e, _ in bad]} (dots {[d for _, d in bad]}) — a "
                f"reentrant / doubled-back corner has no single outward "
                f"direction (ADR 0093 D2). Split the interface at that "
                f"node into separate interface() calls.")
        out[node] = avg
    return out


# ── tributary (D3 / INV-3) ──────────────────────────────────────────

def _tributary_lengths(data: EdgeData) -> dict[int, float]:
    """``ell_trib`` per master node — the ``0.5 * edge_length``
    accumulation of the load resolver's ``resolve_line_tributary``,
    which gives an interior node the two half-shares and a polyline
    endpoint exactly one."""
    ell: dict[int, float] = {t: 0.0 for t in data.node_edges}
    for e, (a, b) in enumerate(data.edges):
        half = 0.5 * data.lengths[e]
        ell[int(a)] += half
        ell[int(b)] += half
    return ell


def _assert_tributary_closure(
    pairs: list[tuple[int, int]],
    ell: dict[int, float],
    data: EdgeData,
    thickness: float,
    label: str,
) -> None:
    """INV-3 — the pairs' tributary areas tile the whole master face.

    Two ways this fails, both loud: a paired node with a zero share (it
    lies on no boundary edge, so its spring would carry no area), and a
    master node the slave side never reached (the sum then falls short
    of the face, i.e. part of the interface is silently unsprung).
    """
    paired = {m for m, _ in pairs}
    zero = sorted(m for m in paired if ell.get(m, 0.0) <= 0.0)
    if zero:
        raise ValueError(
            f"interface{label}: paired master node(s) {zero[:20]} lie on "
            f"no master boundary edge, so their tributary length is zero "
            f"— their spring would carry no area (ADR 0093 INV-3).")

    on_boundary = {t for t, e_list in data.node_edges.items() if e_list}
    unpaired = sorted(on_boundary - paired)
    if unpaired:
        raise ValueError(
            f"interface{label}: master boundary node(s) {unpaired[:20]} "
            f"({len(unpaired)} of {len(on_boundary)}) were not paired to "
            f"any slave node, so the pairs' tributary areas do not tile "
            f"the master face (ADR 0093 INV-3). Scope the master with "
            f"master_entities= to the stretch the slave actually covers.")

    total = sum(ell[m] for m in paired) * thickness
    expected = data.total_length * thickness
    # A sum of n floats against an independently accumulated length
    # cannot be bit-exact — INV-3 budgets O(n * eps) relative.
    atol = 64.0 * np.finfo(float).eps * max(1, len(paired)) * abs(expected)
    if abs(total - expected) > atol:
        raise ValueError(
            f"interface{label}: tributary closure failed (ADR 0093 "
            f"INV-3) — sum(A_trib)={total!r} vs master length x "
            f"thickness={expected!r} (atol={atol!r}).")


# ── backing element (INV-5 / settled Q3) ────────────────────────────

def _backing_element(
    master: int,
    normal: ndarray,
    node_xyz: ndarray,
    data: EdgeData,
    label: str,
) -> int:
    """The domain element the pair's outward normal points AWAY from.

    Among the 2D domain elements incident on ``master`` (never a
    boundary curve element — those are the entities Gmsh replicates
    across a partition cut), the one whose centroid sits furthest
    *behind* the outward normal: the most-negative ``dot(centroid -
    node, n)``. Exact ties break to the lowest element tag, so the pick
    is deterministic and 1-vs-N byte identity survives (INV-5).
    """
    cands = data.adj.get(master) or []
    if not cands:
        raise ValueError(
            f"interface{label}: master node {master} has no incident 2D "
            f"domain element to back it (ADR 0093 INV-5).")
    best = min(
        cands,
        key=lambda i: (
            float(np.dot((data.centroids[i] - node_xyz)[:2], normal)),
            data.elem_tags[i],
        ),
    )
    return int(data.elem_tags[best])
