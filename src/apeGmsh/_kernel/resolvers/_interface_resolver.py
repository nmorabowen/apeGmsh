"""Interface resolver — oriented coincident-pair zeroLength records
(ADR 0093 S4).

The geometry math behind ``g.constraints.interface()``: KD-tree
coincident pairing, per-node **outward** normals on the master,
tributary shares, backing-element stamping (INV-5) and mixed-ndf
phantom minting (D4). Pure kernel — plain NumPy, no Gmsh, no OpenSees.
The composite gathers the inputs (both node pools, the master's
boundary connectivity, the adjacent domain elements) and this module
owns every rule, so each one is testable from bare arrays.

The single sign that matters (INV-1) is fixed here: the record's
``master_node`` is the real continuum node (the future ``iNode``) and
local-x is the master face's **outward** normal, so a separating pair
elongates and an ENT normal law carries zero force. Outward is never
guessed from a winding convention — it is fixed against the adjacent
domain element's centroid, which is what makes a curved master follow
the wall-to-crown swing (D2) instead of collapsing to one face-average
frame.

Two master dimensions, one set of rules. A **dim-1 line master in a 2D
model** walks boundary edges (:func:`edge_frames`), carries an
out-of-plane ``thickness``, orients on six floats and mints the D4
phantom for a beam slave. A **dim-2 surface master in a 3D model**
(TIMs A10 S2) walks boundary facets (:func:`surface_frames`), takes its
``A_trib`` straight off the facet-area accumulation with no
``thickness`` at all, orients on nine — ``(n, t1, t2)``, because a 3D
Coulomb law acts on two tangents — and mints no phantom: fork #808 /
ADR 96 lets the ``zeroLength`` take the mixed pair directly (see
:func:`accepts_3d_ndf_pair`). Emission of a 3D record is TIMs A10
S3; nothing here writes a deck line.
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
from apeGmsh._kernel.geometry._surface_frames import (
    SurfaceFrames, surface_frames,
)

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

#: The least ndf either end of a 3D ``zeroLength`` may carry. Fork #808 /
#: ADR 96 (adoption note
#: ``internal_docs/contact_3d_passenger_dof_adoption.md``) states the
#: rule as "any pair with both ndf >= 3": the element acts on DOFs 1-3
#: and every DOF past the third — the u-p pore pressure, a shell's
#: rotations — rides as an untouched passenger. That is what retires the
#: phantom in 3D: D4 exists because the engine refused
#: ``dofNd1 != dofNd2``, and in 3D it no longer does. The minimum build is
#: ``opensees._target.TIMS_FORK_BATCH_MIN_BUILD`` (``a240b9183``); below
#: it the fork FATALs on the mixed rows. A build's ancestry is not
#: derivable from a hash, so that constant is **documented, not
#: enforced** — the same standing as everywhere else it is cited.
_MIN_3D_NDF = 3

#: The ``(ndf_i, ndf_j)`` pairs ADR 96's adoption note spells out by name
#: — ``(3, 3)`` / ``(6, 6)`` vanilla, the rest fork #808. They are
#: EXAMPLES of :data:`_MIN_3D_NDF`, quoted in the refusals so a reader
#: recognises the note's own wording; the rule is
#: :func:`accepts_3d_ndf_pair`, never this set. Treating the note's list
#: as exhaustive refused ``(4, 6)`` — a u-p soil master under a shell
#: raft — which the fork takes like any other (adversarial review F1).
_NAMED_3D_NDF_PAIRS = ((3, 3), (6, 6), (3, 4), (4, 3), (4, 4), (3, 6),
                       (6, 4))


def accepts_3d_ndf_pair(ndf_i: int, ndf_j: int) -> bool:
    """May a 3D ``zeroLength`` join this ``(ndf_i, ndf_j)`` pair directly?

    ADR 96's rule verbatim: both ends ndf >= 3, nothing else. Declared
    here and imported by the emit-time gate
    (``opensees/_internal/build.py``) so the resolver's ``slave_ndf``
    contract and the deck's cannot drift — and deliberately the SAME
    rule ``validate_adaptive_element_endpoints`` already applies to
    every other zeroLength-family element in 3D.
    """
    return int(ndf_i) >= _MIN_3D_NDF and int(ndf_j) >= _MIN_3D_NDF

#: The slave ndf values the resolver accepts on a 2D line master.
#: ``None`` and ``2`` both mean "the slave matches the 2D continuum"
#: (direct zeroLength, no phantom); ``3`` means a beam slave and mints
#: the phantom bridge.
_SLAVE_NDF_VALUES = (None, 2, 3)

#: The slave ndf values accepted on a 3D surface master. ``None`` and
#: ``3`` are the plain continuum slave; ``4`` is a u-p soil node and
#: ``6`` a shell / beam node — all three connect directly, because every
#: pair they can form with a 3D continuum master satisfies
#: :func:`accepts_3d_ndf_pair`. ``2`` is refused by name: a 2-dof node
#: cannot carry a 3D translation triad at all.
_SLAVE_NDF_VALUES_3D = (None, 3, 4, 6)


def resolve_interface_records(
    node_tags,
    node_coords,
    *,
    master_nodes: Iterable[int],
    slave_nodes: Iterable[int],
    master_edges=None,
    master_facets=None,
    domain_elem_tags: Sequence[int],
    domain_elem_nodes: Sequence[Sequence[int]],
    normal_law: NormalLaw,
    tangential_law: TangentialLaw,
    thickness: float | None = None,
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
        ``ndm=2`` only, and **required** there: ``(n_edges, 2)`` node
        tags, the master boundary polyline's 2-node segments. Both
        endpoints of every edge must be master nodes; a duplicated edge
        is refused (it would double that stretch's tributary share).
    master_facets
        ``ndm=3`` only, and **required** there: the master surface's
        facet connectivity, a ragged sequence of tri3 / quad4 node-tag
        sequences. Quadratic facets are refused by name inside
        :func:`~apeGmsh._kernel.geometry._surface_frames.surface_frames`.
    domain_elem_tags, domain_elem_nodes
        The model's **highest-dimension** (2D or 3D continuum) elements
        — tags and per-element node tags (ragged is fine). Boundary
        curve / surface elements must NOT be in here: INV-5's owner pick
        is exact only over top-dimension elements, which
        ``_fem_extract`` never replicates across partitions.
    normal_law, tangential_law
        The declarative per-area laws (D1), stored verbatim on every
        record; translated to materials only at emit (S5).
    thickness
        ``ndm=2`` only, and **required** there: out-of-plane thickness,
        ``> 0`` — ``A_trib = ell_trib * thickness`` (D3). A 3D surface
        master has a real area, so passing it there is refused by name.
    tolerance
        Coincidence radius for the pairing.
    slave_ndf
        On a 2D line master: ``None`` / ``2`` — the slave matches the 2D
        continuum, so the zeroLength connects the two real nodes; ``3``
        — a beam slave, so each pair gets a phantom bridge (D4). On a 3D
        surface master: ``None`` / ``3`` / ``4`` (u-p soil) / ``6``
        (shell), all connecting directly — the pair is one of
        :func:`accepts_3d_ndf_pair` and needs no bridge. See the
        ``InterfaceDef`` docstring for why this is explicit and never
        inferred.
    ndm
        Model dimension: ``2`` (dim-1 line master) or ``3`` (dim-2
        surface master, TIMs A10 S2). Nothing else.
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
    ndm = int(ndm)
    if ndm not in (2, 3):
        raise NotImplementedError(
            f"interface: the master is a dim-1 line in a 2D model or a "
            f"dim-2 surface in a 3D model (ADR 0093 D2 / TIMs A10 S2), "
            f"got ndm={ndm}.")
    if ndm == 2:
        if slave_ndf not in _SLAVE_NDF_VALUES:
            raise ValueError(
                f"interface: slave_ndf must be one of {_SLAVE_NDF_VALUES} "
                f"(None/2 = the slave matches the 2D continuum, direct "
                f"zeroLength; 3 = beam slave, phantom bridge per ADR 0093 "
                f"D4), got {slave_ndf!r}.")
        if thickness is None or not (float(thickness) > 0.0):
            raise ValueError(
                f"interface: thickness must be > 0 on a 2D line master "
                f"(A_trib = ell_trib * thickness, ADR 0093 D3), got "
                f"{thickness!r}.")
        depth = float(thickness)
        if master_edges is None:
            raise ValueError(
                "interface: a 2D line master needs master_edges= (the "
                "boundary polyline's 2-node segments).")
        if master_facets is not None:
            raise ValueError(
                "interface: master_facets= is the 3D surface master's "
                "connectivity; a 2D line master takes master_edges=.")
    else:
        depth = 0.0   # no out-of-plane leg in 3D; never read below
        if slave_ndf not in _SLAVE_NDF_VALUES_3D:
            raise ValueError(
                f"interface: on a 3D surface master slave_ndf must be one "
                f"of {_SLAVE_NDF_VALUES_3D} (None/3 = a continuum slave, "
                f"4 = a u-p soil node, 6 = a shell / beam node — every "
                f"such pair is one the fork's zeroLength takes directly, "
                f"ADR 96), got {slave_ndf!r}.")
        if thickness is not None:
            raise ValueError(
                f"interface: thickness={thickness!r} was passed, but the "
                f"master is a 3D SURFACE and already has a real area — "
                f"A_trib comes from the facet-area accumulation (ADR 0093 "
                f"D3, the 3D reading). thickness is the 2D line master's "
                f"out-of-plane depth only. Drop thickness=.")
        if master_facets is None:
            raise ValueError(
                "interface: a 3D surface master needs master_facets= (the "
                "boundary surface's tri3 / quad4 connectivity).")
        if master_edges is not None:
            raise ValueError(
                "interface: master_edges= is the 2D line master's "
                "connectivity; a 3D surface master takes master_facets=.")

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

    data: "EdgeData | SurfaceFrames"
    if ndm == 2:
        data = edge_frames(
            master_edges, m_set, xyz,
            domain_elem_tags, domain_elem_nodes, label,
            verb="interface",
        )
        normals = _node_normals(data, label)
        a_trib = {
            t: ell * depth
            for t, ell in _tributary_lengths(data).items()
        }
        _assert_tributary_closure(
            pairs, a_trib, data.node_edges,
            data.total_length * depth, label,
            cell="boundary edge", measure="length")
    else:
        data = surface_frames(
            master_facets, m_set, xyz,
            domain_elem_tags, domain_elem_nodes, label,
            verb="interface",
        )
        # No thickness leg to multiply in: the facet-area accumulation
        # IS the area (D3, the 3D reading), and the closure reference is
        # the master surface's own area rather than a length.
        normals = data.normals
        a_trib = dict(data.a_trib)
        _assert_tributary_closure(
            pairs, a_trib, data.node_facets, data.total_area, label,
            cell="boundary facet", measure="area")

    records: list[InterfaceRecord] = []
    next_tag = (
        int(phantom_tag_start) if phantom_tag_start is not None
        else int(tags.max()) + 1
    )
    for master, slave in pairs:
        n = normals[master]
        # ADR 0093 D2 / INV-1: local-x is the master face's OUTWARD
        # normal. The tangential yield cap is symmetric (|tau| <= tau_b),
        # so the tangent directions are a determinism choice, not
        # mechanics — but they must be a *fixed* choice so two runs of
        # the same model produce the same deck.
        if ndm == 2:
            # local-y is z-hat x x-hat, the unique right-handed in-plane
            # completion in 2D.
            orient: tuple[float, ...] = (
                float(n[0]), float(n[1]), 0.0,
                float(-n[1]), float(n[0]), 0.0,
            )
        else:
            # (n, t1, t2) — the S1 frame verbatim. The first six floats
            # are still the zeroLength ``-orient`` argument; t2 rides
            # along because the 3D Coulomb law acts on two tangents.
            t1, t2 = data.tangents[master]  # type: ignore[union-attr]
            orient = tuple(
                float(v) for v in (*n, *t1, *t2)
            )
        backing = _backing_element(
            master, n, xyz[master], data, label)

        phantom_node = None
        phantom_coords = None
        phantom_ndf = None
        equal_dofs: list[NodePairRecord] = []
        if ndm == 2 and slave_ndf == 3:
            # D4 — the beam slave cannot take the zeroLength directly
            # (ZeroLength::setDomain refuses dofNd1 != dofNd2), so the
            # pair gets a 2-dof phantom standing on the SLAVE side and
            # rigidly tied to the beam node's translations. The
            # zeroLength then runs master continuum -> phantom (S5's
            # job); ``slave_node`` below still carries the real beam
            # node for provenance and for this nested equalDOF.
            #
            # 2D ONLY, and the ``ndm == 2`` above is the whole reason:
            # in 3D the same slave_ndf=3 needs no bridge, because the
            # fork's zeroLength takes the mixed pair itself
            # (:func:`accepts_3d_ndf_pair`). Minting one there would
            # put a phantom between two nodes that can already be joined.
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
            a_trib=float(a_trib[master]),
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
    a_trib: dict[int, float],
    node_cells: dict[int, list[int]],
    expected: float,
    label: str,
    *,
    cell: str,
    measure: str,
) -> None:
    """INV-3 — the pairs' tributary areas tile the whole master face.

    Two ways this fails, both loud: a paired node with a zero share (it
    lies on no boundary cell, so its spring would carry no area), and a
    master node the slave side never reached (the sum then falls short
    of the face, i.e. part of the interface is silently unsprung).

    Dimension-neutral by construction: ``node_cells`` is the master's
    node-to-boundary-cell map — ``EdgeData.node_edges`` on a 2D line
    master, ``SurfaceFrames.node_facets`` on a 3D surface one —
    ``expected`` is the whole master's area, already carrying the 2D
    ``thickness`` leg when there is one, and ``cell`` / ``measure`` name
    the master's own geometry in the refusals ("boundary edge" and
    "length" in 2D, "boundary facet" and "area" in 3D).
    """
    paired = {m for m, _ in pairs}
    zero = sorted(m for m in paired if a_trib.get(m, 0.0) <= 0.0)
    if zero:
        raise ValueError(
            f"interface{label}: paired master node(s) {zero[:20]} lie on "
            f"no master {cell}, so their tributary {measure} is zero "
            f"— their spring would carry no area (ADR 0093 INV-3).")

    on_boundary = {t for t, cells in node_cells.items() if cells}
    unpaired = sorted(on_boundary - paired)
    if unpaired:
        raise ValueError(
            f"interface{label}: master boundary node(s) {unpaired[:20]} "
            f"({len(unpaired)} of {len(on_boundary)}) were not paired to "
            f"any slave node, so the pairs' tributary areas do not tile "
            f"the master face (ADR 0093 INV-3). Scope the master with "
            f"master_entities= to the stretch the slave actually covers.")

    total = sum(a_trib[m] for m in paired)
    # A sum of n floats against an independently accumulated total
    # cannot be bit-exact — INV-3 budgets O(n * eps) relative.
    atol = 64.0 * np.finfo(float).eps * max(1, len(paired)) * abs(expected)
    if abs(total - expected) > atol:
        raise ValueError(
            f"interface{label}: tributary closure failed (ADR 0093 "
            f"INV-3) — sum(A_trib)={total!r} vs the master face's own "
            f"area={expected!r} (atol={atol!r}).")


# ── backing element (INV-5 / settled Q3) ────────────────────────────

def _backing_element(
    master: int,
    normal: ndarray,
    node_xyz: ndarray,
    data: "EdgeData | SurfaceFrames",
    label: str,
) -> int:
    """The domain element the pair's outward normal points AWAY from.

    Among the top-dimension domain elements incident on ``master``
    (never a boundary curve / surface element — those are the entities
    Gmsh replicates across a partition cut), the one whose centroid sits
    furthest *behind* the outward normal: the most-negative
    ``dot(centroid - node, n)``. Exact ties break to the lowest element
    tag, so the pick is deterministic and 1-vs-N byte identity survives
    (INV-5).

    Dimension-neutral: the arm is truncated to the normal's own width —
    2 on a 2D line master (``edge_frames`` normals are in-plane
    2-vectors), 3 on a 3D surface one — and ``adj`` / ``centroids`` /
    ``elem_tags`` carry the same meaning in both frame objects.
    """
    cands = data.adj.get(master) or []
    if not cands:
        raise ValueError(
            f"interface{label}: master node {master} has no incident "
            f"top-dimension domain element to back it (ADR 0093 INV-5).")
    width = int(np.asarray(normal).size)
    best = min(
        cands,
        key=lambda i: (
            float(np.dot((data.centroids[i] - node_xyz)[:width], normal)),
            data.elem_tags[i],
        ),
    )
    return int(data.elem_tags[best])
