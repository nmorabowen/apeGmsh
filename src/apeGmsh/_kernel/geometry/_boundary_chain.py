"""Boundary-polyline geometry - per-edge outward frames + the chained walk.

Lifted out of ``_kernel/resolvers/_interface_resolver.py`` (ADR 0093 D2)
so the contact lane can reuse the same frames, per the ADR 0041
precedent that moved Kuhn decomposition here out of
``ConstraintsComposite``.  The lift is a refactor: :func:`edge_frames`
is the old ``_edge_frames`` verbatim, with every refusal's ``interface``
prefix parameterised as ``verb=`` - so with the default the rendered
text is byte-identical and the ADR 0093 suite is the proof of behaviour
preservation.

:func:`chain_edges` is new.  It turns the (unordered, arbitrarily wound)
boundary edges into ONE head-to-tail chain, which is exactly the fork's
chained stride-2 pair list::

    contactSurface <tag> -master 2  n0 n1  n1 n2  n2 n3

The flat shorthand ``n0 n1 n2 n3`` is **silently legal** fork-side and
declares a HOLED master (``LadrunoContactHandler.cpp:1214`` - a node
used once is skipped) that converges to a wrong answer.  apeGmsh
generates this connectivity, so the walk exists to make that form
unreachable by construction rather than documented against.

Pure numpy - no gmsh, no OpenSees, no session.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy import ndarray

from ..resolvers._constraint_resolver._geom import _SpatialIndex

__all__ = [
    "DomainFrames", "domain_frames",
    "EdgeData", "edge_frames", "chain_edges", "refuse_wrong_side_master",
]


#: Directions/lengths below this are treated as degenerate rather than
#: normalised into noise (the resolver's own ``_ZERO_TOL``).
_ZERO_TOL = 1e-12

#: How far BEHIND a master segment its nearest slave node may sit before
#: the wrong-side guard counts that segment as confidently opposite, as a
#: multiple of the segment's own length.  ``2.0`` is the fork's own
#: narrow-phase reach: an NTS pair stops arming once penetration exceeds
#: roughly twice the local facet length (``LadrunoContact2D_guide.md``,
#: "Curved masters"), so a slave farther behind than this cannot be a
#: legitimately seeded initial penetration — that segment simply faces
#: away from the body it is supposed to contact.
_WRONG_SIDE_REACH = 2.0


def _duplicate_edge_consequence(verb: str, role: str = "master") -> str:
    """Why a duplicated boundary edge is refused, per calling lane.

    The refusal itself is lane-independent (one stretch of boundary,
    listed twice); only the consequence differs - the interface lane
    doubles that stretch's tributary share, the contact lane declares
    the same segment to the kernel twice.
    """
    if verb == "contact":
        return (f"a duplicated segment would declare that stretch of the "
                f"{role} to the contact kernel twice.")
    return ("a duplicated edge would double that stretch's tributary "
            "share.")


# ── per-edge outward frames ─────────────────────────────────────────

class DomainFrames:
    """The whole-domain scratch :func:`edge_frames` needs, computed once.

    ``elem_tags`` / ``elem_nodes`` are the domain elements parsed to int
    arrays, ``centroids[i]`` element ``i``'s centroid, and
    ``node_elems[t]`` the element indices incident on node ``t`` (absent
    for a node no element touches).

    It exists because building it is the expensive half of
    :func:`edge_frames` — one Python-level pass over EVERY top-dimension
    element, with a numpy allocation per element — while depending on
    nothing but the mesh.  Two calls over the same mesh recompute it
    identically, and the 2D mortar lane makes two per contact (master
    chain + slave chain), so the caller builds it once and passes it in.
    The lists in ``node_elems`` are SHARED across the ``EdgeData`` objects
    derived from it; every consumer reads them (``edge_frames``' owner
    intersection, the interface resolver's backing pick) and none mutates.
    """

    __slots__ = ("elem_tags", "elem_nodes", "centroids", "node_elems")

    def __init__(self, elem_tags, elem_nodes, centroids, node_elems) -> None:
        self.elem_tags = elem_tags
        self.elem_nodes = elem_nodes
        self.centroids = centroids
        self.node_elems = node_elems


def domain_frames(
    domain_elem_tags: Sequence[int],
    domain_elem_nodes: Sequence[Sequence[int]],
    xyz: dict[int, ndarray],
    label: str,
    *,
    verb: str = "interface",
    role: str = "master",
) -> DomainFrames:
    """Parse the domain elements once: centroids + node adjacency.

    Lifted verbatim out of :func:`edge_frames` (same checks, same order,
    same messages) so it can be computed once per resolve instead of once
    per surface.  ``label`` / ``verb`` / ``role`` only prefix the refusals.
    """
    elem_tags = [int(t) for t in domain_elem_tags]
    elem_nodes = [np.asarray(n, dtype=int).ravel() for n in domain_elem_nodes]
    if len(elem_tags) != len(elem_nodes):
        raise ValueError(
            f"{verb}{label}: domain_elem_tags ({len(elem_tags)}) and "
            f"domain_elem_nodes ({len(elem_nodes)}) disagree in length.")
    if not elem_tags:
        raise ValueError(
            f"{verb}{label}: no 2D domain elements were supplied — "
            f"the outward sign (ADR 0093 D2) and the backing element "
            f"(INV-5) are both derived from them.")

    node_elems: dict[int, list[int]] = {}
    centroids = np.empty((len(elem_tags), 3), dtype=float)
    for i, conn in enumerate(elem_nodes):
        try:
            pts = np.array([xyz[int(k)] for k in conn], dtype=float)
        except KeyError as exc:
            raise ValueError(
                f"{verb}{label}: domain element {elem_tags[i]} "
                f"references node {exc.args[0]}, which is not in the "
                f"model node pool.") from exc
        centroids[i] = pts.mean(axis=0)
        for k in conn:
            node_elems.setdefault(int(k), []).append(i)

    return DomainFrames(elem_tags, elem_nodes, centroids, node_elems)


class EdgeData:
    """The master polyline's per-edge geometry, indexed for reuse.

    ``normals[e]`` is edge ``e``'s **outward** in-plane unit normal
    (signed against its owning domain element's centroid),
    ``lengths[e]`` its length, ``node_edges[t]`` the edges incident on
    master node ``t``, and ``adj[t]`` the domain-element indices
    incident on it (with ``elem_tags`` / ``centroids`` alongside).
    """

    __slots__ = ("edges", "normals", "lengths", "node_edges",
                 "adj", "elem_tags", "centroids", "total_length")

    def __init__(self, edges, normals, lengths, node_edges,
                 adj, elem_tags, centroids) -> None:
        self.edges = edges
        self.normals = normals
        self.lengths = lengths
        self.node_edges = node_edges
        self.adj = adj
        self.elem_tags = elem_tags
        self.centroids = centroids
        self.total_length = float(sum(lengths))


def edge_frames(
    master_edges,
    m_set: set[int],
    xyz: dict[int, ndarray],
    domain_elem_tags: Sequence[int],
    domain_elem_nodes: Sequence[Sequence[int]],
    label: str,
    *,
    verb: str = "interface",
    role: str = "master",
    frames: "DomainFrames | None" = None,
) -> EdgeData:
    """Per-edge outward normals + lengths + the domain adjacency map.

    ``verb`` prefixes every refusal ("interface" / "contact"); with the
    default the rendered text is byte-identical to the pre-extraction
    resolver's, which is what makes the ADR 0093 suite the proof of
    behaviour preservation.

    ``role`` names the surface in those refusals.  It exists because the
    2D mortar lane runs this walk on the SLAVE side too
    (``-slave-segments``), where every message saying "master" would point
    the reader at the wrong curve; the default keeps the interface lane's
    text byte-identical for the same reason ``verb`` does.

    ``frames`` is the whole-domain scratch (:class:`DomainFrames`).  Pass
    one when several surfaces share a mesh — the 2D mortar lane builds two
    chains per contact — to skip its per-element pass; ``None`` builds it
    here, which is what keeps the interface lane on the original path.
    """
    edges = np.asarray(master_edges, dtype=int).reshape(-1, 2)
    if edges.size == 0:
        raise ValueError(
            f"{verb}{label}: the {role} label carries no boundary "
            f"line elements — the outward normal and the tributary "
            f"length are both derived from them (is the {role} a meshed "
            f"curve?).")
    stray = sorted({int(t) for t in edges.ravel()} - m_set)
    if stray:
        raise ValueError(
            f"{verb}{label}: {role} boundary edge(s) reference "
            f"node(s) {stray[:20]} that are not in the {role} node set.")
    seen: dict[frozenset, int] = {}
    for e, (a, b) in enumerate(edges):
        key = frozenset((int(a), int(b)))
        if len(key) == 1:
            raise ValueError(
                f"{verb}{label}: {role} boundary edge {e} is "
                f"degenerate (both endpoints are node {int(a)}).")
        if key in seen:
            raise ValueError(
                f"{verb}{label}: {role} boundary edge "
                f"({int(a)}, {int(b)}) appears twice (rows {seen[key]} "
                f"and {e}) — {_duplicate_edge_consequence(verb, role)} "
                f"Deduplicate the {role} entities.")
        seen[key] = e

    if frames is None:
        frames = domain_frames(
            domain_elem_tags, domain_elem_nodes, xyz, label,
            verb=verb, role=role)
    elem_tags = frames.elem_tags
    centroids = frames.centroids
    # Narrow the global map to this surface's nodes, keeping a key for
    # every one of them (a node no element touches maps to the empty list
    # and draws the "no adjacent 2D domain element" refusal below, exactly
    # as the per-call build did).
    adj = {t: frames.node_elems.get(t, []) for t in m_set}

    normals: list[ndarray] = []
    lengths: list[float] = []
    node_edges: dict[int, list[int]] = {t: [] for t in m_set}
    for e, (a, b) in enumerate(edges):
        a, b = int(a), int(b)
        pa, pb = xyz[a], xyz[b]
        tan = pb - pa
        length = float(np.linalg.norm(tan))
        if length <= _ZERO_TOL:
            raise ValueError(
                f"{verb}{label}: {role} boundary edge ({a}, {b}) has "
                f"zero length.")
        if abs(float(tan[2])) > 1e-9 * length:
            raise NotImplementedError(
                f"{verb}{label}: {role} boundary edge ({a}, {b}) is "
                f"out of the z=const plane (dz={float(tan[2])!r}). Only "
                f"in-plane 2D line {role}s are implemented — ADR 0093 "
                f"D2.")
        # In-plane normal candidate; the sign is decided below, never by
        # the edge's node ordering (a mesh's winding is not a contract).
        n = np.array([tan[1], -tan[0]], dtype=float) / length

        owners = sorted(set(adj[a]) & set(adj[b]))
        if not owners:
            raise ValueError(
                f"{verb}{label}: {role} boundary edge ({a}, {b}) has "
                f"no adjacent 2D domain element. The outward sign is "
                f"fixed against the adjacent element's centroid "
                f"(ADR 0093 D2) — is the {role} a boundary curve of the "
                f"meshed continuum?")
        if len(owners) > 1:
            raise ValueError(
                f"{verb}{label}: {role} boundary edge ({a}, {b}) is "
                f"shared by {len(owners)} 2D domain elements "
                f"{[elem_tags[i] for i in owners]} — it is an INTERIOR "
                f"edge with material on both sides, so it has no "
                f"outward direction. The {role} must be a free boundary "
                f"of the continuum.")
        centroid = centroids[owners[0]]
        mid = 0.5 * (pa + pb)
        arm = mid - centroid
        d = float(np.dot(n, arm[:2]))
        if abs(d) <= _ZERO_TOL * max(1.0, float(np.linalg.norm(arm))):
            raise ValueError(
                f"{verb}{label}: cannot sign the outward normal of "
                f"{role} edge ({a}, {b}) — its owning element "
                f"{elem_tags[owners[0]]} has its centroid on the edge "
                f"line (a degenerate / inverted element?).")
        if d < 0.0:
            n = -n

        normals.append(n)
        lengths.append(length)
        node_edges[a].append(e)
        node_edges[b].append(e)

    return EdgeData(edges, normals, lengths, node_edges,
                    adj, elem_tags, centroids)


# -- the chained stride-2 walk ---------------------------------------

def chain_edges(
    data: EdgeData,
    xyz: dict[int, ndarray],
    label: str,
    *,
    verb: str = "contact",
    role: str = "master",
) -> ndarray:
    """Order + direct the boundary edges into ONE head-to-tail chain.

    Returns ``(n_edges, 2)`` node tags in traversal order, row ``k``'s
    second column equal to row ``k+1``'s first - the fork's chained
    stride-2 pair list, flattened by the emitter into
    ``n0 n1  n1 n2  n2 n3``.

    The direction of each edge is **not** re-derived here.
    :func:`edge_frames` already fixed it: ``normals[e]`` is the edge's
    outward normal, signed against the owning element's centroid, and
    since it was built from ``(t_y, -t_x)/L`` while the fork's
    ``sigma = +1`` convention is ``perp(t) = (-t_y, t_x)/L``, the two
    are exact negatives of each other.  So ``dot(perp(t), normals[e])``
    is **exactly +-1** - a sign read, not a tolerance test.  The edge is
    emitted as listed when that sign is ``+1`` and reversed when it is
    ``-1``, so that every emitted segment satisfies
    ``perp(t) == normals[e]``.

    That direction is the load-bearing one, not a free convention: the
    fork's ``sigma = +1`` normal IS ``perp(t)``, and it must point AT
    the slave, i.e. out of the master's material.  Equivalently - the
    fork guide's phrasing - the slave lies to the **left** of the
    chain's travel, ``(-t_y, t_x)`` being the left of ``t``.  Winding
    the other way round leaves the fork normal pointing into the master
    and the contact resolves the wrong way while still converging, so
    this sign is pinned by
    ``test_winding_puts_the_slave_on_the_left``.

    With every edge directed, the whole refusal taxonomy of the walk
    falls out of one condition: **each node is the start of at most one
    directed edge and the end of at most one**.  Three refusals are
    named here - branching, disjoint runs, and the directed clash;
    everything else (degenerate / duplicated / zero-length /
    out-of-plane edges, an unbacked or interior edge, a centroid on the
    edge line) is inherited from :func:`edge_frames`.

    ``role`` names the surface in the refusals below, for the 2D mortar
    lane's ``-slave-segments`` side (see :func:`edge_frames`).

    Open chains, single segments and **closed loops** are all supported
    - the fork wrap is explicitly legal
    (``LadrunoContactHandler.cpp:1219``).  A closed loop starts at the
    lowest node tag on it, so a repeated ``get_fem_data()`` emits the
    same deck.
    """
    edges = np.asarray(data.edges, dtype=int).reshape(-1, 2)

    # Branching first: ``node_edges`` is undirected and unordered, which
    # is not enough to walk with, but its DEGREE is enough to name the
    # branch before the walk reports a vaguer multiplicity.
    branching = sorted(t for t, es in data.node_edges.items() if len(es) > 2)
    if branching:
        raise ValueError(
            f"{verb}{label}: {role} node(s) {branching[:20]} carry "
            f"{[len(data.node_edges[t]) for t in branching[:20]]} boundary "
            f"segments each - the {role} BRANCHES there, and a branching "
            f"surface has no single head-to-tail chain. Declare one "
            f"contact() per branch.")

    directed: list[tuple[int, int]] = []
    for e, (a, b) in enumerate(edges):
        a, b = int(a), int(b)
        tan = xyz[b] - xyz[a]
        length = float(np.linalg.norm(tan))
        perp = np.array([-tan[1], tan[0]], dtype=float) / length
        sign = float(np.dot(perp, np.asarray(data.normals[e], dtype=float)))
        directed.append((a, b) if sign > 0.0 else (b, a))

    starts_at: dict[int, int] = {}
    ends_at: dict[int, int] = {}
    for e, (a, b) in enumerate(directed):
        clash = None
        if a in starts_at:
            clash = (a, starts_at[a], "start")
        elif b in ends_at:
            clash = (b, ends_at[b], "end")
        if clash is not None:
            node, other, which = clash
            raise ValueError(
                f"{verb}{label}: {role} node {node} is the {which} of both "
                f"segment {tuple(int(x) for x in edges[other])} and segment "
                f"{tuple(int(x) for x in edges[e])} once each segment is "
                f"wound against its own material - the boundary either "
                f"doubles back there, or the two segments have the "
                f"continuum on OPPOSITE sides, so there is no consistent "
                f"traversal through it. Split the {role} at that node into "
                f"separate contact() calls. (Same physical situation the "
                f"interface lane's cancelling / reentrant node-normal "
                f"refusals detect, ADR 0093 D2.)")
        starts_at[a] = e
        ends_at[b] = e

    runs = _directed_runs(directed, starts_at)
    if len(runs) > 1:
        span = [
            (int(directed[r[0]][0]), int(directed[r[-1]][1])) for r in runs
        ]
        raise ValueError(
            f"{verb}{label}: the {role}'s {len(edges)} boundary segments "
            f"form {len(runs)} DISJOINT runs, spanning {span[:20]} "
            f"(start, end) - they cannot be listed as one chained "
            f"stride-2 pair list. A disjoint listing is silently legal "
            f"fork-side (LadrunoContactHandler.cpp:1214 skips a node used "
            f"once), so emitting it would declare a HOLED {role} that "
            f"converges to a wrong answer. Declare one contact() per "
            f"connected stretch.")

    return np.asarray([directed[e] for e in runs[0]], dtype=np.int64)


def refuse_wrong_side_master(
    chain,
    xyz: dict[int, ndarray],
    slave_tags,
    label: str,
    *,
    verb: str = "contact",
) -> None:
    """Refuse a master whose boundary confidently FACES AWAY from the slave.

    The check apeGmsh has to own, on BOTH 2D lanes.  Nothing fork-side
    refuses it: the interface-level centroid vote
    (``LadrunoContactHandler.cpp:440-523``) only picks a per-interface
    SIGN, so on a far-side master it resolves happily — unanimously — and
    orients a contact against a boundary the slave never reaches; and
    ``-outward winding`` (2D NTS) bypasses the vote by construction, which
    removes even the split/degenerate refusals that might have fired by
    accident.  Either way the deck runs, converges, balances, and
    transmits nothing.  apeGmsh generates the master chain, so apeGmsh
    owes the deck the check.

    Deliberately WEAKER than the vote it stands in for, because the two
    cases that vote refuses wrongly are exactly the two winding exists to
    unlock:

    * **flush** interfaces give ``dot ≈ 0``, not negative — the guard is
      silent on the masonry joint / footing-on-soil workhorse;
    * **curved and closed** masters are tested per segment against that
      segment's OWN nearest slave, never against one global centroid, so
      a ring or an indenter profile passes with no split to refuse.

    It fires only on *confidently opposite*: a strict majority of
    segments whose nearest slave node lies more than
    :data:`_WRONG_SIDE_REACH` × that segment's length behind it.  The
    headline case is the far side of the beam, the wrong edge of the
    block.

    That threshold scales with the segment, so on a FINE master mesh an
    over-seeded initial overlap reaches it too — and the refusal is still
    right, because the fork's own narrow phase stops arming a pair past
    roughly the same reach, so such a deck would not transmit either way.
    The message therefore names both causes rather than diagnosing a
    wrong-side master outright: at this distance the two are not
    distinguishable from the geometry alone, and telling a user to
    re-name a boundary that is in fact correct would send them the wrong
    way.

    ``chain`` is :func:`chain_edges`' output, so each row's normal is
    ``perp(t)`` by construction (the invariant pinned by
    ``test_winding_puts_the_slave_on_the_left``) and is recomputed here
    rather than indexed back through ``EdgeData`` — one source of truth,
    no row-to-edge map to get wrong.
    """
    rows = np.asarray(chain, dtype=int).reshape(-1, 2)
    slaves = [int(t) for t in slave_tags]
    if rows.shape[0] == 0 or not slaves:
        return

    pts = np.asarray([xyz[t][:2] for t in slaves], dtype=float)
    index = _SpatialIndex(pts)

    opposed: list[tuple[int, int, float, float]] = []
    for a, b in rows:
        pa = np.asarray(xyz[int(a)], dtype=float)[:2]
        pb = np.asarray(xyz[int(b)], dtype=float)[:2]
        tan = pb - pa
        length = float(np.linalg.norm(tan))
        if length <= _ZERO_TOL:
            continue
        normal = np.array([-tan[1], tan[0]], dtype=float) / length
        mid = 0.5 * (pa + pb)
        _, idx = index.query(mid)
        gap = float(np.dot(normal, pts[int(idx)] - mid))
        if gap < -_WRONG_SIDE_REACH * length:
            opposed.append((int(a), int(b), gap, length))

    if 2 * len(opposed) <= rows.shape[0]:
        return

    a, b, gap, length = opposed[0]
    raise ValueError(
        f"{verb}{label}: {len(opposed)} of the master's {rows.shape[0]} "
        f"segments FACE AWAY from the slave — segment ({a}, {b}) has its "
        f"nearest slave node {abs(gap):.6g} BEHIND it (more than "
        f"{_WRONG_SIDE_REACH:g}x its own length {length:.6g}), i.e. on the "
        f"master's material side, where a contact can never arm. Two "
        f"causes look identical at this distance, so check both: either "
        f"the named master boundary is not the one that meets the slave "
        f"(the far-side / wrong-edge mistake — name the boundary curve "
        f"that faces the slave body), or the initial overlap seeded "
        f"between the bodies is too large for this mesh, in which case "
        f"reduce it below {_WRONG_SIDE_REACH:g}x the master segment "
        f"length ({_WRONG_SIDE_REACH * length:.6g} here) or coarsen the "
        f"master — the fork's narrow phase stops arming a pair past that "
        f"reach too, so the deck would not transmit either way. (apeGmsh "
        f"owns this check because nothing fork-side refuses it: the "
        f"orientation vote only picks a SIGN and resolves happily against "
        f"a boundary that never meets the slave, and the "
        f"declared-winding orientation apeGmsh emits on the NTS lane "
        f"BYPASSES the fork's own centroid vote outright — either way the "
        f"deck would converge on the wrong boundary instead of "
        f"aborting.)")


def _directed_runs(
    directed: Sequence[tuple[int, int]],
    starts_at: dict[int, int],
) -> list[list[int]]:
    """Every maximal directed run, as lists of edge indices.

    With branching already refused and no node the start (or end) of two
    edges, the directed graph is a disjoint union of simple paths and
    simple cycles, so this terminates and covers every edge exactly
    once.  Open runs come first in ascending start-node order; each
    cycle starts at the lowest node tag on it - both for determinism
    (ADR 0027: two ``get_fem_data()`` calls must emit the same deck),
    matching the lowest-tag tie-break the interface resolver's backing
    pick already uses.
    """
    n = len(directed)
    used: set[int] = set()
    runs: list[list[int]] = []
    ends = {b for _, b in directed}

    for start in sorted(a for a in starts_at if a not in ends):
        run: list[int] = []
        node = start
        while node in starts_at and starts_at[node] not in used:
            e = starts_at[node]
            used.add(e)
            run.append(e)
            node = directed[e][1]
        if run:
            runs.append(run)

    while len(used) < n:
        seed = next(e for e in range(n) if e not in used)
        # `seen` mirrors `cycle_nodes` purely for O(1) membership: the
        # list is kept because the cycle's start is its min() and the
        # walk below needs a deterministic one, but scanning the list
        # each step made a finely-meshed ring quadratic — and a ring is
        # exactly what wants many segments (a coarse one transmits
        # nothing, ADR-85 F1).
        cycle_nodes: list[int] = []
        seen: set[int] = set()
        node = directed[seed][0]
        while node not in seen:
            cycle_nodes.append(node)
            seen.add(node)
            node = directed[starts_at[node]][1]
        start = min(cycle_nodes)
        run = []
        node = start
        while True:
            e = starts_at[node]
            used.add(e)
            run.append(e)
            node = directed[e][1]
            if node == start:
                break
        runs.append(run)

    return runs
