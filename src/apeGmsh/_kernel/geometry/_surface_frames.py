"""Boundary-surface geometry â€” per-node outward frames + tributary areas.

The 3D sibling of :mod:`_boundary_chain`, for ``g.constraints.interface()``
on a dim-2 master (ADR 0093 D2/D3, 3D slice S1).  Where the 2D lane walks
boundary *edges* of a 2D continuum and signs each edge normal against its
owning element's centroid, this walks boundary *facets* of a 3D continuum
and does the same thing one dimension up.  Everything else follows: the
per-node normal is the renormalised average of the adjacent facet normals,
and the tributary area is the facet-area accumulation.

Three things are decided here rather than inherited, and each is pinned by
a test in ``tests/_kernel/geometry/test_surface_frames.py``:

* **Winding is not a contract.**  A facet's listed node order only supplies
  a normal *candidate*; the sign comes from the adjacent 3D element's
  centroid, exactly as :func:`_boundary_chain.edge_frames` does in 2D.  So
  reversing one facet's winding changes nothing.
* **The averaging weight is uniform**, not area- or angle-weighted â€” the
  literal sibling of the 2D ``_node_normals``, which sums the adjacent
  *unit* edge normals and renormalises.  It is also the only weight under
  which a cube's corner node comes out at ``(1,1,1)/sqrt(3)`` and an edge
  node at the normalised sum of its two face normals, which is the frame a
  reader expects from a box.
* **A reentrant fold is refused, a convex corner is not.**  The two are
  indistinguishable from ``dot(n_i, n_j)`` alone (a 90Â° convex corner and a
  270Â° reentrant one both give zero), so the *sense* of the fold is read
  from the facet centroids and the refusal is stated on the interior
  dihedral angle.  See :data:`_MAX_REENTRANT_FOLD_DEG`.

S1 is kernel-only: nothing here is wired into the resolver, which still
refuses ``ndm != 2``.  Pure numpy â€” no gmsh, no OpenSees, no session.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy import ndarray

from ._boundary_chain import DomainFrames, domain_frames

__all__ = ["SurfaceFrames", "surface_frames"]


#: Directions/areas below this are treated as degenerate rather than
#: normalised into noise (the 2D lane's own ``_ZERO_TOL``).
_ZERO_TOL = 1e-12

#: Facet sizes S1 handles, by node count.  Every node of a linear facet is
#: a corner, so "split the area equally among the corners" and "split it
#: equally among the nodes" are the same rule â€” which is what keeps the
#: tributary model identical to the one ``distributing_coupling(
#: weighting="area")`` already uses (``_resolver.py:742``).
_LINEAR_FACETS = {3: "tri3", 4: "quad4"}

#: Facet sizes that are recognised but deferred: their mid-side nodes need
#: a shape-function-weighted split, not an equal one, so accepting them
#: with the linear rule would quietly mis-weight every spring.
_QUADRATIC_FACETS = {6: "tri6", 8: "quad8", 9: "quad9"}

#: How far the surface may fold back INTO its own body at a node before
#: the averaged normal is refused, in degrees of reentrancy (interior
#: dihedral minus 180Â°).  A flat patch folds 0Â°, a convex box corner folds
#: -90Â° (never refused â€” that is the box the verb exists to spring), and
#: the canonical L-shaped reentrant corner folds +90Â°.  45Â° sits clear of
#: both: by then the averaged normal already leans 22.5Â° into the material
#: it is supposed to point out of, so the pair's local-x is meaningless and
#: an ENT law would arm against the wrong side.
_MAX_REENTRANT_FOLD_DEG = 45.0

#: The global axes, in the order the tangent convention scans them.
_AXES = np.eye(3, dtype=float)


class SurfaceFrames:
    """The master surface's per-facet and per-node geometry.

    Per facet (index ``f``, in the order supplied): ``facets[f]`` the node
    tags, ``facet_normals[f]`` the **outward** unit normal,
    ``facet_areas[f]`` the area, ``facet_centroids[f]`` the centroid.

    Per master node ``t``: ``node_facets[t]`` the incident facet indices,
    ``normals[t]`` the outward unit normal, ``tangents[t]`` the ``(2, 3)``
    in-plane pair ``(t1, t2)`` completing the right-handed frame, and
    ``a_trib[t]`` the tributary area.

    ``total_area`` is the surface's own area, the closure reference for
    ``sum(a_trib)`` (ADR 0093 INV-3, the 3D reading).
    """

    __slots__ = ("facets", "facet_normals", "facet_areas", "facet_centroids",
                 "node_facets", "normals", "tangents", "a_trib",
                 "elem_tags", "adj", "centroids", "total_area")

    def __init__(self, facets, facet_normals, facet_areas, facet_centroids,
                 node_facets, normals, tangents, a_trib,
                 elem_tags, adj, centroids) -> None:
        self.facets = facets
        self.facet_normals = facet_normals
        self.facet_areas = facet_areas
        self.facet_centroids = facet_centroids
        self.node_facets = node_facets
        self.normals = normals
        self.tangents = tangents
        self.a_trib = a_trib
        self.elem_tags = elem_tags
        self.adj = adj
        self.centroids = centroids
        self.total_area = float(sum(facet_areas))


def surface_frames(
    master_facets,
    m_set: set[int],
    xyz: dict[int, ndarray],
    domain_elem_tags: Sequence[int],
    domain_elem_nodes: Sequence[Sequence[int]],
    label: str,
    *,
    verb: str = "interface",
    role: str = "master",
    frames: "DomainFrames | None" = None,
) -> SurfaceFrames:
    """Per-node outward frames + tributary areas on a dim-2 master.

    ``master_facets`` is the surface's facet connectivity â€” a ragged
    sequence of node-tag sequences, tri3 or quad4 (:data:`_LINEAR_FACETS`).
    ``domain_elem_tags`` / ``domain_elem_nodes`` are the model's
    **highest-dimension** (3D) elements; the outward sign is fixed against
    the owning one's centroid, so boundary surface elements must not be in
    there.

    ``verb`` / ``role`` prefix and name the refusals, as in
    :func:`_boundary_chain.edge_frames`.  ``frames`` is the whole-domain
    scratch: pass one when several surfaces share a mesh, ``None`` builds
    it here.
    """
    facets = _parse_facets(master_facets, m_set, label, verb, role)

    if frames is None:
        frames = domain_frames(
            domain_elem_tags, domain_elem_nodes, xyz, label,
            verb=verb, role=role, elem_kind="3D")
    elem_tags = frames.elem_tags
    centroids = frames.centroids
    adj = {t: frames.node_elems.get(t, []) for t in m_set}

    facet_normals: list[ndarray] = []
    facet_areas: list[float] = []
    facet_centroids: list[ndarray] = []
    node_facets: dict[int, list[int]] = {t: [] for t in m_set}
    for f, facet in enumerate(facets):
        pts = np.array([xyz[t] for t in facet], dtype=float)
        # Fan triangulation from the first node â€” the polygon model the
        # load resolver's ``face_area`` and the RBE3 ``_tributary_areas``
        # already share, so a quad's area here is the same number those
        # lanes would report.  The vector sum of the same fan gives the
        # normal candidate; on a planar facet its magnitude IS the area.
        vec = np.zeros(3, dtype=float)
        area = 0.0
        for i in range(1, len(facet) - 1):
            tri = np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
            vec += 0.5 * tri
            area += 0.5 * float(np.linalg.norm(tri))
        norm = float(np.linalg.norm(vec))
        if norm <= _ZERO_TOL or area <= _ZERO_TOL:
            raise ValueError(
                f"{verb}{label}: {role} facet {tuple(facet)} is degenerate "
                f"(area {area!r}) â€” it has no normal to orient a pair with.")
        n = vec / norm
        c = pts.mean(axis=0)

        owners = _facet_owner(facet, adj, elem_tags, label, verb, role)
        # The sign is decided here and nowhere else: the facet's listed
        # winding is not a contract (ADR 0093 D2, the 3D reading of the 2D
        # centroid sign-fix).
        arm = c - centroids[owners]
        d = float(np.dot(n, arm))
        if abs(d) <= _ZERO_TOL * max(1.0, float(np.linalg.norm(arm))):
            raise ValueError(
                f"{verb}{label}: cannot sign the outward normal of {role} "
                f"facet {tuple(facet)} â€” its owning element "
                f"{elem_tags[owners]} has its centroid in the facet plane "
                f"(a degenerate / inverted element?).")
        if d < 0.0:
            n = -n

        facet_normals.append(n)
        facet_areas.append(area)
        facet_centroids.append(c)
        for t in facet:
            node_facets[t].append(f)

    _refuse_reentrant_folds(
        facets, facet_normals, facet_centroids, node_facets,
        label, verb, role)

    normals: dict[int, ndarray] = {}
    tangents: dict[int, ndarray] = {}
    a_trib: dict[int, float] = {}
    for t, f_list in node_facets.items():
        if not f_list:
            continue
        normals[t] = _node_normal(t, f_list, facet_normals, label, verb, role)
        tangents[t] = _tangent_pair(normals[t])
        a_trib[t] = float(sum(
            facet_areas[f] / len(facets[f]) for f in f_list))

    return SurfaceFrames(
        facets, facet_normals, facet_areas, facet_centroids, node_facets,
        normals, tangents, a_trib, elem_tags, adj, centroids)


# â”€â”€ facet parsing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _parse_facets(
    master_facets,
    m_set: set[int],
    label: str,
    verb: str,
    role: str,
) -> list[tuple[int, ...]]:
    """The facet connectivity, validated â€” the sibling of the edge checks.

    Same taxonomy one dimension up: an empty surface, a facet touching a
    node outside the surface's own node set, a facet repeating a node, and
    a facet listed twice (which would double that patch's tributary share).
    """
    facets: list[tuple[int, ...]] = []
    for row in master_facets:
        facets.append(tuple(int(t) for t in np.asarray(row, dtype=int).ravel()))
    if not facets:
        raise ValueError(
            f"{verb}{label}: the {role} label carries no boundary surface "
            f"elements â€” the outward normal and the tributary area are both "
            f"derived from them (is the {role} a meshed surface?).")

    stray = sorted({t for facet in facets for t in facet} - m_set)
    if stray:
        raise ValueError(
            f"{verb}{label}: {role} boundary facet(s) reference node(s) "
            f"{stray[:20]} that are not in the {role} node set.")

    seen: dict[frozenset, int] = {}
    for f, facet in enumerate(facets):
        kind = _LINEAR_FACETS.get(len(facet))
        if kind is None:
            quad = _QUADRATIC_FACETS.get(len(facet))
            if quad is not None:
                raise NotImplementedError(
                    f"{verb}{label}: {role} boundary facet {facet} is a "
                    f"{quad} â€” only linear facets "
                    f"({', '.join(_LINEAR_FACETS.values())}) carry the "
                    f"equal-share tributary rule (ADR 0093 D3). A quadratic "
                    f"facet needs a shape-function-weighted split, which is "
                    f"not in the 3D S1 slice; mesh the {role} with linear "
                    f"surface elements.")
            raise ValueError(
                f"{verb}{label}: {role} boundary facet {facet} has "
                f"{len(facet)} nodes â€” expected a tri3 (3) or a quad4 (4).")
        key = frozenset(facet)
        if len(key) != len(facet):
            raise ValueError(
                f"{verb}{label}: {role} boundary facet {facet} is degenerate "
                f"(it repeats a node).")
        if key in seen:
            raise ValueError(
                f"{verb}{label}: {role} boundary facet {facet} appears twice "
                f"(rows {seen[key]} and {f}) â€” a duplicated facet would "
                f"double that patch's tributary share. Deduplicate the "
                f"{role} entities.")
        seen[key] = f
    return facets


def _facet_owner(
    facet: tuple[int, ...],
    adj: dict[int, list[int]],
    elem_tags: list[int],
    label: str,
    verb: str,
    role: str,
) -> int:
    """The single 3D element index the facet bounds.

    The intersection of the incident-element sets over ALL the facet's
    nodes; the two refusals are the 2D lane's verbatim, one dimension up â€”
    an unbacked facet has no outward direction to derive, and a facet with
    material on both sides is an interior face, not a boundary.
    """
    owners: set[int] | None = None
    for t in facet:
        here = set(adj.get(t) or ())
        owners = here if owners is None else (owners & here)
        if not owners:
            break
    if not owners:
        raise ValueError(
            f"{verb}{label}: {role} boundary facet {facet} has no adjacent "
            f"3D domain element. The outward sign is fixed against the "
            f"adjacent element's centroid (ADR 0093 D2) â€” is the {role} a "
            f"boundary surface of the meshed continuum?")
    if len(owners) > 1:
        raise ValueError(
            f"{verb}{label}: {role} boundary facet {facet} is shared by "
            f"{len(owners)} 3D domain elements "
            f"{[elem_tags[i] for i in sorted(owners)]} â€” it is an INTERIOR "
            f"face with material on both sides, so it has no outward "
            f"direction. The {role} must be a free boundary of the "
            f"continuum.")
    return int(next(iter(owners)))


# â”€â”€ the fold refusal (D2, the 3D reading) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _refuse_reentrant_folds(
    facets: list[tuple[int, ...]],
    facet_normals: list[ndarray],
    facet_centroids: list[ndarray],
    node_facets: dict[int, list[int]],
    label: str,
    verb: str,
    role: str,
) -> None:
    """Refuse a node where the surface folds back into its own body.

    ``dot(n_i, n_j)`` alone cannot tell a convex corner from a reentrant
    one â€” a cube's 90Â° corner and an L's 270Â° corner both give exactly
    zero â€” so the sense is read from the centroids: facet ``j``'s centroid
    sitting AHEAD of facet ``i``'s outward normal (and symmetrically) means
    the surface turns towards the material, i.e. reentrant.  The interior
    dihedral is then ``180Â° Â± angle(n_i, n_j)``, and the refusal is stated
    on it, against :data:`_MAX_REENTRANT_FOLD_DEG`.
    """
    for t, f_list in sorted(node_facets.items()):
        for a in range(len(f_list)):
            for b in range(a + 1, len(f_list)):
                i, j = f_list[a], f_list[b]
                ni, nj = facet_normals[i], facet_normals[j]
                cos_a = float(np.clip(np.dot(ni, nj), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cos_a)))
                sep = facet_centroids[j] - facet_centroids[i]
                sense = float(np.dot(ni, sep) - np.dot(nj, sep))
                tol = _ZERO_TOL * max(1.0, float(np.linalg.norm(sep)))
                fold = angle if sense > tol else -angle
                if fold > _MAX_REENTRANT_FOLD_DEG:
                    raise ValueError(
                        f"{verb}{label}: {role} node {t} sits on a REENTRANT "
                        f"fold â€” its adjacent facets {facets[i]} and "
                        f"{facets[j]} meet at an interior dihedral of "
                        f"{180.0 + fold:.4g}Â° (a fold of {fold:.4g}Â° back "
                        f"into the body, past the {_MAX_REENTRANT_FOLD_DEG:g}Â° "
                        f"limit), so their averaged normal points into the "
                        f"material instead of out of it and the node has no "
                        f"single outward direction (ADR 0093 D2). Split the "
                        f"{role} at that fold into separate {verb}() calls.")


# â”€â”€ per-node frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_normal(
    node: int,
    f_list: list[int],
    facet_normals: list[ndarray],
    label: str,
    verb: str,
    role: str,
) -> ndarray:
    """The node's outward unit normal â€” the 3D ``_node_normals``.

    The renormalised UNIFORM average of the adjacent facet unit normals
    (see the module docstring for why the weight is uniform), with the 2D
    lane's two refusals carried over verbatim: an average that cancels, and
    an average that opposes one of its own contributors.  Sharp reentrancy
    is already gone by here; these catch the rest of the doubled-back
    family, which many facets can reach without any single pair folding.
    """
    stack = [facet_normals[f] for f in f_list]
    avg = np.sum(stack, axis=0)
    norm = float(np.linalg.norm(avg))
    if norm <= _ZERO_TOL:
        raise ValueError(
            f"{verb}{label}: {role} node {node}'s adjacent facet normals "
            f"cancel out (their average is the zero vector) â€” the surface "
            f"doubles back on itself there, so the node has no single "
            f"outward direction (ADR 0093 D2).")
    avg = avg / norm
    bad = [
        (int(f), float(np.dot(avg, facet_normals[f])))
        for f in f_list
        if float(np.dot(avg, facet_normals[f])) <= 0.0
    ]
    if bad:
        raise ValueError(
            f"{verb}{label}: {role} node {node}'s averaged outward normal "
            f"{tuple(round(float(v), 6) for v in avg)} opposes the normal of "
            f"its own adjacent facet(s) {[f for f, _ in bad]} (dots "
            f"{[d for _, d in bad]}) â€” a doubled-back corner has no single "
            f"outward direction (ADR 0093 D2). Split the {role} at that node "
            f"into separate {verb}() calls.")
    return avg


def _tangent_pair(n: ndarray) -> ndarray:
    """``(2, 3)`` in-plane tangents completing the right-handed frame.

    ``t1`` is the global axis LEAST aligned with ``n``, projected into the
    plane and normalised; ties go to the lowest axis index, so the pick is
    a pure function of ``n`` and two runs of the same model emit the same
    ``-orient`` (ADR 0027).  ``t2 = n x t1``, which makes ``(n, t1, t2)``
    right-handed â€” ``n=x`` gives ``(x, y, z)``.

    The tangential yield cap is isotropic in the plane (``|tau| <= tau_b``
    on both directions), so which in-plane direction ``t1`` lands on is a
    determinism choice, not mechanics â€” exactly as the 2D lane's single
    tangent sign is.  What IS mechanics is that the triad is orthonormal
    and right-handed, since ``zeroLength -orient`` takes ``(x-axis,
    y-plane-vector)`` and infers the third.
    """
    axis = _AXES[int(np.argmin(np.abs(n)))]
    t1 = axis - float(np.dot(axis, n)) * n
    t1 = t1 / float(np.linalg.norm(t1))
    t2 = np.cross(n, t1)
    return np.array([t1, t2], dtype=float)
