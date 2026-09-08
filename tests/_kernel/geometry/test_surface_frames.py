"""The 3D master surface's per-node frames — pure arrays, no gmsh, no session.

``surface_frames`` is the 3D sibling of ``edge_frames``: it turns a dim-2
master's facet connectivity into a per-node **outward** frame ``(n, t1,
t2)`` and a tributary area, which is what ``g.constraints.interface()``
needs before it can spring a footing onto a soil block (ADR 0093 D2/D3,
3D slice S1).

Three decisions are pinned here rather than assumed. The sign comes from
the adjacent solid's centroid, so a facet's **winding is irrelevant**
(``test_flipped_winding_still_points_outward``). The per-node average is
**uniform** over the adjacent facet unit normals, which is what makes a
cube corner ``(1,1,1)/sqrt(3)`` and a cube edge the normalised sum of two
face normals (``test_box_corner_*``). And a **reentrant** fold is refused
while a convex one is not, which ``dot(n_i, n_j)`` alone cannot tell apart
(``test_reentrant_fold_*``).

The harness mirrors ``tests/_kernel/geometry/test_boundary_chain.py``'s
``_Patch``: hand-placed nodes, hand-listed hexes, hand-listed facets.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.geometry._surface_frames import surface_frames

TOL = 1e-12


class _Solid:
    """A hand-built 3D block: nodes, solid elements, master facets."""

    def __init__(self) -> None:
        self.coords: dict[int, np.ndarray] = {}
        self.elem_tags: list[int] = []
        self.elem_nodes: list[list[int]] = []
        self.facets: list[list[int]] = []

    def node(self, tag: int, x: float, y: float, z: float) -> int:
        self.coords[int(tag)] = np.array([float(x), float(y), float(z)])
        return int(tag)

    def elem(self, tag: int, nodes) -> None:
        self.elem_tags.append(int(tag))
        self.elem_nodes.append([int(n) for n in nodes])

    def frames(self, facets=None, label="", verb="interface", role="master"):
        rows = facets if facets is not None else self.facets
        m_set = {int(t) for facet in rows for t in facet}
        return surface_frames(
            rows, m_set, self.coords, self.elem_tags, self.elem_nodes,
            label, verb=verb, role=role,
        )


def _flat_patch() -> _Solid:
    """A 2x2 quad4 patch on ``z = 0``, with a 2x2 block of hexes BELOW it.

    Top nodes ``1..9`` on a unit grid at ``z=0`` (tag ``1 + i + 3j``),
    bottom nodes ``11..19`` at ``z=-1``. The body is below, so every
    outward normal is ``+z`` and the patch area is 4.
    """
    s = _Solid()
    for j in range(3):
        for i in range(3):
            s.node(1 + i + 3 * j, float(i), float(j), 0.0)
            s.node(11 + i + 3 * j, float(i), float(j), -1.0)
    for j in range(2):
        for i in range(2):
            top = [1 + i + 3 * j, 2 + i + 3 * j,
                   5 + i + 3 * j, 4 + i + 3 * j]
            s.elem(100 + i + 2 * j, top + [t + 10 for t in top])
            s.facets.append(top)
    return s


def _unit_cube() -> _Solid:
    """One hex, with its ``+x`` / ``+y`` / ``+z`` faces as the master.

    Nodes ``1..4`` at ``z=0`` (ccw from the origin), ``5..8`` above them.
    The three faces meet at node 7, the ``(1,1,1)`` corner.
    """
    s = _Solid()
    for k, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        s.node(1 + k, float(x), float(y), 0.0)
        s.node(5 + k, float(x), float(y), 1.0)
    s.elem(100, [1, 2, 3, 4, 5, 6, 7, 8])
    s.facets = [[2, 3, 7, 6], [3, 4, 8, 7], [5, 6, 7, 8]]  # +x, +y, +z
    return s


def _reentrant_L() -> _Solid:
    """Three unit hexes in an L (the ``x>1, z>1`` cell is missing).

    Node ``1 + i + 3j + 6k`` sits at ``(i, j, k)`` for ``i, k`` in
    ``0..2`` and ``j`` in ``0..1``. The master is the two faces that meet
    at the reentrant edge ``x = 1, z = 1``: the top of the ``x>1`` cell
    (outward ``+z``) and the ``+x`` face of the ``z>1`` cell (outward
    ``+x``). Their interior dihedral is 270 degrees.
    """
    s = _Solid()
    for k in range(3):
        for j in range(2):
            for i in range(3):
                s.node(1 + i + 3 * j + 6 * k, float(i), float(j), float(k))

    def _hex(tag, i0, k0):
        s.elem(tag, [
            1 + di + 3 * dj + 6 * dk
            for dk in (k0, k0 + 1) for dj in (0, 1) for di in (i0, i0 + 1)
        ])

    _hex(100, 0, 0)   # x in [0,1], z in [0,1]
    _hex(101, 1, 0)   # x in [1,2], z in [0,1]
    _hex(102, 0, 1)   # x in [0,1], z in [1,2]
    s.facets = [[8, 9, 12, 11], [8, 14, 17, 11]]
    return s


def _assert_orthonormal_right_handed(data) -> None:
    for t, n in data.normals.items():
        t1, t2 = data.tangents[t]
        assert abs(float(np.linalg.norm(n)) - 1.0) < TOL
        assert abs(float(np.linalg.norm(t1)) - 1.0) < TOL
        assert abs(float(np.linalg.norm(t2)) - 1.0) < TOL
        assert abs(float(np.dot(n, t1))) < TOL
        assert abs(float(np.dot(n, t2))) < TOL
        assert abs(float(np.dot(t1, t2))) < TOL
        # (n, t1, t2) right-handed: det of the triad is +1, not -1.
        assert abs(float(np.linalg.det(np.array([n, t1, t2]))) - 1.0) < TOL


# =====================================================================
# The flat patch — outward sign, tributary shares, frame algebra
# =====================================================================

def test_flat_patch_normals_point_out_of_the_body_below():
    data = _flat_patch().frames()
    assert len(data.normals) == 9
    for n in data.normals.values():
        # +z, not -z: the block sits at z < 0, so out of it is UP.
        assert np.allclose(n, [0.0, 0.0, 1.0], atol=TOL)


def test_flat_patch_tributary_shares_are_quarter_half_one():
    data = _flat_patch().frames()
    # Every facet has unit area and four nodes, so a node's share is
    # 0.25 per incident facet: the interior node touches four (one whole
    # facet-area), an edge node two, a corner node one.
    assert data.a_trib[5] == pytest.approx(1.00, abs=TOL)   # (1,1) interior
    assert data.a_trib[2] == pytest.approx(0.50, abs=TOL)   # (1,0) edge
    assert data.a_trib[4] == pytest.approx(0.50, abs=TOL)   # (0,1) edge
    assert data.a_trib[1] == pytest.approx(0.25, abs=TOL)   # (0,0) corner
    assert data.a_trib[9] == pytest.approx(0.25, abs=TOL)   # (2,2) corner
    # And they tile the patch — the 3D reading of INV-3's closure.
    assert data.total_area == pytest.approx(4.0, abs=TOL)
    assert sum(data.a_trib.values()) == pytest.approx(4.0, abs=TOL)


def test_flat_patch_frames_are_orthonormal_and_right_handed():
    _assert_orthonormal_right_handed(_flat_patch().frames())


def test_tangents_are_deterministic_for_a_given_normal():
    a = _flat_patch().frames()
    b = _flat_patch().frames()
    for t in a.normals:
        assert np.array_equal(a.tangents[t], b.tangents[t])
    # n = +z picks the least-aligned global axis, x (ties to the lowest
    # index), so t1 = x and t2 = z x x = y.
    assert np.allclose(a.tangents[5][0], [1.0, 0.0, 0.0], atol=TOL)
    assert np.allclose(a.tangents[5][1], [0.0, 1.0, 0.0], atol=TOL)


# =====================================================================
# Winding is not a contract (the mutation check)
# =====================================================================

def test_flipped_winding_still_points_outward():
    """Reverse ONE facet's node order — nothing moves.

    The listed order only supplies a normal candidate; the sign is fixed
    against the owning hex's centroid, exactly as the 2D lane does. So a
    mesh whose facets are wound inconsistently is accepted and oriented
    correctly rather than refused — a mesh's winding is not a contract.
    """
    ref = _flat_patch().frames()
    s = _flat_patch()
    s.facets[0] = list(reversed(s.facets[0]))
    flipped = s.frames()
    for t, n in ref.normals.items():
        assert np.allclose(flipped.normals[t], n, atol=TOL)
        assert flipped.a_trib[t] == pytest.approx(ref.a_trib[t], abs=TOL)


# =====================================================================
# The box corner — averaged and renormalised across faces
# =====================================================================

def test_box_corner_normal_is_the_diagonal():
    data = _unit_cube().frames()
    root3 = 1.0 / np.sqrt(3.0)
    assert np.allclose(data.normals[7], [root3, root3, root3], atol=TOL)


def test_box_edge_normal_is_the_normalised_sum_of_two_faces():
    data = _unit_cube().frames()
    root2 = 1.0 / np.sqrt(2.0)
    assert np.allclose(data.normals[3], [root2, root2, 0.0], atol=TOL)  # +x,+y
    assert np.allclose(data.normals[6], [root2, 0.0, root2], atol=TOL)  # +x,+z
    assert np.allclose(data.normals[8], [0.0, root2, root2], atol=TOL)  # +y,+z
    # A node on one face only keeps that face's normal untouched.
    assert np.allclose(data.normals[2], [1.0, 0.0, 0.0], atol=TOL)


def test_box_corner_tributary_sums_across_faces():
    data = _unit_cube().frames()
    assert data.a_trib[7] == pytest.approx(0.75, abs=TOL)   # three faces
    assert data.a_trib[3] == pytest.approx(0.50, abs=TOL)   # two faces
    assert data.a_trib[2] == pytest.approx(0.25, abs=TOL)   # one face
    assert data.total_area == pytest.approx(3.0, abs=TOL)
    assert sum(data.a_trib.values()) == pytest.approx(3.0, abs=TOL)


def test_box_corner_frames_are_orthonormal_and_right_handed():
    _assert_orthonormal_right_handed(_unit_cube().frames())


# =====================================================================
# The reentrant fold — loud, and told apart from a convex corner
# =====================================================================

def test_reentrant_fold_is_refused_and_names_the_nodes():
    with pytest.raises(ValueError) as exc:
        _reentrant_L().frames()
    msg = str(exc.value)
    assert "REENTRANT" in msg
    assert "node 8" in msg                 # the lower fold-line node
    assert "(8, 9, 12, 11)" in msg         # both offending facets, named
    assert "(8, 14, 17, 11)" in msg
    assert "270" in msg                    # the interior dihedral, in degrees
    assert "ADR 0093 D2" in msg


def test_a_convex_ninety_degree_corner_is_not_a_fold():
    """The case a bare ``dot(n_i, n_j)`` test would refuse by mistake.

    A cube's corner and the L's reentrant corner both give ``dot == 0``.
    Only the centroid sense tells them apart, and the box must pass — it
    is the geometry the verb exists to spring.
    """
    data = _unit_cube().frames()
    assert len(data.normals) == 7


# =====================================================================
# Mixed facet types
# =====================================================================

def test_mixed_tri3_and_quad4_give_the_same_normals():
    s = _flat_patch()
    quad = s.facets.pop(0)                      # nodes 1, 2, 5, 4
    s.facets += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
    data = s.frames()
    assert len(data.normals) == 9
    for n in data.normals.values():
        assert np.allclose(n, [0.0, 0.0, 1.0], atol=TOL)
    # The area still closes; only the split inside that patch changes
    # (a tri gives each of its three nodes a third of half a unit).
    assert data.total_area == pytest.approx(4.0, abs=TOL)
    assert sum(data.a_trib.values()) == pytest.approx(4.0, abs=TOL)


def test_quadratic_facets_are_gated_by_name():
    s = _flat_patch()
    s.facets[0] = s.facets[0] + [101, 102, 103, 104]   # a quad8-sized row
    for t in (101, 102, 103, 104):
        s.node(t, 0.0, 0.0, 0.0)
    with pytest.raises(NotImplementedError, match="quad8"):
        s.frames()


# =====================================================================
# The rest of the refusal taxonomy, inherited from the 2D lane
# =====================================================================

def test_interior_facet_is_refused():
    """A facet with a hex on both sides has no outward direction."""
    s = _flat_patch()
    s.elem(200, [1, 2, 5, 4, 21, 22, 25, 24])
    for t, (x, y) in zip((21, 22, 25, 24), ((0, 0), (1, 0), (1, 1), (0, 1))):
        s.node(t, float(x), float(y), 1.0)
    with pytest.raises(ValueError, match="INTERIOR"):
        s.frames()


def test_unbacked_facet_is_refused():
    s = _flat_patch()
    for t, (x, y) in zip((31, 32, 33, 34), ((5, 5), (6, 5), (6, 6), (5, 6))):
        s.node(t, float(x), float(y), 0.0)
    s.facets.append([31, 32, 33, 34])
    with pytest.raises(ValueError, match="no adjacent 3D domain element"):
        s.frames()


def test_duplicated_facet_is_refused():
    s = _flat_patch()
    s.facets.append(list(s.facets[0]))
    with pytest.raises(ValueError, match="appears twice"):
        s.frames()


def test_an_empty_surface_is_refused():
    s = _flat_patch()
    with pytest.raises(ValueError, match="no boundary surface elements"):
        s.frames(facets=[])
