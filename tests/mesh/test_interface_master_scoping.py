"""ADR 0093 D2/D3 — how the interface lane reads its MASTER face.

Two widenings of ``_collect_master_edges`` / ``_collect_node_set``, both
driven by real models and both invisible to the rest of the interface
battery (which only ever meshes a linear master out of a PG):

1. **A ``line3`` master.** A quadratic continuum puts a mid-side node on
   the boundary curve. Dropping to corners would leave every mid-side
   station unsprung against a coincident quadratic slave, so the element
   is expanded into its two half-segments and the mid becomes a real
   polyline station with its own tributary and normal (D3's ``0.5 *
   edge_length`` accumulation, unchanged).

2. **A master scoped to an ``addElements``-only discrete host.** Naming a
   sub-stretch of a longer curve with ``master_entities=`` must not
   re-``addNodes`` those tags — Gmsh keeps the tag and the deck then
   emits ``node`` twice. The host therefore carries elements only,
   ``getNodes`` is empty on it, and the node set is the unique
   connectivity of its line elements.

Both cases must close INV-3 exactly: no exemption, no relaxed tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e6)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e5, tau_b=0.25)
THICKNESS = 0.3
N = 4  # transfinite division ⇒ 3 segments on the shared face


# =====================================================================
# Fixtures
# =====================================================================

def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _build_two_squares(g, *, order: int = 1):
    """Two un-fragmented unit squares meeting at ``x=1``.

    Same topology as ``test_interface_verb``: each side keeps its own
    nodes on the shared line, so the interface is node-for-node
    coincident. ``order=2`` elevates both, which is what puts a
    ``line3`` on the master curve.
    """
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=N)
    g.mesh.generation.generate(2)
    if order > 1:
        g.mesh.generation.set_order(order)
    face = _curve_at_x(left, 1.0)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [face], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
    return left, right, face


def _face_chain(curve: int) -> list[int]:
    """The master curve's node tags, ordered bottom-to-top by y."""
    tags, coords, _ = gmsh.model.mesh.getNodes(
        1, int(curve), includeBoundary=True, returnParametricCoord=False)
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    order = np.argsort(xyz[:, 1])
    return [int(tags[i]) for i in order]


def _discrete_line2_chain(node_tags: list[int]) -> int:
    """Host an existing node chain on a new discrete curve, ELEMENTS ONLY.

    Deliberately no ``addNodes``: the tags stay classified on their
    original entity, so naming this host in ``master_entities=`` does not
    duplicate them in the emitted deck. ``getNodes`` is then empty on the
    host — which is precisely the case the collector has to survive.
    """
    disc = int(gmsh.model.addDiscreteEntity(1))
    high = int(gmsh.model.mesh.getMaxElementTag())
    etags = list(range(high + 1, high + len(node_tags)))
    conn: list[int] = []
    for a, b in zip(node_tags[:-1], node_tags[1:]):
        conn.extend([int(a), int(b)])
    gmsh.model.mesh.addElements(1, disc, [1], [etags], [conn])
    return disc


# =====================================================================
# 1 — a line3 master expands into half-segments
# =====================================================================

def test_quadratic_master_springs_its_mid_side_nodes():
    """Order 2 ⇒ 3 ``line3`` on the face ⇒ 7 stations, not 4.

    The mid-sides are paired too. If ``line3`` were dropped to its
    corners they would be unsprung and INV-3 would refuse the model
    (the mids are master boundary nodes the slave did reach).
    """
    with apeGmsh(model_name="iface_line3", verbose=False) as g:
        _build_two_squares(g, order=2)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS)
        fem = g.mesh.queries.get_fem_data(dim=2)

    recs = fem.elements.interfaces
    assert len(recs) == 2 * (N - 1) + 1        # 4 vertices + 3 mid-sides
    assert [r.slave_node for r in recs] == sorted(r.slave_node for r in recs)
    assert {r.master_node for r in recs}.isdisjoint(
        {r.slave_node for r in recs})


def test_quadratic_master_tributary_closes_over_the_whole_face():
    """INV-3 with the half-segment expansion: 6 half-segments of 1/6,
    so the two polyline ends carry 1/12 and the five interior stations
    (3 mids + 2 vertices) carry 1/6 each. Sums to the full face."""
    with apeGmsh(model_name="iface_line3_trib", verbose=False) as g:
        _build_two_squares(g, order=2)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS)
        fem = g.mesh.queries.get_fem_data(dim=2)

    shares = sorted(r.a_trib for r in fem.elements.interfaces)
    assert sum(shares) == pytest.approx(1.0 * THICKNESS, rel=1e-12)
    half = (1.0 / (2.0 * (N - 1))) * THICKNESS
    assert shares[0] == pytest.approx(0.5 * half)
    assert shares[1] == pytest.approx(0.5 * half)
    for s in shares[2:]:
        assert s == pytest.approx(half)


def test_quadratic_master_orientation_is_still_the_outward_normal():
    """A straight master's two half-segments share one normal, so the
    mid's averaged frame must come out exactly ``+x`` like the rest
    (D2) — no drift from the averaging."""
    with apeGmsh(model_name="iface_line3_orient", verbose=False) as g:
        _build_two_squares(g, order=2)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS)
        fem = g.mesh.queries.get_fem_data(dim=2)

    for r in fem.elements.interfaces:
        np.testing.assert_allclose(r.orient[:3], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(r.orient[3:], [0.0, 1.0, 0.0], atol=1e-12)


def test_cubic_master_is_still_refused_by_name():
    """The widening is line2 + line3 only. A line4 master still raises,
    and the refusal names what IS implemented."""
    with apeGmsh(model_name="iface_line4", verbose=False) as g:
        _build_two_squares(g, order=3)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS)
        with pytest.raises(NotImplementedError, match="line2 and"):
            g.mesh.queries.get_fem_data(dim=2)


# =====================================================================
# 2 — a master scoped to an addElements-only discrete host
# =====================================================================

def test_master_entities_may_be_an_add_elements_only_host():
    """The RevA arch lane: the master is a discrete curve carrying
    elements over tags classified on another entity.

    ``getNodes`` is empty there, so the collector has to fall back to
    the line connectivity. The resolved masters must be exactly the face
    chain, and INV-3 must close over the full face — no exemption.
    """
    with apeGmsh(model_name="iface_disc", verbose=False) as g:
        _left, _right, face = _build_two_squares(g)
        chain = _face_chain(face)
        disc = _discrete_line2_chain(chain)

        # The premise of the fallback: the host carries no classified
        # nodes of its own. If Gmsh ever changes that, this test is
        # measuring nothing and should fail here rather than pass.
        host_nodes, _, _ = gmsh.model.mesh.getNodes(
            1, disc, includeBoundary=True, returnParametricCoord=False)
        assert len(host_nodes) == 0

        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, master_entities=[(1, disc)])
        fem = g.mesh.queries.get_fem_data(dim=2)

    recs = fem.elements.interfaces
    assert len(recs) == N
    assert {r.master_node for r in recs} == set(chain)
    assert sum(r.a_trib for r in recs) == pytest.approx(
        1.0 * THICKNESS, rel=1e-12)


def test_add_elements_only_host_does_not_duplicate_nodes():
    """Why the host may not ``addNodes``: the tags must stay single in
    the model node pool, or the emitted deck declares them twice."""
    with apeGmsh(model_name="iface_disc_pool", verbose=False) as g:
        _left, _right, face = _build_two_squares(g)
        chain = _face_chain(face)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS,
            master_entities=[(1, _discrete_line2_chain(chain))])
        fem = g.mesh.queries.get_fem_data(dim=2)

    ids = [int(t) for t in fem.nodes.ids]
    assert len(ids) == len(set(ids))
    assert set(chain) <= set(ids)


def test_an_unmeshed_master_host_is_still_refused():
    """The fallback must not turn a genuinely empty entity into a silent
    pass: no classified nodes AND no elements is still a refusal."""
    with apeGmsh(model_name="iface_disc_empty", verbose=False) as g:
        _build_two_squares(g)
        empty = int(gmsh.model.addDiscreteEntity(1))
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, master_entities=[(1, empty)])
        with pytest.raises(ValueError, match="no mesh nodes"):
            g.mesh.queries.get_fem_data(dim=2)
