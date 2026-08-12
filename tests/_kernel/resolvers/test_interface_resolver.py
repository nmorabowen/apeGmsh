"""ADR 0093 S4 — the interface resolver's geometry math.

Everything here runs on bare arrays: no Gmsh, no OpenSees, no session.
The composite gathers the node pools / master edges / domain elements
from the live model; this module owns pairing, the per-pair **outward**
frame (D2 / INV-1), tributary closure (D3 / INV-3), backing-element
stamping (INV-5) and the mixed-ndf phantom bridge (D4) — so each rule
gets a fixture that isolates it.

The load-bearing case is ``test_curved_master_*``: a coarse quarter-arc
where every node's normal must point radially outward. A single
face-average frame is off by 45° at the arc's ends, which is exactly
the wall-to-crown swing ADR 0093 exists to follow.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh._kernel.resolvers._interface_resolver import (
    resolve_interface_records,
)

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e6)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e5, tau_b=0.25)
THICKNESS = 2.0


# =====================================================================
# Fixture builders
# =====================================================================

class _Model:
    """A hand-built 2D patch: nodes, domain elements, master edges."""

    def __init__(self) -> None:
        self.coords: dict[int, tuple[float, float, float]] = {}
        self.elem_tags: list[int] = []
        self.elem_nodes: list[list[int]] = []
        self.master: list[int] = []
        self.slave: list[int] = []
        self.edges: list[tuple[int, int]] = []

    def node(self, tag: int, x: float, y: float) -> int:
        self.coords[int(tag)] = (float(x), float(y), 0.0)
        return int(tag)

    def elem(self, tag: int, nodes) -> None:
        self.elem_tags.append(int(tag))
        self.elem_nodes.append([int(n) for n in nodes])

    def resolve(self, **kw):
        tags = sorted(self.coords)
        coords = np.array([self.coords[t] for t in tags], dtype=float)
        kw.setdefault("normal_law", NORMAL)
        kw.setdefault("tangential_law", TANGENTIAL)
        kw.setdefault("thickness", THICKNESS)
        return resolve_interface_records(
            np.array(tags, dtype=int), coords,
            master_nodes=self.master, slave_nodes=self.slave,
            master_edges=np.array(self.edges, dtype=int),
            domain_elem_tags=self.elem_tags,
            domain_elem_nodes=self.elem_nodes,
            **kw,
        )


def _strip(xs=(0.0, 1.0, 2.0, 3.0), tops=None) -> _Model:
    """A row of quads above the master line y=0.

    Master nodes ``1..n`` sit on ``y=0`` (the free bottom boundary);
    top nodes ``11..`` close each quad at ``tops[i]``; slave nodes
    ``21..`` duplicate the master line. Outward is therefore ``-y``.
    """
    m = _Model()
    n = len(xs)
    tops = tops if tops is not None else [1.0] * n
    for i, x in enumerate(xs):
        m.master.append(m.node(1 + i, x, 0.0))
        m.node(11 + i, x, tops[i])
        m.slave.append(m.node(21 + i, x, 0.0))
    for i in range(n - 1):
        # ccw quad: bottom-left, bottom-right, top-right, top-left
        m.elem(100 + i, [1 + i, 2 + i, 12 + i, 11 + i])
        m.edges.append((1 + i, 2 + i))
    return m


def _arc(n_seg: int = 8, r_out: float = 1.0, r_in: float = 0.6) -> _Model:
    """A coarse quarter-annulus; the master is the OUTER arc.

    Material lies inside (``r_in..r_out``), so every master node's
    outward normal must point radially outward — the INV-1 curved case.
    """
    m = _Model()
    ang = np.linspace(0.0, 0.5 * np.pi, n_seg + 1)
    for i, a in enumerate(ang):
        m.master.append(
            m.node(1 + i, r_out * np.cos(a), r_out * np.sin(a)))
        m.node(101 + i, r_in * np.cos(a), r_in * np.sin(a))
        m.slave.append(
            m.node(201 + i, r_out * np.cos(a), r_out * np.sin(a)))
    for i in range(n_seg):
        m.elem(1000 + i, [1 + i, 2 + i, 102 + i, 101 + i])
        m.edges.append((1 + i, 2 + i))
    return m


def _radial(model: _Model, tag: int) -> np.ndarray:
    v = np.array(model.coords[tag][:2], dtype=float)
    return v / np.linalg.norm(v)


# =====================================================================
# Pairing
# =====================================================================

def test_pairs_are_one_per_slave_and_ordered_by_slave_tag():
    m = _strip()
    recs, _ = m.resolve()
    assert len(recs) == 4
    assert [r.slave_node for r in recs] == [21, 22, 23, 24]
    assert [r.master_node for r in recs] == [1, 2, 3, 4]
    assert all(r.kind == ConstraintKind.INTERFACE for r in recs)


def test_pair_order_is_deterministic_across_calls():
    m = _strip()
    a, _ = m.resolve()
    b, _ = m.resolve()
    assert [(r.master_node, r.slave_node) for r in a] == \
           [(r.master_node, r.slave_node) for r in b]


def test_unmatched_slave_node_is_refused():
    m = _strip()
    # A slave hanging 0.5 off the master line — no master within tol.
    m.slave.append(m.node(31, 1.0, 0.5))
    with pytest.raises(ValueError, match=r"no master node within tolerance"):
        m.resolve()


def test_many_to_one_pairing_is_refused():
    m = _strip()
    m.slave.append(m.node(31, 1.0 + 1e-12, 0.0))  # 2nd slave on master 2
    with pytest.raises(ValueError, match="ambiguous"):
        m.resolve()


def test_master_and_slave_sets_must_be_disjoint():
    m = _strip()
    m.slave.append(m.master[0])
    with pytest.raises(ValueError, match="share"):
        m.resolve()


# =====================================================================
# Orientation (D2 / INV-1)
# =====================================================================

def test_straight_master_orient_is_outward_and_right_handed():
    m = _strip()
    recs, _ = m.resolve()
    for r in recs:
        x = np.array(r.orient[:3])
        yp = np.array(r.orient[3:])
        # Material is above the line, so outward is -y.
        np.testing.assert_allclose(x, [0.0, -1.0, 0.0], atol=1e-12)
        # y_p = z_hat x x_hat — the deterministic right-handed choice.
        np.testing.assert_allclose(yp, np.cross([0, 0, 1], x), atol=1e-12)


def test_outward_sign_follows_the_material_side_not_the_edge_winding():
    """Reversing every master edge's node order must not flip outward.

    The sign comes from the adjacent element's centroid, never from the
    mesh's winding — a flipped normal is the silent tension-only bug
    ADR 0093 INV-1 exists to kill.
    """
    m = _strip()
    forward, _ = m.resolve()
    m.edges = [(b, a) for a, b in m.edges]
    reversed_, _ = m.resolve()
    for f, r in zip(forward, reversed_):
        assert f.orient == r.orient


def test_curved_master_normals_point_radially_outward():
    m = _arc()
    recs, _ = m.resolve()
    assert len(recs) == 9
    for r in recs:
        n = np.array(r.orient[:2])
        assert float(np.dot(n, _radial(m, r.master_node))) > 0.99


def test_curved_master_defeats_a_single_face_average_frame():
    """The same arc, measured against the frame a face-average would
    give. The per-node normals span the arc's 90° less half a chord at
    each end (8 chords ⇒ 78.75°), so the average frame is ~39.4° off at
    the ends — nowhere near the ``> 0.99`` radial gate the per-pair
    frames pass. A single-frame implementation cannot pass both.
    """
    m = _arc()
    recs, _ = m.resolve()
    normals = np.array([r.orient[:2] for r in recs])
    avg = normals.mean(axis=0)
    avg /= np.linalg.norm(avg)

    worst = np.degrees(np.arccos(np.clip(normals @ avg, -1.0, 1.0))).max()
    assert worst >= 39.0, f"face-average frame is only {worst:.1f} deg off"

    # …and it flunks the very gate the per-pair frames clear.
    radial_dots = [float(np.dot(avg, _radial(m, r.master_node))) for r in recs]
    assert min(radial_dots) < 0.8


def test_polyline_endpoints_use_their_single_edge_normal():
    m = _arc(n_seg=2)
    recs, _ = m.resolve()
    by_master = {r.master_node: r for r in recs}
    end = by_master[1]                       # angle 0, one adjacent edge
    edge_mid_angle = 0.25 * np.pi / 2        # midpoint of the first chord
    expect = np.array([np.cos(edge_mid_angle), np.sin(edge_mid_angle)])
    np.testing.assert_allclose(
        np.array(end.orient[:2]), expect, atol=1e-12)


def test_reentrant_corner_is_refused():
    """Three edges meet at the master node with the material on
    opposite sides — the averaged normal opposes one contributor, so
    the node has no single outward direction."""
    m = _Model()
    hub = m.node(1, 0.0, 0.0)
    right = m.node(2, 1.0, 0.0)
    left = m.node(3, -1.0, 0.0)
    down = m.node(4, 1.0, -0.1)          # near-horizontal, material BELOW
    m.node(10, 0.5, 1.0)
    m.node(11, -0.5, 1.0)
    m.node(12, 0.5, -1.0)
    m.master = [hub, right, left, down]
    for i, t in enumerate(m.master):
        x, y, _ = m.coords[t]
        m.slave.append(m.node(50 + i, x, y))
    m.elem(100, [hub, right, 10])         # above → outward -y
    m.elem(101, [left, hub, 11])          # above → outward -y
    m.elem(102, [hub, down, 12])          # below → outward +y-ish
    m.edges = [(hub, right), (hub, left), (hub, down)]
    with pytest.raises(ValueError, match="reentrant"):
        m.resolve()


def test_doubled_back_boundary_is_refused():
    """Two collinear edges with the material on opposite sides: the
    normals cancel exactly, so there is no average to normalise."""
    m = _Model()
    hub = m.node(1, 0.0, 0.0)
    right = m.node(2, 1.0, 0.0)
    left = m.node(3, -1.0, 0.0)
    m.node(10, 0.5, 1.0)                  # element above the right edge
    m.node(11, -0.5, -1.0)                # element below the left edge
    m.master = [hub, right, left]
    for i, t in enumerate(m.master):
        x, y, _ = m.coords[t]
        m.slave.append(m.node(50 + i, x, y))
    m.elem(100, [hub, right, 10])
    m.elem(101, [left, hub, 11])
    m.edges = [(hub, right), (hub, left)]
    with pytest.raises(ValueError, match="cancel out"):
        m.resolve()


def test_interior_master_edge_is_refused():
    """An edge with material on both sides has no outward direction."""
    m = _strip()
    # Mirror the strip below y=0 so edge (1,2) is shared by two domains.
    m.node(41, 0.0, -1.0)
    m.node(42, 1.0, -1.0)
    m.elem(200, [1, 41, 42, 2])
    with pytest.raises(ValueError, match="INTERIOR edge"):
        m.resolve()


def test_master_edge_with_no_domain_element_is_refused():
    m = _strip()
    m.master.append(m.node(5, 4.0, 0.0))
    m.slave.append(m.node(25, 4.0, 0.0))
    m.edges.append((4, 5))                # no quad backs this stretch
    with pytest.raises(ValueError, match="no adjacent 2D domain element"):
        m.resolve()


def test_duplicated_master_edge_is_refused():
    m = _strip()
    m.edges.append((2, 1))
    with pytest.raises(ValueError, match="appears twice"):
        m.resolve()


# =====================================================================
# Tributary (D3 / INV-3)
# =====================================================================

def test_tributary_closure_and_endpoint_half_shares():
    m = _strip(xs=(0.0, 1.0, 2.0, 3.0))
    recs, _ = m.resolve()
    by_master = {r.master_node: r.a_trib for r in recs}
    # Endpoints get one half-share, interior nodes two.
    assert by_master[1] == pytest.approx(0.5 * THICKNESS)
    assert by_master[4] == pytest.approx(0.5 * THICKNESS)
    assert by_master[2] == pytest.approx(1.0 * THICKNESS)
    assert by_master[3] == pytest.approx(1.0 * THICKNESS)
    # INV-3: the shares tile the whole master face.
    assert sum(by_master.values()) == pytest.approx(3.0 * THICKNESS, rel=1e-12)


def test_tributary_closure_holds_on_uneven_spacing():
    m = _strip(xs=(0.0, 0.25, 1.75, 3.0))
    recs, _ = m.resolve()
    total = sum(r.a_trib for r in recs)
    assert total == pytest.approx(3.0 * THICKNESS, rel=1e-12)


def test_unpaired_master_boundary_node_is_refused():
    """A slave covering only part of the master face would leave the
    rest silently unsprung — INV-3 refuses the partial tiling."""
    m = _strip()
    m.slave.pop()                          # drop the slave of master 4
    del m.coords[24]
    with pytest.raises(ValueError, match="do not tile the master face"):
        m.resolve()


def test_paired_node_with_zero_tributary_share_is_refused():
    m = _strip()
    m.master.append(m.node(9, 10.0, 10.0))   # on no boundary edge
    m.slave.append(m.node(29, 10.0, 10.0))
    with pytest.raises(ValueError, match="tributary length is zero"):
        m.resolve()


def test_thickness_scales_every_tributary_area():
    thin, _ = _strip().resolve(thickness=1.0)
    thick, _ = _strip().resolve(thickness=3.0)
    for a, b in zip(thin, thick):
        assert b.a_trib == pytest.approx(3.0 * a.a_trib)


# =====================================================================
# Backing element (INV-5 / settled Q3)
# =====================================================================

def test_backing_element_is_the_one_the_normal_points_away_from():
    # Element 101 is taller, so its centroid sits further behind the
    # (0,-1) outward normal at master node 2 than element 100's.
    m = _strip(tops=[1.0, 1.0, 2.0, 2.0])
    recs, _ = m.resolve()
    by_master = {r.master_node: r.backing_element for r in recs}
    assert by_master[2] == 101
    assert by_master[1] == 100          # only one incident element


def test_backing_element_ties_break_to_the_lowest_tag():
    # A uniform strip: at master node 2 both incident quads have their
    # centroid the same distance behind the normal — an exact tie.
    m = _strip()
    recs, _ = m.resolve()
    by_master = {r.master_node: r.backing_element for r in recs}
    assert by_master[2] == 100
    assert by_master[3] == 101


def test_backing_elements_are_all_domain_elements():
    m = _arc()
    recs, _ = m.resolve()
    assert {r.backing_element for r in recs} <= set(m.elem_tags)


# =====================================================================
# Mixed ndf phantom bridge (D4)
# =====================================================================

def test_slave_ndf_none_makes_no_phantom():
    recs, next_tag = _strip().resolve(phantom_tag_start=500)
    assert next_tag == 500
    for r in recs:
        assert r.phantom_node is None
        assert r.phantom_coords is None
        assert r.phantom_ndf is None
        assert r.equal_dof_records == []


def test_slave_ndf_two_is_the_explicit_form_of_none():
    a, _ = _strip().resolve(slave_ndf=None)
    b, _ = _strip().resolve(slave_ndf=2)
    assert [r.phantom_node for r in b] == [r.phantom_node for r in a] == \
           [None] * 4


def test_slave_ndf_three_mints_one_phantom_bridge_per_pair():
    m = _strip()
    recs, next_tag = m.resolve(slave_ndf=3, phantom_tag_start=500)
    assert [r.phantom_node for r in recs] == [500, 501, 502, 503]
    assert next_tag == 504
    for r in recs:
        # The record keeps the REAL slave; the phantom is a separate
        # field (INV-1: S5 emits iNode=master continuum, jNode=phantom
        # when present, else the real slave).
        assert r.slave_node in (21, 22, 23, 24)
        assert r.phantom_ndf == 2
        np.testing.assert_allclose(
            r.phantom_coords, m.coords[r.slave_node], atol=0)
        assert len(r.equal_dof_records) == 1
        eq = r.equal_dof_records[0]
        assert eq.kind == ConstraintKind.EQUAL_DOF
        assert eq.master_node == r.slave_node     # retained: the beam node
        assert eq.slave_node == r.phantom_node    # constrained: the phantom
        assert eq.dofs == [1, 2]


def test_phantom_tags_default_above_the_model_node_pool():
    m = _strip()
    recs, _ = m.resolve(slave_ndf=3)
    assert min(r.phantom_node for r in recs) > max(m.coords)


@pytest.mark.parametrize("bad", [0, 1, 4, 6, "3"])
def test_unknown_slave_ndf_is_refused(bad):
    with pytest.raises(ValueError, match="slave_ndf"):
        _strip().resolve(slave_ndf=bad)


# =====================================================================
# Scope gates
# =====================================================================

def test_three_dimensional_model_is_refused():
    with pytest.raises(NotImplementedError, match="ADR 0093 D2"):
        _strip().resolve(ndm=3)


def test_out_of_plane_master_edge_is_refused():
    m = _strip()
    m.coords[2] = (1.0, 0.0, 0.5)     # lift master 2 …
    m.coords[22] = (1.0, 0.0, 0.5)    # … and its slave, so pairing still holds
    with pytest.raises(NotImplementedError, match="out of the z=const plane"):
        m.resolve()


def test_non_positive_thickness_is_refused():
    with pytest.raises(ValueError, match="thickness"):
        _strip().resolve(thickness=0.0)


def test_laws_are_carried_verbatim_onto_every_record():
    recs, _ = _strip().resolve()
    for r in recs:
        assert r.normal_law is NORMAL
        assert r.tangential_law is TANGENTIAL
