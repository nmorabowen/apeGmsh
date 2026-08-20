"""The 2D contact master's wound chain — pure arrays, no gmsh, no session.

``chain_edges`` turns the (unordered, arbitrarily wound) boundary edges of
a 2D master into ONE head-to-tail chain, which is the fork's chained
stride-2 pair list::

    contactSurface <tag> -master 2  n0 n1  n1 n2  n2 n3

The unchained shorthand ``n0 n1 n2 n3`` is silently legal fork-side and
declares a HOLED surface that converges to a wrong answer, so the walk
exists to make that form unreachable. The winding and the refusal
taxonomy are pinned here, on bare arrays — the session-level path is
covered in ``tests/mesh/test_contact_2d.py``.

The harness mirrors ``tests/_kernel/resolvers/test_interface_resolver.py``'s
``_Model``: hand-placed nodes, hand-listed quads, hand-listed master
edges.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.geometry._boundary_chain import (
    chain_edges, domain_frames, edge_frames,
)


class _Patch:
    """A hand-built 2D patch: nodes, domain elements, master edges."""

    def __init__(self) -> None:
        self.coords: dict[int, np.ndarray] = {}
        self.elem_tags: list[int] = []
        self.elem_nodes: list[list[int]] = []
        self.edges: list[tuple[int, int]] = []

    def node(self, tag: int, x: float, y: float) -> int:
        self.coords[int(tag)] = np.array([float(x), float(y), 0.0])
        return int(tag)

    def elem(self, tag: int, nodes) -> None:
        self.elem_tags.append(int(tag))
        self.elem_nodes.append([int(n) for n in nodes])

    def chain(self, edges=None, verb="contact", label="", role="master",
              frames=None):
        rows = np.asarray(edges if edges is not None else self.edges, dtype=int)
        m_set = {int(t) for t in rows.ravel()}
        data = edge_frames(
            rows, m_set, self.coords, self.elem_tags, self.elem_nodes,
            label, verb=verb, role=role, frames=frames,
        )
        return chain_edges(data, self.coords, label, verb=verb, role=role)

    def frames(self, label="", verb="contact", role="master"):
        return domain_frames(self.elem_tags, self.elem_nodes, self.coords,
                             label, verb=verb, role=role)


def _strip(n_seg: int = 3) -> _Patch:
    """A row of ccw quads sitting ABOVE the master line ``y = 0``.

    Master nodes ``1..n`` on ``y=0`` (the free bottom boundary), top nodes
    ``11..`` closing each quad, so the master's outward is ``-y``.
    """
    p = _Patch()
    for i in range(n_seg + 1):
        p.node(1 + i, float(i), 0.0)
        p.node(11 + i, float(i), 1.0)
    for i in range(n_seg):
        p.elem(100 + i, [1 + i, 2 + i, 12 + i, 11 + i])
        p.edges.append((1 + i, 2 + i))
    return p


# =====================================================================
# The happy path: one chain, consistently wound
# =====================================================================

def test_three_segments_chain_head_to_tail():
    out = _strip(3).chain()
    assert out.shape == (3, 2)
    for k in range(out.shape[0] - 1):
        assert int(out[k, 1]) == int(out[k + 1, 0])
    # SIX tags for three segments once flattened — the whole point.
    #
    # Travel is -x, not +x: the strip's material sits ABOVE y=0, so the
    # master's outward is -y, and the slave has to be below. Winding so
    # the slave lies to the LEFT of travel therefore walks 4 -> 1. See
    # test_winding_puts_the_slave_on_the_left.
    assert out.reshape(-1).tolist() == [4, 3, 3, 2, 2, 1]


def test_chain_is_never_the_holed_shorthand():
    """The flat four-tag form ``n0 n1 n2 n3`` must be unreachable."""
    flat = _strip(3).chain().reshape(-1).tolist()
    assert len(flat) == 6                       # not 4
    assert len(set(flat)) == 4                  # each interior node twice
    assert flat[1] == flat[2] and flat[3] == flat[4]


def test_winding_is_consistent_with_the_stored_outward():
    """Every emitted segment satisfies ``dot(perp(t), n_out) == +1``.

    The dot of two unit vectors perpendicular to the same tangent is
    exactly +-1, so this is a sign read, not a tolerance test.  ``+1``
    and not ``-1``: the fork's ``sigma = +1`` normal is ``perp(t)`` and
    has to point AT the slave, which is where the master's outward
    normal points.  See :func:`test_winding_puts_the_slave_on_the_left`
    for the physical statement of the same rule.
    """
    p = _strip(3)
    rows = np.asarray(p.edges, dtype=int)
    data = edge_frames(rows, {int(t) for t in rows.ravel()}, p.coords,
                       p.elem_tags, p.elem_nodes, "", verb="contact")
    by_key = {frozenset((int(a), int(b))): data.normals[e]
              for e, (a, b) in enumerate(rows)}
    out = chain_edges(data, p.coords, "", verb="contact")
    for a, b in out:
        tan = p.coords[int(b)] - p.coords[int(a)]
        perp = np.array([-tan[1], tan[0]]) / np.linalg.norm(tan)
        n = by_key[frozenset((int(a), int(b)))]
        assert float(np.dot(perp, n)) == pytest.approx(1.0, abs=1e-12)


def test_winding_puts_the_slave_on_the_left():
    """The physical statement of the sign rule, on a case with a side.

    Master body is the unit square, so its material is at ``x < 1`` and
    a slave in contact with the ``x = 1`` face can only be at ``x > 1``.
    The fork reads ``perp(t) = (-t_y, t_x)`` of the emitted travel
    direction as its ``sigma = +1`` normal and expects the slave on that
    side.  Winding the chain the other way round still converges - it
    just resolves the contact against the master's interior - which is
    why this is asserted on a geometry where "the slave side" is a fact
    and not a convention.
    """
    xyz = {1: np.array([0., 0., 0.]), 2: np.array([1., 0., 0.]),
           3: np.array([1., 1., 0.]), 4: np.array([0., 1., 0.])}
    data = edge_frames([(2, 3)], {2, 3}, xyz, [100], [[1, 2, 3, 4]],
                       "", verb="contact")
    # edge_frames' own outward must point out of the square, +x.
    assert data.normals[0][0] == pytest.approx(1.0, abs=1e-12)

    (a, b), = chain_edges(data, xyz, "", verb="contact")
    tan = xyz[int(b)] - xyz[int(a)]
    perp = np.array([-tan[1], tan[0]]) / np.linalg.norm(tan)
    toward_slave = np.array([1.0, 0.0])
    assert float(np.dot(perp, toward_slave)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("order", [
    [(2, 3), (1, 2), (3, 4)],          # permuted
    [(2, 1), (3, 2), (4, 3)],          # every edge listed backwards
    [(3, 4), (2, 1), (2, 3)],          # both
])
def test_input_listing_order_and_winding_do_not_matter(order):
    p = _strip(3)
    assert p.chain(order).reshape(-1).tolist() == [4, 3, 3, 2, 2, 1]


def test_single_segment():
    out = _strip(1).chain()
    assert out.shape == (1, 2)
    assert out.reshape(-1).tolist() == [2, 1]


# =====================================================================
# Closed loops — legal, and deterministic
# =====================================================================

def _square_ring() -> _Patch:
    """A ring of four quads around a square hole: the hole's boundary is a
    CLOSED master loop (the fork wrap is explicitly legal)."""
    p = _Patch()
    #  inner square (1..4) at +-1, outer square (11..14) at +-2
    inner = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    outer = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    for i, (x, y) in enumerate(inner):
        p.node(1 + i, x, y)
    for i, (x, y) in enumerate(outer):
        p.node(11 + i, x, y)
    for i in range(4):
        j = (i + 1) % 4
        p.elem(100 + i, [1 + i, 11 + i, 11 + j, 1 + j])
        p.edges.append((1 + i, 1 + j))
    return p


def test_closed_loop_chains_and_wraps():
    out = _square_ring().chain()
    assert out.shape == (4, 2)
    for k in range(3):
        assert int(out[k, 1]) == int(out[k + 1, 0])
    assert int(out[-1, 1]) == int(out[0, 0])          # the wrap


def test_closed_loop_starts_at_the_lowest_node_tag():
    p = _square_ring()
    first = int(p.chain()[0, 0])
    assert first == min(int(t) for t in np.asarray(p.edges).ravel())
    # and it is deterministic under a permuted / re-wound listing
    shuffled = [(4, 1), (2, 3), (1, 2), (3, 4)]
    np.testing.assert_array_equal(p.chain(shuffled), p.chain())


# =====================================================================
# The new, named refusals
# =====================================================================

def test_branching_master_refused_by_name():
    """A T: three segments meet at one node."""
    p = _strip(2)                       # nodes 1-2-3 on y=0, quads above
    p.node(50, 1.0, -1.0)               # a spur hanging BELOW node 2
    p.node(51, 1.5, -1.0)
    p.node(52, 1.5, 0.0)
    p.elem(200, [2, 50, 51, 52])        # the spur's own backing quad
    p.edges.append((2, 50))
    with pytest.raises(ValueError, match="BRANCHES"):
        p.chain()


def test_disjoint_runs_refused_by_name():
    """Two separate stretches — the holed form the fork cannot refuse."""
    p = _Patch()
    for i in range(6):
        p.node(1 + i, float(i), 0.0)
        p.node(11 + i, float(i), 1.0)
    for i in (0, 1, 3, 4):
        p.elem(100 + i, [1 + i, 2 + i, 12 + i, 11 + i])
        p.edges.append((1 + i, 2 + i))
    with pytest.raises(ValueError, match="DISJOINT"):
        p.chain()


def test_disjoint_refusal_names_each_run_and_the_hole_hazard():
    p = _Patch()
    for i in range(6):
        p.node(1 + i, float(i), 0.0)
        p.node(11 + i, float(i), 1.0)
    for i in (0, 1, 3, 4):
        p.elem(100 + i, [1 + i, 2 + i, 12 + i, 11 + i])
        p.edges.append((1 + i, 2 + i))
    with pytest.raises(ValueError) as exc:
        p.chain()
    msg = str(exc.value)
    assert "2 DISJOINT runs" in msg
    assert "HOLED" in msg
    assert "one contact() per connected stretch" in msg


def test_directed_clash_refused_by_name():
    """Two segments meeting at a node with the material on OPPOSITE sides.

    Same physical situation the interface lane's cancelling / reentrant
    node-normal refusals detect (ADR 0093 D2).
    """
    p = _Patch()
    p.node(1, 0.0, 0.0)
    p.node(2, 1.0, 0.0)
    p.node(3, 2.0, 0.0)
    p.node(11, 0.0, 1.0)
    p.node(12, 1.0, 1.0)                # quad ABOVE the left segment
    p.node(22, 1.0, -1.0)
    p.node(23, 2.0, -1.0)               # quad BELOW the right segment
    p.elem(100, [1, 2, 12, 11])
    p.elem(101, [2, 3, 23, 22])
    p.edges += [(1, 2), (2, 3)]
    with pytest.raises(ValueError, match="OPPOSITE sides"):
        p.chain()


def test_directed_clash_message_cross_references_the_interface_lane():
    p = _Patch()
    p.node(1, 0.0, 0.0)
    p.node(2, 1.0, 0.0)
    p.node(3, 2.0, 0.0)
    p.node(11, 0.0, 1.0)
    p.node(12, 1.0, 1.0)
    p.node(22, 1.0, -1.0)
    p.node(23, 2.0, -1.0)
    p.elem(100, [1, 2, 12, 11])
    p.elem(101, [2, 3, 23, 22])
    p.edges += [(1, 2), (2, 3)]
    with pytest.raises(ValueError) as exc:
        p.chain()
    assert "reentrant" in str(exc.value)


# =====================================================================
# Inherited refusals — same rules, contact's own verb
# =====================================================================

def test_inherited_refusals_report_the_calling_verb():
    p = _strip(2)
    with pytest.raises(ValueError, match=r"^contact: master boundary edge"):
        p.chain(p.edges + [(1, 2)])          # duplicated edge


def test_duplicate_edge_consequence_is_lane_specific():
    """The duplicate-edge body is verb-neutral; only the tail differs."""
    from apeGmsh._kernel.geometry._boundary_chain import (
        _duplicate_edge_consequence,
    )
    assert "tributary" in _duplicate_edge_consequence("interface")
    assert "contact kernel twice" in _duplicate_edge_consequence("contact")


def test_interior_edge_refusal_reads_correctly_for_contact():
    """An edge with material on BOTH sides is not a contact surface."""
    p = _Patch()
    p.node(1, 0.0, 0.0)
    p.node(2, 1.0, 0.0)
    p.node(11, 0.0, 1.0)
    p.node(12, 1.0, 1.0)
    p.node(21, 0.0, -1.0)
    p.node(22, 1.0, -1.0)
    p.elem(100, [1, 2, 12, 11])
    p.elem(101, [1, 2, 22, 21])
    p.edges.append((1, 2))
    with pytest.raises(ValueError, match="INTERIOR"):
        p.chain()


# =====================================================================
# The wrong-side guard — the check declared winding switches off
# =====================================================================
#
# ``-outward winding`` bypasses the fork's interface-level centroid vote,
# and that vote is the only thing that today catches a master named on the
# side of the body FACING AWAY from the slave. Bypassed, such a deck
# converges on the wrong boundary. So apeGmsh owns the check — and owns it
# deliberately WEAKER than the vote, because the two cases the vote
# refuses wrongly (flush interfaces, curved/closed masters) are exactly
# the two winding exists to unlock.


def _guard(patch, slaves, label="", verb="contact"):
    """Run the guard on ``patch``'s chain against hand-placed slaves."""
    from apeGmsh._kernel.geometry._boundary_chain import (
        refuse_wrong_side_master,
    )
    xyz = dict(patch.coords)
    tags = []
    for i, (x, y) in enumerate(slaves):
        t = 900 + i
        xyz[t] = np.array([float(x), float(y), 0.0])
        tags.append(t)
    return refuse_wrong_side_master(
        patch.chain(), xyz, tags, label, verb=verb)


def test_wrong_side_master_is_refused_by_name():
    """The far side of the block: every segment's nearest slave sits
    behind it, well past the fork's own narrow-phase reach."""
    p = _strip(3)                       # material ABOVE y=0 ⇒ outward -y
    with pytest.raises(ValueError) as exc:
        _guard(p, [(0.5, 3.0), (1.5, 3.0), (2.5, 3.0)])
    msg = str(exc.value)
    assert "FACE AWAY from the slave" in msg
    assert "3 of the master's 3 segments" in msg
    assert "BYPASSES the fork's own centroid vote" in msg


def test_wrong_side_guard_is_silent_on_a_flush_interface():
    """dot ~ 0, not negative. Flush is the case winding exists to unlock —
    a guard that fired here would refuse the masonry joint, the footing on
    soil, and every other zero-gap 2D deck."""
    p = _strip(3)
    coincident = [(float(i), 0.0) for i in range(4)]      # ON the master
    assert _guard(p, coincident) is None


def test_wrong_side_guard_is_silent_on_a_curved_closed_master():
    """A ring master, oriented per segment against its OWN nearest slave —
    never against a global centroid. The ring is precisely the shape the
    fork's vote cannot orient (master and slave centroids coincide at the
    hole's centre, and no single direction agrees with every segment), so
    a guard that reasoned globally would refuse the very case winding was
    added for."""
    p = _square_ring()                  # outward points INTO the hole
    inside = [(0.4, 0.4), (-0.4, 0.4), (-0.4, -0.4), (0.4, -0.4)]
    assert _guard(p, inside) is None
    # and the global datum really is degenerate here — both centroids are
    # the origin, which is what makes this the winding-only case
    ring_nodes = sorted({int(t) for t in np.asarray(p.edges).ravel()})
    m_c = np.mean([p.coords[t][:2] for t in ring_nodes], axis=0)
    np.testing.assert_allclose(m_c, np.mean(inside, axis=0), atol=1e-12)


def test_wrong_side_guard_needs_a_STRICT_majority():
    """A minority of opposed segments is not a refusal.

    Two of six segments here have a far-above slave as their nearest
    neighbour; the rest see the slaves below. The guard stays silent — it
    is a wrong-side detector, not a geometry critic."""
    p = _strip(6)
    assert _guard(p, [(0.5, 2.5), (4.5, -0.1), (5.5, -0.1)]) is None


def test_wrong_side_guard_tolerates_a_seeded_initial_penetration():
    """A slave pushed slightly INTO the master is a legitimate deck (the
    fork's own 'just-penetrated start'), and its dots are negative. The
    reach margin is what keeps the guard off it."""
    p = _strip(3)
    seeded = [(0.5, 0.1), (1.5, 0.1), (2.5, 0.1)]   # 0.1 into 1.0 segments
    assert _guard(p, seeded) is None


def test_wrong_side_guard_is_a_no_op_without_slaves():
    assert _guard(_strip(3), []) is None


# =====================================================================
# role= — the 2D mortar lane walks the SLAVE side through the same code
# =====================================================================
#
# `-slave-segments 2` reuses this walk, so every refusal has to name the
# surface the user actually mis-declared. A message saying "master" on a
# slave chain points the reader at the wrong curve, and a green suite that
# only pins ONE of the twenty substituted strings would not notice a
# partial revert.


def _slave_msg(build) -> str:
    """The rendered refusal of *build* under ``role="slave"``."""
    with pytest.raises((ValueError, NotImplementedError)) as exc:
        build()
    return str(exc.value)


def test_edge_frames_slave_refusals_never_say_master():
    """Every ``edge_frames`` refusal reachable from a bare patch."""
    # no line elements at all
    empty = _strip(2)
    assert "the slave label carries no boundary line elements" in _slave_msg(
        lambda: empty.chain(edges=np.empty((0, 2), dtype=int), role="slave"))

    # a duplicated edge
    dup = _strip(2)
    msg = _slave_msg(lambda: dup.chain(dup.edges + [(1, 2)], role="slave"))
    assert "slave boundary edge" in msg
    assert "Deduplicate the slave entities" in msg

    # an INTERIOR edge (material on both sides)
    interior = _Patch()
    interior.node(1, 0.0, 0.0)
    interior.node(2, 1.0, 0.0)
    interior.node(11, 0.0, 1.0)
    interior.node(12, 1.0, 1.0)
    interior.node(21, 0.0, -1.0)
    interior.node(22, 1.0, -1.0)
    interior.elem(100, [1, 2, 12, 11])
    interior.elem(101, [1, 2, 22, 21])
    interior.edges.append((1, 2))
    msg = _slave_msg(lambda: interior.chain(role="slave"))
    assert "slave boundary edge" in msg
    assert "The slave must be a free boundary" in msg

    # an edge with NO backing element
    unbacked = _strip(2)
    unbacked.node(90, 5.0, 0.0)
    unbacked.node(91, 6.0, 0.0)
    msg = _slave_msg(
        lambda: unbacked.chain(unbacked.edges + [(90, 91)], role="slave"))
    assert "slave boundary edge" in msg
    assert "is the slave a boundary curve of the" in msg

    for m in (_slave_msg(lambda: empty.chain(
                  edges=np.empty((0, 2), dtype=int), role="slave")),):
        assert "master" not in m


def test_chain_edges_slave_refusals_never_say_master():
    """The three ``chain_edges`` refusals, which nothing pinned before."""
    # BRANCHING
    branch = _strip(2)
    branch.node(50, 1.0, -1.0)
    branch.node(51, 1.5, -1.0)
    branch.node(52, 1.5, 0.0)
    branch.elem(200, [2, 50, 51, 52])
    branch.edges.append((2, 50))
    msg = _slave_msg(lambda: branch.chain(role="slave"))
    assert "slave node(s)" in msg and "the slave BRANCHES there" in msg
    assert "master" not in msg

    # DISJOINT runs
    disjoint = _Patch()
    for i in range(6):
        disjoint.node(1 + i, float(i), 0.0)
        disjoint.node(11 + i, float(i), 1.0)
    for i in (0, 1, 3, 4):
        disjoint.elem(100 + i, [1 + i, 2 + i, 12 + i, 11 + i])
        disjoint.edges.append((1 + i, 2 + i))
    msg = _slave_msg(lambda: disjoint.chain(role="slave"))
    assert "the slave's 4 boundary segments" in msg
    assert "declare a HOLED slave" in msg
    assert "master" not in msg

    # the DIRECTED CLASH (two segments with the continuum on opposite sides)
    clash = _Patch()
    clash.node(1, 0.0, 0.0)
    clash.node(2, 1.0, 0.0)
    clash.node(3, 2.0, 0.0)
    clash.node(11, 0.0, 1.0)
    clash.node(12, 1.0, 1.0)
    clash.node(21, 1.0, -1.0)
    clash.node(22, 2.0, -1.0)
    clash.elem(100, [1, 2, 12, 11])       # material ABOVE segment (1,2)
    clash.elem(101, [2, 3, 22, 21])       # material BELOW segment (2,3)
    clash.edges.extend([(1, 2), (2, 3)])
    msg = _slave_msg(lambda: clash.chain(role="slave"))
    assert "slave node 2 is the" in msg
    assert "Split the slave at that node" in msg
    assert "master" not in msg


def test_role_defaults_to_master_so_the_interface_lane_is_untouched():
    """The default keeps the ADR 0093 text byte-identical — the same
    reason ``verb`` has a default."""
    p = _strip(2)
    with pytest.raises(ValueError, match=r"^interface: master boundary edge"):
        p.chain(p.edges + [(1, 2)], verb="interface")


# =====================================================================
# DomainFrames — the whole-domain scratch, shared across surfaces
# =====================================================================

def test_shared_frames_give_the_same_chain_as_a_per_call_build():
    """Passing the cache must not change a single row: the 2D mortar lane
    builds it once and walks two surfaces with it."""
    p = _strip(4)
    solo = p.chain()
    shared = p.chain(frames=p.frames())
    np.testing.assert_array_equal(solo, shared)


def test_shared_frames_still_refuse_an_unbacked_edge():
    """The narrowed adjacency must keep a key for every surface node, so a
    node no element touches still draws the named refusal rather than a
    KeyError."""
    p = _strip(2)
    p.node(90, 5.0, 0.0)
    p.node(91, 6.0, 0.0)
    with pytest.raises(ValueError, match="no adjacent 2D domain element"):
        p.chain(p.edges + [(90, 91)], frames=p.frames())


def test_domain_frames_refusals_match_the_inline_build():
    """The extraction is code motion: same checks, same messages."""
    p = _Patch()
    p.node(1, 0.0, 0.0)
    with pytest.raises(ValueError, match="no 2D domain elements were"):
        domain_frames([], [], p.coords, " 'x'", verb="contact")
    with pytest.raises(ValueError, match="disagree in length"):
        domain_frames([1, 2], [[1]], p.coords, " 'x'", verb="contact")
    with pytest.raises(ValueError, match="not in the model node pool"):
        domain_frames([1], [[1, 999]], p.coords, " 'x'", verb="contact")
