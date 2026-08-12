"""ADR 0093 S4 — ``g.constraints.interface()``: the verb and the
end-to-end factory path.

The geometry math is pinned array-by-array in
``tests/_kernel/resolvers/test_interface_resolver.py``; this file covers
the layer above it — argument validation, the 2D-only scope gate, and
the live-Gmsh gather that turns two coincident physical groups into
``fem.elements.interfaces``.

The fixture is two transfinite unit squares meeting at ``x=1`` and
*never fragmented*, so each side keeps its own nodes on the shared line:
a node-for-node coincident interface, which is exactly the topology the
verb pairs. Persistence still refuses loudly (the ADR 0093 S3 h5 guard
holds until S6), and that refusal is asserted here too — a resolver
that populated a slot the writer silently dropped would be worse than
no resolver.
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


# =====================================================================
# Fixtures
# =====================================================================

def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _build_two_squares(g, n: int = 4, ref_point: bool = False):
    """Left square [0,1]^2 (the continuum) + right square [1,2]^2.

    Both transfinite with the same edge division, un-fragmented, so the
    two curves at ``x=1`` carry coincident-but-distinct node sets.
    ``ref_point`` adds a free vertex off to the side (meshed as its own
    0D entity) for the ``node_to_surface`` half of the phantom-tag test.
    """
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    point = g.model.geometry.add_point(3.0, 0.5, 0.0, lc=1.0) if ref_point \
        else None
    g.model.sync()
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
    g.mesh.generation.generate(2)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
    return left, right, point


def _interface_fem(**kw):
    with apeGmsh(model_name="iface_s4", verbose=False) as g:
        _build_two_squares(g)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, **kw)
        return g.mesh.queries.get_fem_data(dim=2)


def _element_ids(fem) -> set[int]:
    return {int(t) for t in fem.elements.ids}


# =====================================================================
# Verb-level validation
# =====================================================================

def test_thickness_is_required():
    with apeGmsh(model_name="iface_thk", verbose=False) as g:
        with pytest.raises(TypeError):
            g.constraints.interface(
                "face", "wire", normal=NORMAL, tangential=TANGENTIAL)


@pytest.mark.parametrize("bad", [0.0, -1.0, None])
def test_non_positive_thickness_is_refused(bad):
    with apeGmsh(model_name="iface_thk2", verbose=False) as g:
        with pytest.raises(ValueError, match="thickness"):
            g.constraints.interface(
                "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
                thickness=bad)
        assert g.constraints.interface_defs == []


def test_laws_must_be_declarative_kernel_laws():
    with apeGmsh(model_name="iface_law", verbose=False) as g:
        with pytest.raises(ValueError, match="NormalLaw"):
            g.constraints.interface(
                "face", "wire", normal=1.0e6, tangential=TANGENTIAL,
                thickness=THICKNESS)
        with pytest.raises(ValueError, match="TangentialLaw"):
            g.constraints.interface(
                "face", "wire", normal=NORMAL, tangential=1.0e5,
                thickness=THICKNESS)


@pytest.mark.parametrize("bad", [1, 4, 6])
def test_unknown_slave_ndf_is_refused(bad):
    with apeGmsh(model_name="iface_ndf", verbose=False) as g:
        with pytest.raises(ValueError, match="slave_ndf"):
            g.constraints.interface(
                "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
                thickness=THICKNESS, slave_ndf=bad)


def test_three_dimensional_model_is_refused_at_the_verb():
    with apeGmsh(model_name="iface_3d", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        with pytest.raises(NotImplementedError, match="ADR 0093 D2"):
            g.constraints.interface(
                "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
                thickness=THICKNESS)


def test_surface_master_is_refused_at_resolve():
    """A dim-2 master on a 2D model — the label resolves, but a surface
    master needs per-facet frames (deferred, D2)."""
    with apeGmsh(model_name="iface_surf", verbose=False) as g:
        _build_two_squares(g)
        g.constraints.interface(
            "rock", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS)
        with pytest.raises(NotImplementedError, match="ADR 0093 D2"):
            g.mesh.queries.get_fem_data(dim=2)


# =====================================================================
# End-to-end through the factory
# =====================================================================

def test_records_land_on_fem_elements_interfaces():
    fem = _interface_fem()
    recs = fem.elements.interfaces
    assert len(recs) == 4          # n=4 transfinite ⇒ 4 nodes on the face
    assert [r.kind for r in recs] == ["interface"] * 4
    # Deterministic: ordered by ascending slave tag.
    assert [r.slave_node for r in recs] == sorted(r.slave_node for r in recs)
    masters = {r.master_node for r in recs}
    slaves = {r.slave_node for r in recs}
    assert len(masters) == len(slaves) == 4
    assert masters.isdisjoint(slaves)


def test_orientation_is_the_master_face_outward_normal():
    """The master is the LEFT square's right edge, so the continuum
    sits at smaller x and outward is ``+x`` (INV-1: local-x away from
    the master's own material, so separation elongates the spring)."""
    fem = _interface_fem()
    for r in fem.elements.interfaces:
        np.testing.assert_allclose(r.orient[:3], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(r.orient[3:], [0.0, 1.0, 0.0], atol=1e-12)


def test_tributary_areas_close_over_the_master_face():
    fem = _interface_fem()
    recs = fem.elements.interfaces
    total = sum(r.a_trib for r in recs)
    assert total == pytest.approx(1.0 * THICKNESS, rel=1e-12)
    # The two polyline endpoints get a half-share of one segment each.
    shares = sorted(r.a_trib for r in recs)
    assert shares[0] == pytest.approx(shares[1])
    assert shares[2] == pytest.approx(2.0 * shares[0])


def test_backing_elements_are_real_domain_elements():
    fem = _interface_fem()
    backing = {r.backing_element for r in fem.elements.interfaces}
    assert backing <= _element_ids(fem)
    # …and every one of them is a 2D continuum element, never a
    # boundary line element (INV-5).
    assert all(t.dim == 2 for t in fem.elements.types)


def test_laws_ride_the_records_unchanged():
    fem = _interface_fem()
    for r in fem.elements.interfaces:
        assert r.normal_law == NORMAL
        assert r.tangential_law == TANGENTIAL


def test_equal_ndf_pairs_mint_no_phantom():
    fem = _interface_fem()
    for r in fem.elements.interfaces:
        assert r.phantom_node is None
        assert r.equal_dof_records == []


def test_beam_slave_mints_a_phantom_bridge_per_pair():
    fem = _interface_fem(slave_ndf=3)
    recs = fem.elements.interfaces
    top = int(max(fem.nodes.ids))
    phantoms = [r.phantom_node for r in recs]
    assert len(set(phantoms)) == 4
    assert min(phantoms) > top          # above every real node tag
    for r in recs:
        assert r.phantom_ndf == 2
        eq, = r.equal_dof_records
        assert eq.master_node == r.slave_node
        assert eq.slave_node == r.phantom_node
        assert eq.dofs == [1, 2]


def test_re_extraction_mints_the_same_phantom_tags():
    """Two ``get_fem_data()`` calls on one session must agree — the
    phantom high-water mark is per-extraction, not cumulative
    (ADR 0027 tag determinism)."""
    with apeGmsh(model_name="iface_redo", verbose=False) as g:
        _build_two_squares(g)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, slave_ndf=3)
        first = g.mesh.queries.get_fem_data(dim=2)
        first_tags = [r.phantom_node for r in first.elements.interfaces]
        # ``dim=`` is a non-default signature, so this really re-extracts.
        second = g.mesh.queries.get_fem_data(dim=2)
        second_tags = [r.phantom_node for r in second.elements.interfaces]
    assert first_tags == second_tags


def test_resolved_interfaces_survive_a_save(tmp_path):
    """The S3 guard's replacement (ADR 0093 S6): a resolved interface
    now persists instead of refusing the save. This file only pins that
    the verb's output reaches disk and comes back — the field-exact
    contract lives in ``test_interface_h5_roundtrip.py``."""
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5

    fem = _interface_fem()
    assert fem.elements.interfaces
    path = tmp_path / "iface.h5"
    fem.to_h5(str(path))
    back = read_fem_h5(str(path))
    assert len(back.elements.interfaces) == len(fem.elements.interfaces)


# =====================================================================
# Phantom-tag coordination with the MP lane
# =====================================================================

def test_interface_and_node_to_surface_phantoms_never_collide():
    """Two independent phantom minters on one model.

    ``ConstraintResolver._next_phantom_tag`` (the MP lane, feeding
    ``node_to_surface``) and the interface resolver both start from
    ``max(node_tags) + 1``. The factory resolves the MP pass first and
    hands its high-water mark on, so the two ranges are disjoint by
    construction — this is the test that would catch a reordering.
    """
    with apeGmsh(model_name="iface_phantom", verbose=False) as g:
        _left, right, ref = _build_two_squares(g, ref_point=True)
        g.constraints.node_to_surface((0, ref), (2, right))
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, slave_ndf=3)
        fem = g.mesh.queries.get_fem_data(dim=2)

    mp_phantoms = {int(nid) for nid, _ in fem.nodes.constraints.phantom_nodes()}
    iface_phantoms = {
        int(r.phantom_node) for r in fem.elements.interfaces
    }
    assert mp_phantoms and iface_phantoms
    assert mp_phantoms.isdisjoint(iface_phantoms)
    # Both ranges also stay clear of the real node pool.
    real = {int(t) for t in fem.nodes.ids}
    assert (mp_phantoms | iface_phantoms).isdisjoint(real)
