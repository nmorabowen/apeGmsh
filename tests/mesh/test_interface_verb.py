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


# ── 3D fixture (TIMs A10 S2) ─────────────────────────────────────────

def _surface_at_z(volume: int, z: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False):
        bb = gmsh.model.getBoundingBox(2, abs(tag))
        if abs(bb[2] - z) < tol and abs(bb[5] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary surface of volume {volume} at z={z}")


def _build_two_boxes(g, n: int = 3):
    """Soil block [0,1]^2 x [0,1] under a footing block [0,1]^2 x [1,2].

    The 3D sibling of :func:`_build_two_squares`: transfinite hexes, the
    two volumes NEVER fragmented, so the shared plane at ``z=1`` carries
    two coincident-but-distinct node sets. ``"face"`` is the soil's top
    surface (the master — its material sits below, so outward is ``+z``)
    and ``"skin"`` the footing's underside.
    """
    soil = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    footing = g.model.geometry.add_box(0, 0, 1, 1, 1, 1)
    g.model.sync()
    g.mesh.structured.set_transfinite([(3, soil), (3, footing)], n=n)
    g.mesh.generation.generate(3)
    top = _surface_at_z(soil, 1.0)
    g.physical.add(3, [soil], name="soil")
    g.physical.add(3, [footing], name="footing")
    g.physical.add(2, [top], name="face")
    g.physical.add(2, [_surface_at_z(footing, 1.0)], name="skin")
    g.physical.add(
        1, [abs(t) for _, t in gmsh.model.getBoundary(
            [(2, top)], oriented=False)][:1], name="edge")
    return soil, footing


def _interface_fem_3d(**kw):
    with apeGmsh(model_name="iface_a10", verbose=False) as g:
        _build_two_boxes(g)
        g.constraints.interface(
            "face", "skin", normal=NORMAL, tangential=TANGENTIAL, **kw)
        return g.mesh.queries.get_fem_data(dim=3)


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


def test_thickness_is_refused_by_name_on_a_3d_model():
    """``thickness`` is the 2D line master's out-of-plane depth. A 3D
    surface master has a real area, so the kwarg is refused rather than
    quietly multiplied into ``A_trib`` (TIMs A10 S2)."""
    with apeGmsh(model_name="iface_3d_thk", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        with pytest.raises(ValueError, match="thickness"):
            g.constraints.interface(
                "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
                thickness=THICKNESS)
        assert g.constraints.interface_defs == []


@pytest.mark.parametrize("ndf,ok", [(None, True), (3, True), (4, True),
                                    (6, True), (2, False)])
def test_slave_ndf_set_is_the_3d_one_on_a_3d_model(ndf, ok):
    with apeGmsh(model_name="iface_3d_ndf", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        if ok:
            g.constraints.interface(
                "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
                slave_ndf=ndf)
            assert g.constraints.interface_defs[-1].thickness is None
        else:
            with pytest.raises(ValueError, match="slave_ndf"):
                g.constraints.interface(
                    "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
                    slave_ndf=ndf)


def test_surface_master_is_refused_at_resolve_in_a_2d_model():
    """A dim-2 master on a 2D model — the label resolves, but a 2D
    model's master is its dim-1 boundary curve (D2)."""
    with apeGmsh(model_name="iface_surf", verbose=False) as g:
        _build_two_squares(g)
        g.constraints.interface(
            "rock", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS)
        with pytest.raises(NotImplementedError, match="ADR 0093 D2"):
            g.mesh.queries.get_fem_data(dim=2)


def test_line_master_is_refused_at_resolve_in_a_3d_model():
    """The mirror: a dim-1 master in a 3D model. The surface lane needs
    facets, and a curve carries none — refused by name on the label."""
    with apeGmsh(model_name="iface_3d_line", verbose=False) as g:
        _build_two_boxes(g)
        g.constraints.interface(
            "edge", "skin", normal=NORMAL, tangential=TANGENTIAL)
        with pytest.raises(NotImplementedError, match="dimension \\[1\\]"):
            g.mesh.queries.get_fem_data(dim=3)


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


# =====================================================================
# 3D surface master, end to end (TIMs A10 S2)
# =====================================================================

def test_3d_surface_master_resolves_into_records():
    fem = _interface_fem_3d()
    recs = fem.elements.interfaces
    assert len(recs) == 9                     # n=3 nodes/edge => 3x3 grid
    assert [r.kind for r in recs] == ["interface"] * 9
    assert [r.slave_node for r in recs] == sorted(
        r.slave_node for r in recs)
    masters = {r.master_node for r in recs}
    slaves = {r.slave_node for r in recs}
    assert len(masters) == len(slaves) == 9
    assert masters.isdisjoint(slaves)


def test_3d_records_carry_a_nine_float_outward_frame():
    for r in _interface_fem_3d().elements.interfaces:
        assert len(r.orient) == 9
        n, t1, t2 = (np.asarray(r.orient[i:i + 3]) for i in (0, 3, 6))
        # The soil is below z=1, so INV-1's outward is +z on every pair,
        # corners and edges of the face included (they are averages).
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-12)
        for v in (n, t1, t2):
            assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-12)
        assert float(np.dot(n, t1)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.dot(n, t2)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.dot(t1, t2)) == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(np.cross(n, t1), t2, atol=1e-12)


def test_3d_tributary_areas_close_over_the_master_surface():
    recs = _interface_fem_3d().elements.interfaces
    assert sum(r.a_trib for r in recs) == pytest.approx(1.0, rel=1e-12)
    # 2x2 quads of area 0.25: the 4 face corners take one quarter-share
    # each, the 4 edge nodes two, the centre node four -- and no
    # thickness leg multiplies any of them (D3, the 3D reading).
    cell = 0.25
    shares = sorted(r.a_trib for r in recs)
    assert shares[0] == pytest.approx(0.25 * cell)
    assert shares[4] == pytest.approx(0.50 * cell)
    assert shares[-1] == pytest.approx(cell)


def test_3d_backing_elements_are_solid_domain_elements():
    fem = _interface_fem_3d()
    backing = {r.backing_element for r in fem.elements.interfaces}
    assert backing <= _element_ids(fem)
    assert all(t.dim == 3 for t in fem.elements.types)


@pytest.mark.parametrize("ndf", [None, 3, 4, 6])
def test_3d_pairs_connect_directly_at_every_accepted_slave_ndf(ndf):
    # ADR 96: the fork's zeroLength takes the mixed 3D pair itself, so
    # D4's phantom bridge has nothing left to bridge.
    for r in _interface_fem_3d(slave_ndf=ndf).elements.interfaces:
        assert r.phantom_node is None
        assert r.phantom_ndf is None
        assert r.equal_dof_records == []


def test_3d_quadratic_master_facets_are_refused_by_name():
    with apeGmsh(model_name="iface_3d_quad", verbose=False) as g:
        soil = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        footing = g.model.geometry.add_box(0, 0, 1, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(3, soil), (3, footing)], n=2)
        g.mesh.generation.generate(3)
        g.mesh.generation.set_order(2)
        g.physical.add(3, [soil], name="soil")
        g.physical.add(3, [footing], name="footing")
        g.physical.add(2, [_surface_at_z(soil, 1.0)], name="face")
        g.physical.add(2, [_surface_at_z(footing, 1.0)], name="skin")
        g.constraints.interface(
            "face", "skin", normal=NORMAL, tangential=TANGENTIAL)
        with pytest.raises(NotImplementedError, match="quad8|quad9|tri6"):
            g.mesh.queries.get_fem_data(dim=3)
