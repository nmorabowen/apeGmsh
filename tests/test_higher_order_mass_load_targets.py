"""Masses and loads on higher-order meshes land in full, or fail loud.

The load and mass composites walk ``gmsh.model.mesh.getElements`` for a
target.  Their inline element-type tables omitted quad9 and hex27, so
on a mesh elevated with ``set_order(2)`` (``bubble=True`` is the
default, which gives quad9 / hex27) a surface mass, a tributary
pressure, a volume mass or a gravity load produced **zero** records
without a warning.  They also took a line3's mid node (gmsh lists the
two end nodes first) as its far end, which halved line masses and
tributary line loads.  The walks now consult the shared table in
``apeGmsh.mesh._element_types``; an element type they cannot handle
raises instead of being skipped.

Each test checks the element type the mesher produced, so a test
cannot pass while exercising a different type than its id names.
The companion guard is ``tests/test_element_topology.py``.
"""
from __future__ import annotations

import warnings

import gmsh
import numpy as np
import pytest


# =====================================================================
# Helpers
# =====================================================================

def _slab(g, *, quad: bool, order: int, bubble: bool = True) -> None:
    """2 x 1 rectangle labelled ``slab``, meshed at *order*."""
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 2.0, 1.0, label="slab")
    if quad:
        g.mesh.structured.set_transfinite("slab", n=3, recombine=True)
    g.mesh.generation.generate(2)
    if order > 1:
        g.mesh.generation.set_order(order, bubble=bubble)


def _block(g, *, hexa: bool, order: int, bubble: bool = True) -> None:
    """2 x 1 x 1 box labelled ``blk``, meshed at *order*."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 1.0, 1.0, label="blk")
    if hexa:
        g.mesh.structured.set_transfinite("blk", n=3, recombine=True)
    g.mesh.generation.generate(3)
    if order > 1:
        g.mesh.generation.set_order(order, bubble=bubble)


def _beam(g, *, order: int) -> None:
    """4-long line along x labelled ``beam``, meshed at *order*."""
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
    p1 = g.model.geometry.add_point(4.0, 0.0, 0.0)
    g.model.geometry.add_line(p0, p1, label="beam")
    g.mesh.structured.set_transfinite("beam", n=3)
    g.mesh.generation.generate(1)
    if order > 1:
        g.mesh.generation.set_order(order)


def _element_types(dim: int) -> set[int]:
    return {int(t) for t in gmsh.model.mesh.getElementTypes(dim)}


def _total_mass(fem) -> float:
    return sum(float(r.mass[0]) for r in fem.nodes.masses)


def _total_force(fem) -> np.ndarray:
    total = np.zeros(3)
    for r in fem.nodes.loads:
        if r.force_xyz is not None:
            total += np.asarray(r.force_xyz, dtype=float)
    return total


# (order, bubble, gmsh element type produced)
_QUADS = [
    pytest.param(1, True, 3, id="quad4"),
    pytest.param(2, False, 16, id="quad8"),
    pytest.param(2, True, 10, id="quad9"),
]
_HEXES = [
    pytest.param(1, True, 5, id="hex8"),
    pytest.param(2, False, 17, id="hex20"),
    pytest.param(2, True, 12, id="hex27"),
]
_LINES = [
    pytest.param(1, 1, id="line2"),
    pytest.param(2, 8, id="line3"),
]


# =====================================================================
# Surfaces — quad9 was dropped
# =====================================================================

@pytest.mark.parametrize("reduction", ["lumped", "consistent"])
@pytest.mark.parametrize("order, bubble, etype", _QUADS)
def test_surface_mass_is_conserved(g, order, bubble, etype, reduction):
    _slab(g, quad=True, order=order, bubble=bubble)
    assert _element_types(2) == {etype}
    g.masses.surface("slab", areal_density=100.0, reduction=reduction)

    fem = g.mesh.queries.get_fem_data()

    assert _total_mass(fem) == pytest.approx(100.0 * 2.0)


@pytest.mark.parametrize("reduction", ["tributary", "consistent"])
@pytest.mark.parametrize("order, bubble, etype", _QUADS)
def test_surface_pressure_resultant(g, order, bubble, etype, reduction):
    _slab(g, quad=True, order=order, bubble=bubble)
    assert _element_types(2) == {etype}
    with g.loads.case("p"):
        g.loads.surface.pressure("slab", magnitude=10.0,
                                 reduction=reduction)

    fem = g.mesh.queries.get_fem_data()

    # a free-standing face in the XY plane: gmsh's normal is +z, and a
    # positive pressure pushes against it
    np.testing.assert_allclose(_total_force(fem), [0.0, 0.0, -20.0],
                               atol=1e-9)


def test_pressure_on_hex27_solid_uses_the_physical_outward(g):
    """Faces of a hex27 solid are quad9, and the outward normal comes
    from the adjacent hex27 — which the adjacency walk must know."""
    _block(g, hexa=True, order=2)
    assert _element_types(3) == {12}
    bottom = next(
        int(t) for d, t in g.model.queries.boundary("blk", dim=2)
        if abs(g.model.queries.center_of_mass(int(t), dim=int(d))[2]) < 1e-9
    )
    g.physical.add_surface([bottom], name="Bottom")
    with g.loads.case("p"):
        g.loads.surface.pressure("Bottom", magnitude=10.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fem = g.mesh.queries.get_fem_data()

    assert not [w for w in caught
                if "unsupported 3-D element type" in str(w.message)]
    # outward of the z = 0 face is -z; +p pushes into the body (+z)
    np.testing.assert_allclose(_total_force(fem), [0.0, 0.0, 10.0 * 2.0],
                               atol=1e-9)


# =====================================================================
# Volumes — hex27 was dropped
# =====================================================================

@pytest.mark.parametrize("mass_reduction, load_reduction",
                         [("lumped", "tributary"),
                          ("consistent", "consistent")])
@pytest.mark.parametrize("order, bubble, etype", _HEXES)
def test_volume_mass_and_gravity_are_conserved(
        g, order, bubble, etype, mass_reduction, load_reduction):
    _block(g, hexa=True, order=order, bubble=bubble)
    assert _element_types(3) == {etype}
    g.masses.volume("blk", density=10.0, reduction=mass_reduction)
    with g.loads.case("grav"):
        g.loads.gravity("blk", g=(0.0, 0.0, -10.0), density=1.0,
                        reduction=load_reduction)

    fem = g.mesh.queries.get_fem_data()

    assert _total_mass(fem) == pytest.approx(10.0 * 2.0)
    np.testing.assert_allclose(_total_force(fem), [0.0, 0.0, -20.0],
                               atol=1e-9)


# =====================================================================
# Lines — a line3's mid node was taken for its far end
# =====================================================================

@pytest.mark.parametrize("reduction", ["lumped", "consistent"])
@pytest.mark.parametrize("order, etype", _LINES)
def test_line_mass_is_conserved(g, order, etype, reduction):
    _beam(g, order=order)
    assert _element_types(1) == {etype}
    g.masses.line("beam", linear_density=10.0, reduction=reduction)

    fem = g.mesh.queries.get_fem_data()

    assert _total_mass(fem) == pytest.approx(10.0 * 4.0)


@pytest.mark.parametrize("order, etype", _LINES)
def test_tributary_line_load_resultant(g, order, etype):
    _beam(g, order=order)
    assert _element_types(1) == {etype}
    with g.loads.case("q"):
        g.loads.line("beam", magnitude=5.0, direction="z")

    fem = g.mesh.queries.get_fem_data()

    np.testing.assert_allclose(_total_force(fem), [0.0, 0.0, 5.0 * 4.0],
                               atol=1e-9)


@pytest.mark.parametrize("order, etype", _LINES)
def test_tributary_normal_line_load_resultant(g, order, etype):
    _beam(g, order=order)
    assert _element_types(1) == {etype}
    with g.loads.case("q"):
        g.loads.line("beam", magnitude=5.0, normal=True,
                     away_from=(2.0, -1.0, 0.0))

    fem = g.mesh.queries.get_fem_data()

    # the in-plane normal points away from ``away_from`` (+y)
    np.testing.assert_allclose(_total_force(fem), [0.0, 5.0 * 4.0, 0.0],
                               atol=1e-9)


# =====================================================================
# Element types the walks cannot handle raise instead of vanishing
# =====================================================================

def test_consistent_pressure_refuses_tri9(g):
    """tri9 has 9 nodes, like quad9; integrating it as a quad9 would be
    silently wrong, so the consistent path refuses it."""
    _slab(g, quad=False, order=3, bubble=False)
    assert _element_types(2) == {20}
    with g.loads.case("p"):
        g.loads.surface.pressure("slab", magnitude=10.0,
                                 reduction="consistent")

    with pytest.raises(ValueError, match="tri9"):
        g.mesh.queries.get_fem_data()


def test_volume_mass_refuses_tet20(g):
    _block(g, hexa=False, order=3)
    assert _element_types(3) == {29}
    g.masses.volume("blk", density=10.0)

    with pytest.raises(ValueError, match="tet20"):
        g.mesh.queries.get_fem_data()


def test_unknown_face_type_raises(g):
    """quad12 (order-3 serendipity) is not in the table."""
    _slab(g, quad=True, order=3, bubble=False)
    assert _element_types(2) == {39}
    with g.loads.case("p"):
        g.loads.surface.pressure("slab", magnitude=10.0)

    with pytest.raises(ValueError, match="element type 39"):
        g.mesh.queries.get_fem_data()
