"""Guard: the shared element-topology table covers what the mesher makes.

Every ``gmsh.model.mesh.getElements`` walk in ``core`` consults
``apeGmsh.mesh._element_types.element_topology``.  Two inline copies of
this table once omitted quad9 (and hex27), and every mass and load on
those elements vanished without a warning.  These tests pin the table
to gmsh's own element properties and to the element types the mesher
actually produces, so a type the walks cannot resolve fails here rather
than in a user's model.  The end-to-end regressions are in
``tests/test_higher_order_mass_load_targets.py``.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh.fem._shape_functions import get_shape_functions
from apeGmsh.mesh._element_types import (
    TOPOLOGY_BY_CODE,
    _SHAPE_PREFIXES,
    _SHAPE_TOPOLOGY,
    element_topology,
)


def test_every_shape_has_a_topology():
    assert set(_SHAPE_TOPOLOGY) == set(_SHAPE_PREFIXES.values())


def test_table_agrees_with_gmsh(g):
    for code, topo in TOPOLOGY_BY_CODE.items():
        name, dim, _order, npe, _coords, n_primary = (
            gmsh.model.mesh.getElementProperties(code))
        assert (topo.dim, topo.npe, topo.n_corner) == (dim, npe, n_primary), (
            f"code {code} ({name})")


def test_integrable_types_are_told_apart_by_node_count():
    """The resolvers identify a full connectivity row by its node count,
    so within one dimension no two integrable types may share it."""
    for dim in (2, 3):
        widths = [t.npe for c, t in TOPOLOGY_BY_CODE.items()
                  if t.dim == dim and get_shape_functions(c) is not None]
        assert len(widths) == len(set(widths)), f"dim {dim}: {widths}"


# (shape, order, bubble, gmsh element type produced)
_MESHER_OUTPUT = [
    ("tri",  1, True,  2),    # tri3
    ("tri",  2, True,  9),    # tri6
    ("tri",  2, False, 9),    # tri6
    ("quad", 1, True,  3),    # quad4
    ("quad", 2, True,  10),   # quad9
    ("quad", 2, False, 16),   # quad8
    ("tet",  1, True,  4),    # tet4
    ("tet",  2, True,  11),   # tet10
    ("tet",  2, False, 11),   # tet10
    ("hex",  1, True,  5),    # hex8
    ("hex",  2, True,  12),   # hex27
    ("hex",  2, False, 17),   # hex20
]


@pytest.mark.parametrize(
    "shape, order, bubble, code", _MESHER_OUTPUT,
    ids=[f"{s}-order{o}-{'bubble' if b else 'nobubble'}"
         for s, o, b, _ in _MESHER_OUTPUT])
def test_mesher_output_is_resolvable(g, shape, order, bubble, code):
    """Every type ``set_order`` produces at orders 1 and 2 — the cells,
    their faces and their edges — is in the table, and the cells and
    faces have the shape functions the consistent / gravity / volume
    paths integrate with."""
    dim = 2 if shape in ("tri", "quad") else 3
    if dim == 2:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 2.0, 1.0, label="body")
    else:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 1.0, 1.0, label="body")
    if shape in ("quad", "hex"):
        g.mesh.structured.set_transfinite("body", n=3, recombine=True)
    g.mesh.generation.generate(dim)
    if order > 1:
        g.mesh.generation.set_order(order, bubble=bubble)

    assert {int(t) for t in gmsh.model.mesh.getElementTypes(dim)} == {code}
    for d in range(1, dim + 1):
        for etype in gmsh.model.mesh.getElementTypes(d):
            element_topology(etype, dim=d, integrable=d > 1)


def test_unknown_or_other_dim_type_raises():
    with pytest.raises(ValueError, match="element type 39"):
        element_topology(39, dim=2)     # quad12 is not in the table
    with pytest.raises(ValueError, match="not a known 2-D"):
        element_topology(5, dim=2)      # hex8 is 3-D


def test_integrable_refuses_a_type_without_shape_functions():
    assert element_topology(20, dim=2).n_corner == 3     # tri9 corners
    with pytest.raises(ValueError, match="tri9"):
        element_topology(20, dim=2, integrable=True)
