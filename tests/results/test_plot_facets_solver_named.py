"""``extract_facets`` keys on topology ``(dim, npe)``, not the type name.

Regression: a ``.ladruno`` / ``.mpco``-synthesized FEMData names its
element groups after the **OpenSees class** — ``make_type_info`` gets
``gmsh_name="SSPbrickUP"`` and no curated alias, so ``type_name`` comes
out ``"sspbrickup"``. The extractor used to allowlist ``{"tet4",
"hex8"}`` / ``{"tri3", "quad4"}``, so every solid and every shell in a
solver-read model was silently skipped and ``results.plot.*`` drew an
empty figure. Only the 1-D branch survived — it was already dim-based.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.results.plot._facets import extract_facets


# One unit hex, nodes 1..8 in the shared Gmsh / OpenSees brick ordering.
_HEX_COORDS = np.array([
    [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
    [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
])


def _one_element_fem(
    *, gmsh_name: str, code: int, dim: int, npe: int,
) -> FEMData:
    """A single element spanning the first ``npe`` of 8 unit-cube nodes."""
    node_ids = np.arange(1, 9, dtype=np.int64)
    info = make_type_info(
        code=code, gmsh_name=gmsh_name, dim=dim, order=1, npe=npe, count=1,
    )
    group = ElementGroup(
        element_type=info,
        ids=np.array([1], dtype=np.int64),
        connectivity=node_ids[:npe].reshape(1, npe),
    )
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(
            node_ids=node_ids, node_coords=_HEX_COORDS,
            physical=pg, labels=LabelSet({}),
        ),
        elements=ElementComposite(
            groups={info.code: group}, physical=pg, labels=LabelSet({}),
        ),
        info=MeshInfo(n_nodes=8, n_elems=1, bandwidth=0, types=[info]),
    )


# ``code = -class_tag`` mirrors what the .ladruno / .mpco readers
# synthesize (negated so it never collides with a Gmsh code).
@pytest.mark.parametrize("gmsh_name, code", [
    ("Hexahedron 8", 5),          # native Gmsh — alias 'hex8'
    ("SSPbrickUP", -75),          # solver-named — alias 'sspbrickup'
    ("stdBrick", -8),
    ("LadrunoBrick", -33010),
])
def test_solid_boundary_faces_survive_a_solver_name(
    gmsh_name: str, code: int,
) -> None:
    fem = _one_element_fem(gmsh_name=gmsh_name, code=code, dim=3, npe=8)
    tris, _ = extract_facets(fem)
    # 6 boundary faces, each split on the (0,1,2)+(0,2,3) diagonal.
    assert tris.shape == (12, 3)


@pytest.mark.parametrize("gmsh_name, code, npe, n_tris", [
    ("Quadrilateral 4", 3, 4, 2),     # native Gmsh — alias 'quad4'
    ("ShellMITC4", -53, 4, 2),        # solver-named — alias 'shellmitc4'
    ("Triangle 3", 2, 3, 1),
    ("Tri31", -33, 3, 1),
])
def test_surface_elements_survive_a_solver_name(
    gmsh_name: str, code: int, npe: int, n_tris: int,
) -> None:
    fem = _one_element_fem(gmsh_name=gmsh_name, code=code, dim=2, npe=npe)
    tris, _ = extract_facets(fem)
    assert tris.shape == (n_tris, 3)


def test_tet_faces_from_a_solver_named_group() -> None:
    fem = _one_element_fem(
        gmsh_name="FourNodeTetrahedron", code=-179, dim=3, npe=4,
    )
    tris, _ = extract_facets(fem)
    assert tris.shape == (4, 3)


def test_unsupported_volume_topology_is_still_skipped() -> None:
    """Wedges / pyramids have no face table — they drop out quietly
    rather than being read with tet or hex indices."""
    fem = _one_element_fem(gmsh_name="Prism 6", code=6, dim=3, npe=6)
    tris, segs = extract_facets(fem)
    assert tris.size == 0 and segs.size == 0


# ---------------------------------------------------------------------
# Higher order — rendered from the corner subset (mid-side nodes drop),
# matching the viewer's ``GMSH_LINEAR_FALLBACK``. Same "empty box"
# symptom, reached by npe rather than by name.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("gmsh_name, code, dim, npe, n_tris, n_corner", [
    ("Tetrahedron 10", 11,     3, 10, 4,  4),   # tet10 -> 4 corner faces
    ("BezierTet10",    -33001, 3, 10, 4,  4),
    ("Hexahedron 20",  17,     3, 20, 12, 8),   # hex20 -> 6 faces, split
    ("Hexahedron 27",  12,     3, 27, 12, 8),
    ("Triangle 6",     9,      2, 6,  1,  3),   # tri6  -> 1 corner tri
    ("BezierTri6",     -33000, 2, 6,  1,  3),
    ("Quadrilateral 8", 16,    2, 8,  2,  4),   # quad8 -> 2 corner tris
    ("Quadrilateral 9", 10,    2, 9,  2,  4),
])
def test_higher_order_renders_from_its_corners(
    gmsh_name: str, code: int, dim: int, npe: int,
    n_tris: int, n_corner: int,
) -> None:
    # Corner IDs 1..8 first, then filler mid-side IDs — the Gmsh
    # (and fork Bezier) convention the corner subset relies on.
    node_ids = np.arange(1, npe + 1, dtype=np.int64)
    coords = np.vstack([
        _HEX_COORDS, np.zeros((max(npe - 8, 0), 3)),
    ])[:npe]
    info = make_type_info(
        code=code, gmsh_name=gmsh_name, dim=dim, order=2, npe=npe, count=1,
    )
    group = ElementGroup(
        element_type=info,
        ids=np.array([1], dtype=np.int64),
        connectivity=node_ids.reshape(1, npe),
    )
    pg = PhysicalGroupSet({})
    fem = FEMData(
        nodes=NodeComposite(
            node_ids=node_ids, node_coords=coords,
            physical=pg, labels=LabelSet({}),
        ),
        elements=ElementComposite(
            groups={info.code: group}, physical=pg, labels=LabelSet({}),
        ),
        info=MeshInfo(n_nodes=npe, n_elems=1, bandwidth=0, types=[info]),
    )
    tris, _ = extract_facets(fem)
    assert tris.shape == (n_tris, 3)
    # Only corner IDs may appear — mid-side nodes must not be drawn.
    assert set(np.unique(tris)) <= set(range(1, n_corner + 1))
