"""A distributed load on a mesh selection refuses by name.

A selection stores node ids, not element connectivity, so a line, surface or
body load had nothing to integrate and resolved to zero records without a
word.  A selection also shadows a physical group of the same name, so the
zero could reach a load that named a perfectly good group.  Point loads need
only nodes and still work.
"""
from __future__ import annotations

import gmsh
import numpy as np
import pytest

EDGE, FACE, BODY = "Edge", "Top", "Body"


@pytest.fixture
def cube(g):
    """Meshed unit cube: PGs Edge / Top / Body, selections Sel1 / Sel2 / Sel3."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name=BODY)
    top = [t for d, t in gmsh.model.getEntities(2)
           if abs(gmsh.model.occ.getCenterOfMass(d, t)[2] - 1) < 1e-6]
    g.physical.add_surface(top, name=FACE)
    edge = [t for d, t in gmsh.model.getEntities(1)
            if np.allclose(gmsh.model.occ.getCenterOfMass(d, t), (0.5, 0, 1))]
    g.physical.add_curve(edge, name=EDGE)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    for dim, name in ((1, EDGE), (2, FACE), (3, BODY)):
        g.mesh_selection.from_physical(dim, name, ms_name=f"Sel{dim}")
    return g


DISTRIBUTED = {
    "line": lambda L, t, **k: L.line(t, magnitude=1.0, **k),
    "line-varying": lambda L, t, **k: L.line(
        t, magnitude=lambda p: 1.0 + p[0], **k),
    "line-normal": lambda L, t, **k: L.line(
        t, magnitude=1.0, normal=True, away_from=(0.5, 0.5, 1.0), **k),
    "pressure": lambda L, t, **k: L.surface.pressure(t, magnitude=1.0, **k),
    "traction": lambda L, t, **k: L.surface.traction(t, (0, 0, -1.0), **k),
    "shear": lambda L, t, **k: L.surface.shear(t, (1.0, 0, 0), **k),
    "gravity": lambda L, t, **k: L.gravity(t, density=1.0, **k),
    "volume": lambda L, t, **k: L.volume(t, force_per_volume=(0, 0, -1.0), **k),
}
NODAL = [{"reduction": "tributary"}, {"reduction": "consistent"}]
ELEMENT = [{"target_form": "element"}]
CASES = [
    (verb, "Sel1" if verb.startswith("line") else
     "Sel2" if verb in ("pressure", "traction", "shear") else "Sel3", kw)
    for verb in DISTRIBUTED
    for kw in NODAL + ([] if verb in ("line-normal", "shear") else ELEMENT)
]


@pytest.mark.parametrize(
    "verb,sel,kw", CASES,
    ids=[f"{v}-{next(iter(k.values()))}" for v, _, k in CASES])
def test_distributed_load_refuses_the_selection(cube, verb, sel, kw):
    with cube.loads.case("c"):
        DISTRIBUTED[verb](cube.loads, sel, **kw)
    with pytest.raises(ValueError, match=f"'{sel}' is a mesh selection"):
        cube.mesh.queries.get_fem_data(dim=3)


def test_a_selection_shadowing_a_group_refuses_and_pg_cures_it(cube):
    cube.mesh_selection.from_physical(3, BODY)  # named 'Body' by default
    with cube.loads.case("c"):
        cube.loads.gravity(BODY, density=1.0)
    with pytest.raises(ValueError, match="pg="):
        cube.mesh.queries.get_fem_data(dim=3)
    cube.loads.load_defs.clear()
    with cube.loads.case("c"):
        cube.loads.gravity(pg=BODY, density=1.0, g=(0, 0, -10.0))
    fem = cube.mesh.queries.get_fem_data(dim=3)
    total = np.sum([r.force_xyz for r in fem.nodes.loads], axis=0)
    np.testing.assert_allclose(total, (0, 0, -10.0), atol=1e-9)


def test_face_normal_forms_refuse_too(cube):
    """Already an error, but one that blamed missing face elements."""
    with cube.loads.case("c"):
        cube.loads.surface.force_resultant_center_mass(
            "Sel2", magnitude=1.0, normal=True)
    with pytest.raises(ValueError, match="'Sel2' is a mesh selection"):
        cube.mesh.queries.get_fem_data(dim=3)
    cube.loads.load_defs.clear()
    with cube.displacements.case("d"):
        cube.displacements.surface("Sel2", magnitude=1.0, normal=True)
    with pytest.raises(ValueError, match="'Sel2' is a mesh selection"):
        cube.mesh.queries.get_fem_data(dim=3)


def test_node_only_loads_still_take_a_selection(cube):
    (n_face,) = [len(s["node_ids"]) for s in cube.mesh_selection._sets.values()
                 if s["name"] == "Sel2"]
    with cube.loads.case("p"):
        cube.loads.point.force("Sel2", (0, 0, -1.0))
    with cube.loads.case("f"):
        cube.loads.surface.force_resultant_center_mass(
            "Sel2", force=(0, 0, -6.0))
    fem = cube.mesh.queries.get_fem_data(dim=3)
    by = {p: np.sum([r.force_xyz for r in fem.nodes.loads if r.pattern == p],
                    axis=0) for p in ("p", "f")}
    np.testing.assert_allclose(by["p"], (0, 0, -float(n_face)), atol=1e-9)
    np.testing.assert_allclose(by["f"], (0, 0, -6.0), atol=1e-9)
