"""Consistent pressure pushes into the body, whatever the connectivity says.

``g.loads.surface.pressure(+P)`` promises ``-P`` times the outward normal.
The tributary reduction took that normal from the adjacent volume; the
consistent one took it from the face connectivity, which gmsh points *into*
the body on a cavity wall, so the pressure pulled.  The oracle for a closed
wall is the divergence theorem on the meshed (curved) surface:
``sum x . F = 3 P V_cavity``, exact for the isoparametric faces.
"""
from __future__ import annotations

import gmsh
import numpy as np
import pytest

from apeGmsh._kernel.defs.loads import SurfaceLoadDef
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver

P = 2.0


def _records(fem, pattern=None):
    return [r for r in fem.nodes.loads
            if r.force_xyz is not None
            and (pattern is None or r.pattern == pattern)]


def _total(records):
    return np.sum([r.force_xyz for r in records], axis=0)


def _cavity(g, order):
    """Cube of side 4 with a spherical hole of radius 1, wall named 'Wall'."""
    box = gmsh.model.occ.addBox(-2, -2, -2, 4, 4, 4)
    ball = gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.cut([(3, box)], [(3, ball)])
    gmsh.model.occ.synchronize()
    wall = [t for d, t in gmsh.model.getEntities(2)
            if gmsh.model.getType(d, t) == "Sphere"]
    gmsh.model.addPhysicalGroup(
        3, [t for _, t in gmsh.model.getEntities(3)], name="Body")
    gmsh.model.addPhysicalGroup(2, wall, name="Wall")
    g.mesh.sizing.set_global_size(0.9)
    g.mesh.generation.generate(dim=3)
    if order > 1:
        g.mesh.generation.set_order(order)


def _meshed_volume():
    """Volume of the isoparametric mesh, from gmsh's own Jacobians."""
    vol = 0.0
    etypes, _, _ = gmsh.model.mesh.getElements(3)
    for et in etypes:
        pts, w = gmsh.model.mesh.getIntegrationPoints(int(et), "Gauss6")
        _, dets, _ = gmsh.model.mesh.getJacobians(int(et), pts)
        vol += float(np.sum(np.reshape(dets, (-1, len(w))) @ np.asarray(w)))
    return vol


@pytest.mark.parametrize("order,basis", [
    (1, "lagrange"), (2, "lagrange"), (2, "bernstein")])
def test_cavity_wall_is_pushed_outward(g, order, basis):
    _cavity(g, order)
    cavity = 64.0 - _meshed_volume()
    with g.loads.case("cons"):
        g.loads.surface.pressure(
            "Wall", magnitude=P, reduction="consistent", basis=basis)
    with g.loads.case("trib"):
        g.loads.surface.pressure("Wall", magnitude=P)
    fem = g.mesh.queries.get_fem_data(dim=3)
    xyz = dict(zip((int(n) for n in fem.nodes.ids), np.asarray(fem.nodes.coords)))

    def work(pattern):
        return sum(float(np.dot(xyz[int(r.node_id)], r.force_xyz))
                   for r in _records(fem, pattern))

    # Pressure in the cavity pushes its wall away from the centre.
    assert work("trib") > 0.0
    if basis == "bernstein":
        # The nodes are control points here, so the volume is another
        # surface's; the direction is what is tested.
        assert work("cons") > 0.0
    else:
        np.testing.assert_allclose(work("cons"), 3 * P * cavity, rtol=1e-9)
    np.testing.assert_allclose(_total(_records(fem, "cons")), 0.0, atol=1e-9)


@pytest.mark.parametrize("order", [1, 2])
def test_each_cube_face_is_pushed_in(g, order):
    """A convex boundary, where the connectivity already pointed out."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    faces = {}
    for d, t in g.model.queries.boundary("cube", dim=2):
        com = np.asarray(g.model.queries.center_of_mass(int(t), dim=2))
        faces[f"F{t}"] = np.round(2 * (com - 0.5))
        g.physical.add_surface([int(t)], name=f"F{t}")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    if order > 1:
        g.mesh.generation.set_order(order)
    for name in faces:
        with g.loads.case(name):
            g.loads.surface.pressure(
                name, magnitude=P, reduction="consistent")
    fem = g.mesh.queries.get_fem_data(dim=3)
    for name, outward in faces.items():
        np.testing.assert_allclose(
            _total(_records(fem, name)), -P * outward, atol=1e-9)


def test_crack_faces_open_apart(g):
    """The two faces of a crack share a connectivity normal but not a body."""
    g.model.geometry.add_box(-50, -50, -50, 100, 100, 100, label="box")
    g.model.geometry.add_rectangle(-10, -10, 0, 20, 20, label="plane")
    g.model.boolean.fragment(objects="box", tools="plane", cleanup_free=False)
    g.physical.add_volume("box", name="Body")
    g.physical.add_surface("plane", name="Crack")
    g.mesh.sizing.set_global_size(20.0)
    g.mesh.generation.generate(dim=3)
    g.mesh.editing.crack("Crack", dim=2)
    for name in ("Crack_normal", "Crack_inverted"):
        for reduction in ("tributary", "consistent"):
            with g.loads.case(f"{name}-{reduction}"):
                g.loads.surface.pressure(
                    name, magnitude=P, reduction=reduction)
    fem = g.mesh.queries.get_fem_data(dim=3)
    for name, sign in (("Crack_normal", 1.0), ("Crack_inverted", -1.0)):
        for reduction in ("tributary", "consistent"):
            np.testing.assert_allclose(
                _total(_records(fem, f"{name}-{reduction}")),
                [0.0, 0.0, sign * P * 400.0], atol=1e-6)


def test_traction_and_shear_do_not_read_the_orientation(g):
    _cavity(g, 2)
    with g.loads.case("t"):
        g.loads.surface.traction(
            "Wall", (0.0, 0.0, -P), reduction="consistent")
    before = g.loads._face_outward_normals
    g.loads._face_outward_normals = lambda faces: pytest.fail("not needed")
    try:
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.loads._face_outward_normals = before
    assert _total(_records(fem, "t"))[2] < 0.0


def test_surface_model_keeps_the_connectivity_normal(g):
    """No volumes, so there is no body to push into: unchanged."""
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="plate")
    g.physical.add_surface("plate", name="Plate")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=2)
    with g.loads.case("c"):
        g.loads.surface.pressure("Plate", magnitude=P, reduction="consistent")
    fem = g.mesh.queries.get_fem_data(dim=2)
    np.testing.assert_allclose(
        _total(_records(fem, "c")), [0.0, 0.0, -P], atol=1e-12)


def test_resolver_flips_only_the_faces_that_disagree():
    tags = np.arange(1, 7)
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                    [5, 0, 0], [6, 0, 0], [5, 1, 0]], dtype=float)
    r = LoadResolver(tags, xyz)
    defn = SurfaceLoadDef(target="x", magnitude=P, mode="pressure",
                          reduction="consistent")
    faces = [[1, 2, 3], [4, 5, 6]]
    up = np.array([0.0, 0.0, 1.0])
    recs = {int(x.node_id): x.force_xyz[2] for x in
            r.resolve_surface_consistent(defn, faces, outwards=[up, -up])}
    np.testing.assert_allclose([recs[i] for i in (1, 2, 3)], -P / 6)
    np.testing.assert_allclose([recs[i] for i in (4, 5, 6)], P / 6)
