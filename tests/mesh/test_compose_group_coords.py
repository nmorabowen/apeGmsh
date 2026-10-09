"""Compose places the coordinates cached on groups, not only the node table (#1592).

Each physical group, label and mesh selection caches its own
``node_coords``. ``compose(translate=, rotate=)`` moved ``nodes.coords``
but copied those caches from the source unplaced, so
``fem.physical.node_coords(...)`` and ``fem.labels.node_coords(...)``
returned the module's source geometry while ``nodes.coords`` returned the
placed one.

Oracle: the closed form ``R x + t``. A rotation of 90 degrees about +z maps
``(x, y, z)`` to ``(-y, x, z)``; the translation is then added. The source's
own group coordinates are the ``x``; the placed node table must agree with
the cache row for row. The Assembly v2 case (instance with ``translate=``,
then ``bridge``) runs through the same merge engine.

A node-to-surface constraint caches its phantom nodes' ``phantom_coords``,
which the bridge declares the phantoms at; they must land on the placed
slave nodes they stand on.
"""
from __future__ import annotations

import math
from pathlib import Path

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData

SIDE = 10.0
T = (100.0, -50.0, 7.0)
ROT_Z_90 = (0.0, 0.0, 1.0, math.pi / 2.0)


def _place(xyz: np.ndarray, *, rotated: bool) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if rotated:
        xyz = np.column_stack([-xyz[:, 1], xyz[:, 0], xyz[:, 2]])
    return xyz + np.asarray(T)


def _table_coords(fem: FEMData, ids) -> np.ndarray:
    row = {int(n): i for i, n in enumerate(fem.nodes.ids)}
    return np.asarray(fem.nodes.coords)[[row[int(n)] for n in ids]]


#: Every cache the merge carries: node- and element-side PGs and labels.
_GROUPS = [(side, kind, name)
           for side in ("nodes", "elements")
           for kind, name in (("physical", "Vol"), ("labels", "v"))]


@pytest.fixture(scope="module")
def module_h5(tmp_path_factory) -> Path:
    """A 2x2x2 hex8 block with PG ``Vol``, label ``v`` and a mesh selection."""
    d = tmp_path_factory.mktemp("c1592")
    path = d / "block.h5"
    with apeGmsh(model_name="block", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, SIDE, SIDE, SIDE, label="v")
        g.physical.add_volume("v", name="Vol")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        ids, xyz, _ = gmsh.model.mesh.getNodes()
        on_face = np.isclose(np.reshape(xyz, (-1, 3))[:, 0], SIDE)
        g.mesh_selection.add(0, [int(n) for n in ids[on_face]], name="face_x")
        g.mesh.queries.get_fem_data(dim=None).to_h5(str(path))
    return path


@pytest.mark.parametrize("rotated", [False, True], ids=["translate", "rotate"])
def test_compose_places_group_and_label_coords(module_h5, rotated):
    src = FEMData.from_h5(str(module_h5))
    host = FEMData.from_h5(str(module_h5))
    fem = host.compose(str(module_h5), label="m", translate=T,
                       rotate=ROT_Z_90 if rotated else None)

    for side, kind, name in _GROUPS:
        comp = getattr(getattr(fem, side), kind)
        src_comp = getattr(getattr(src, side), kind)
        got = comp.node_coords(f"m.{name}")
        np.testing.assert_allclose(
            got, _place(src_comp.node_coords(name), rotated=rotated),
            rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            got, _table_coords(fem, comp.node_ids(f"m.{name}")),
            rtol=0, atol=1e-9)
        # The host's own group stays where it was.
        np.testing.assert_array_equal(
            comp.node_coords(name), src_comp.node_coords(name))

    sel = fem.mesh_selection
    (key,) = [k for k, i in sel._sets.items() if i["name"] == "m.face_x"]
    (src_key,) = [k for k, i in src.mesh_selection._sets.items()
                  if i["name"] == "face_x"]
    np.testing.assert_allclose(
        sel._sets[key]["node_coords"],
        _place(src.mesh_selection._sets[src_key]["node_coords"],
               rotated=rotated),
        rtol=0, atol=1e-9)


def test_assembly_v2_bridge_places_group_coords(module_h5, tmp_path):
    from apeGmsh.assembly import Assembly
    from apeGmsh.opensees import apeSees

    src = FEMData.from_h5(str(module_h5))
    ops = apeSees(src)
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=200e3, nu=0.3, name="steel")
    ops.element.stdBrick(pg="Vol", material=steel)
    archive = tmp_path / "block_model.h5"
    ops.h5(str(archive))

    asm = Assembly("pair")
    asm.instance("a", archive)
    asm.instance("b", archive, translate=T)
    fem = asm.bridge(ndm=3, ndf=3).fem

    src_xyz = src.nodes.physical.node_coords("Vol")
    for label, placed in (("a", src_xyz),
                          ("b", _place(src_xyz, rotated=False))):
        pg = fem.nodes.physical
        got = pg.node_coords(f"{label}.Vol")
        np.testing.assert_allclose(got, placed, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            got, _table_coords(fem, pg.node_ids(f"{label}.Vol")),
            rtol=0, atol=1e-9)


@pytest.mark.parametrize("rotated", [False, True], ids=["translate", "rotate"])
def test_compose_places_node_to_surface_phantoms(tmp_path, rotated):
    path = tmp_path / "capped.h5"
    with apeGmsh(model_name="capped", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, SIDE, SIDE, SIDE, label="v")
        g.model.geometry.add_point(5.0, 5.0, 15.0, label="hub")
        g.model.sync()
        top = g.model.select(None, dim=2).in_box(
            (-1.0, -1.0, SIDE - 0.1), (SIDE + 1, SIDE + 1, SIDE + 0.1),
        ).result().tags()
        g.physical.add_surface(top, name="top")
        g.constraints.node_to_surface("hub", "top", name="cap")
        g.mesh.recipe.structured(size=5.0, fallback="strict")
        g.mesh.queries.get_fem_data(dim=None).to_h5(str(path))

    (src_rec,) = FEMData.from_h5(str(path)).nodes.constraints
    fem = FEMData.from_h5(str(path)).compose(
        str(path), label="m", translate=T,
        rotate=ROT_Z_90 if rotated else None)
    (rec,) = [r for r in fem.nodes.constraints
              if getattr(r, "phantom_coords", None) is not None
              and int(r.master_node) != int(src_rec.master_node)]
    got = np.asarray(rec.phantom_coords)
    np.testing.assert_allclose(
        got, _place(src_rec.phantom_coords, rotated=rotated),
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        got, _table_coords(fem, rec.slave_nodes), rtol=0, atol=1e-9)
