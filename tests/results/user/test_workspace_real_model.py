"""The set verdict on archives the bridge really writes (ADR 0132 D3).

Two ``ops.h5`` archives of one tet box with a ground node on a mesh node's
position and two zero-length springs on that node pair, declared in the two
orders. The swap leaves ``fem_hash`` equal and moves ``model_hash``: the springs
read ``lost`` (each centroid fits both), a tet set reads ``moved``, and a node
set stays ``current``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.user import SetDef, WorkspaceFile, write_workspace


def _build(path: Path, *, swap: bool) -> None:
    from apeGmsh import apeGmsh
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

    g = apeGmsh(model_name="workspace_swap")
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
        g.physical.add_volume("body", name="Body")
        ground = g.decouple_node(coords=(1.0, 0.0, 0.0), label="gnd")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
        ops.element.FourNodeTetrahedron(pg="Body", material=mat)
        k1 = ops.uniaxialMaterial.ElasticMaterial(E=1e6)
        k2 = ops.uniaxialMaterial.ElasticMaterial(E=2e6)
        coords = np.asarray(fem.nodes.coords)
        corner = int(np.asarray(fem.nodes.ids)[np.argmin(np.linalg.norm(coords - [1, 0, 0], axis=1))])
        a = dict(nodes=(corner, ground), mat_dirs=(ZeroLengthMatDir(material=k1, dof=1),))
        b = dict(nodes=(corner, ground), mat_dirs=(ZeroLengthMatDir(material=k2, dof=2),))
        for spec in ((b, a) if swap else (a, b)):
            ops.element.ZeroLength(**spec)
        ops.ndf(ground, ndf=3)
        ops.h5(str(path))
    finally:
        g.end()


@pytest.fixture(scope="module")
def archives(tmp_path_factory) -> tuple[Path, Path]:
    folder = tmp_path_factory.mktemp("swap")
    before, after = folder / "before.h5", folder / "after.h5"
    _build(before, swap=False)
    _build(after, swap=True)
    return before, after


def _tags(path: Path) -> tuple[list[int], list[int], int]:
    with h5py.File(path, "r") as f:
        meta = f["opensees/element_meta"]
        springs = meta["zeroLength/ids"][...].tolist()
        tets = meta["FourNodeTetrahedron/ids"][...].tolist()
        node = int(f["nodes/ids"][0])
    return springs, tets[:3], node


def test_a_real_declaration_swap(archives, tmp_path: Path) -> None:
    before, after = archives
    with h5py.File(before, "r") as f, h5py.File(after, "r") as g:
        assert f["meta/lineage"].attrs["fem_hash"] == g["meta/lineage"].attrs["fem_hash"]
        assert f["meta/lineage"].attrs["model_hash"] != g["meta/lineage"].attrs["model_hash"]
    springs, tets, node = _tags(before)
    assert _tags(after)[0] == springs, "the swap kept the tags and exchanged the springs"

    model = shutil.copy(before, tmp_path / "box.h5")
    ws = write_workspace(tmp_path / "box.workspace.h5", sets=[
        SetDef("springs", "elements", springs), SetDef("tets", "elements", tets),
        SetDef("corner", "nodes", [node])])
    assert {v.verdict for v in WorkspaceFile(ws).verdicts().verdicts} == {"current"}

    shutil.copy(after, model)
    table = WorkspaceFile(ws).verdicts()
    assert (table.by_name("springs").verdict, table.by_name("springs").matched) == ("lost", 0)
    assert table.by_name("tets").verdict == "moved"
    assert table.by_name("corner").verdict == "current"
