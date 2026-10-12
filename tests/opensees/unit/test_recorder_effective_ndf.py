"""B5-e (#1759, fixes #1745): the recorder rotation roles read the per-node
EFFECTIVE ndf (inferred plus the ``ops.ndf`` overlay).

Every case runs through each path it applies to: the ``recorder.declare``
emit, ``ops.results`` (a faked deck run) and ``.read`` (a faked live
session). The deck-run oracle is
``tests/opensees/live/test_recorder_effective_ndf_live.py``.

1. An element-less master RAISED to ndf 6 by ``ops.ndf`` in an ndf=3 solid
   model records ``rotation_x`` at DOF 4 (#1738 refused it at emit).
2. An element-less master in a model holding an absorbing boundary records
   its rotation: only a u-p class blocks an element-less node (#1738 let
   any unnamed class block it). A u-p class still blocks it.
3. An element-less node LOWERED to ndf 3 by ``ops.ndf`` in an ndf=6 frame
   refuses ``rotation_x`` at emit (main wrote ``-dof 4`` silently).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.unit.test_node_pair_zerolength import _box_with_ground
from tests.opensees.unit.test_readback import DispSpy, with_session
from tests.opensees.unit.test_results_route import (
    fake_run,
    refuses,
    write_rows,
)

ROT = 0.01


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    return tmp_path


def _recorder_lines(ops: apeSees, path: Path) -> list[str]:
    ops.tcl(str(path))
    return [ln for ln in path.read_text().splitlines()
            if ln.startswith("recorder Node")]


def _ran(ops: apeSees, monkeypatch: pytest.MonkeyPatch, *rows: list[float],
         file: str) -> None:
    fake_run(monkeypatch, write_rows(file, *rows))
    ops.py("run/deck.py", run=True)


# ---------------------------------------------------------------------------
# Case 1: a master raised to ndf 6 in an ndf=3 solid model
# ---------------------------------------------------------------------------


@pytest.fixture
def raised() -> Iterator[tuple[apeSees, int]]:
    g, fem, h = _box_with_ground("master")
    try:
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
        ops.element.FourNodeTetrahedron(pg="Body", material=mat)
        ops.ndf(h, ndf=6)
        ops.recorder.declare(nodes=("rotation_x",), ids=(int(h.tag),),
                             file_root="run", name="m")
        yield ops, int(h.tag)
    finally:
        g.end()


def test_raised_master_emits_dof_4(
    raised: tuple[apeSees, int], tmp_path: Path,
) -> None:
    ops, tag = raised
    assert _recorder_lines(ops, tmp_path / "m.tcl") == [
        f"recorder Node -file run/m__default__disp.out -time -node {tag} "
        "-dof 4 disp",
    ]


def test_raised_master_reads_back_through_ops_results(
    raised: tuple[apeSees, int], run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops, tag = raised
    _ran(ops, monkeypatch, [1.0, ROT], file="run/m__default__disp.out")
    r = ops.results("run/deck.py")
    assert r.nodes.get(ids=[tag], component="rotation_x").item() == ROT


def test_raised_master_reads_live_at_dof_4(
    raised: tuple[apeSees, int],
) -> None:
    ops, tag = raised
    spy = DispSpy(ndf=6)
    with_session(ops, spy, ndm=3)
    assert ops.nodes[tag].read("rotation_x") == 0.5
    assert spy.seen == [(tag, 4)]


# ---------------------------------------------------------------------------
# Case 1 through each recorder emit call site of ``BuiltModel``: flat,
# staged (``s.recorder``), partitioned, staged + partitioned. Each test fails
# when only that site's ``effective_ndf=`` argument is removed.
# ---------------------------------------------------------------------------


def _raised_master_model(
    *, staged: bool, partitions: int,
) -> tuple[apeSees, int, Any]:
    from apeGmsh import apeGmsh

    g = apeGmsh(model_name=f"b5e_sites_{int(staged)}_{partitions}")
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="Body")
    h = g.decouple_node(coords=(0.0, 0.0, 2.0), label="master")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    if partitions > 1:
        g.mesh.partitioning.partition(partitions)
        g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)
    assert (len(fem.partitions) > 1) == (partitions > 1)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="Body", material=mat)
    ops.ndf(h, ndf=6)
    tag = int(h.tag)
    decl = ops.recorder.declare(
        nodes=("rotation_x", "rotation_y", "rotation_z"), ids=(tag,),
        file_root="run", name="m")
    if staged:
        with ops.stage(name="load") as s:
            s.recorder(decl)
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
                algorithm=ops.algorithm.Linear(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                constraints=ops.constraints.Penalty(alpha_sp=1e12,
                                                    alpha_mp=1e12),
                numberer=ops.numberer.RCM(),
                system=ops.system.UmfPack(),
                analysis=ops.analysis.Static(),
            )
            s.run(n_increments=1)
    return ops, tag, g


@pytest.mark.parametrize(("staged", "partitions"), [
    (False, 1), (True, 1), (False, 2), (True, 2)],
    ids=["flat", "staged", "partitioned", "staged_partitioned"])
def test_raised_master_records_dofs_4_to_6_through_each_emit_site(
    staged: bool, partitions: int, tmp_path: Path,
) -> None:
    ops, tag, g = _raised_master_model(staged=staged, partitions=partitions)
    try:
        deck = tmp_path / "m.tcl"
        ops.tcl(str(deck))
        lines = [ln for ln in deck.read_text().splitlines()
                 if ln.lstrip().startswith("recorder Node")]
        assert len(lines) == 1, lines
        assert lines[0].strip().endswith(f"-node {tag} -dof 4 5 6 disp"), lines
    finally:
        g.end()


# ---------------------------------------------------------------------------
# Case 2: an element-less master beside an absorbing boundary (or a u-p
# plate, which still blocks it)
# ---------------------------------------------------------------------------


def _plate_with_master(name: str) -> tuple[Any, Any, Any]:
    import gmsh

    from apeGmsh import apeGmsh

    g = apeGmsh(model_name=name)
    g.begin()
    g.model.geometry.add_rectangle(0, 0, 0, 2.0, 1.0, label="plate")
    g.physical.add_surface("plate", name="Plate")
    surf = gmsh.model.getEntities(2)[0][1]
    for _d, c in gmsh.model.getBoundary([(2, surf)], oriented=False):
        gmsh.model.mesh.setTransfiniteCurve(abs(c), 3)
    gmsh.model.mesh.setTransfiniteSurface(surf)
    gmsh.model.mesh.setRecombine(2, surf)
    h = g.decouple_node(coords=(1.0, 2.0, 0.0), label="master")
    g.mesh.generation.generate(dim=2)
    return g, g.mesh.queries.get_fem_data(dim=2), h


@pytest.fixture(params=["absorbing", "up"])
def plate(request: pytest.FixtureRequest) -> Iterator[tuple[apeSees, int, str]]:
    g, fem, h = _plate_with_master(f"b5e_plate_{request.param}")
    try:
        ops = apeSees(fem, default_orientation=None)
        ops.model(ndm=2, ndf=3)
        if request.param == "absorbing":
            ops.element.ASDAbsorbingBoundary2D(
                pg="Plate", G=1e8, v=0.3, rho=2000.0, thickness=1.0,
                btype="B")
        else:
            mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=0.0)
            ops.element.LadrunoUP(pg="Plate", material=mat, Kf=2.2e6,
                                  poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4))
        ops.recorder.declare(nodes=("rotation_z",), ids=(int(h.tag),),
                             file_root="run", name="m")
        yield ops, int(h.tag), request.param
    finally:
        g.end()


def test_element_less_master_beside_a_plate_emits_or_refuses(
    plate: tuple[apeSees, int, str], tmp_path: Path,
) -> None:
    ops, tag, kind = plate
    if kind == "absorbing":
        assert _recorder_lines(ops, tmp_path / "m.tcl") == [
            f"recorder Node -file run/m__default__disp.out -time -node "
            f"{tag} -dof 3 disp",
        ]
    else:
        with pytest.raises(BridgeError, match=rf"rotation_z.*node {tag} "
                           r".*LadrunoUP.*pore pressure"):
            ops.tcl(str(tmp_path / "m.tcl"))


def test_element_less_master_beside_a_plate_through_ops_results(
    plate: tuple[apeSees, int, str], run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops, tag, kind = plate
    if kind == "absorbing":
        _ran(ops, monkeypatch, [1.0, ROT], file="run/m__default__disp.out")
        r = ops.results("run/deck.py")
        assert r.nodes.get(ids=[tag], component="rotation_z").item() == ROT
        return
    # The emit refuses; a deck from an emit that let it through is
    # refused on read by the same table.
    from apeGmsh.opensees._internal.build import recorders as rec_mod

    real = rec_mod._node_record_layout

    def _no_roles(record: Any, decl: Any, node_ids: Any, element_plan: Any,
                  **kw: Any) -> tuple[int, ...]:
        return real(record, decl, node_ids, None)

    with monkeypatch.context() as m:
        m.setattr(rec_mod, "_node_record_layout", _no_roles)
        _ran(ops, m, [1.0, ROT], file="run/m__default__disp.out")
    refuses(ops, "run/deck.py", run_dir,
            rf"rotation_z.*node {tag} .*LadrunoUP")


def test_element_less_master_beside_a_plate_reads_live(
    plate: tuple[apeSees, int, str],
) -> None:
    ops, tag, kind = plate
    spy = DispSpy(ndf=3)
    with_session(ops, spy, ndm=2)
    node = ops.nodes[tag]
    if kind == "absorbing":
        assert node.read("rotation_z") == 0.5
        assert spy.seen == [(tag, 3)]
    else:
        with pytest.raises(BridgeError, match="LadrunoUP"):
            node.read("rotation_z")
        assert spy.seen == []


# ---------------------------------------------------------------------------
# Case 3: a node lowered to ndf 3 in an ndf=6 frame
# ---------------------------------------------------------------------------


@pytest.fixture
def lowered() -> Iterator[tuple[apeSees, int]]:
    from tests.opensees.unit.test_recorder_declare_dof_layout import (
        _frame_with_master,
    )

    g, fem, h = _frame_with_master("b5e_lowered")
    try:
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        ops.element.elasticBeamColumn(
            pg="Beam", transf=ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
        )
        ops.ndf(h, ndf=3)
        ops.recorder.declare(nodes=("rotation_x",), ids=(int(h.tag),),
                             file_root="run", name="m")
        yield ops, int(h.tag)
    finally:
        g.end()


def test_lowered_node_refuses_at_emit(
    lowered: tuple[apeSees, int], tmp_path: Path,
) -> None:
    ops, tag = lowered
    with pytest.raises(BridgeError, match=rf"node {tag} \(ndm=3, ndf=3\) "
                       r"has no DOF.*rotation_x"):
        ops.tcl(str(tmp_path / "m.tcl"))


def test_lowered_node_refuses_on_read(lowered: tuple[apeSees, int]) -> None:
    ops, tag = lowered
    spy = DispSpy(ndf=3)
    with_session(ops, spy, ndm=3)
    with pytest.raises(BridgeError, match=rf"node {tag} has no 'rotation_x'"
                       r": it has ndf=3"):
        ops.nodes[tag].read("rotation_x")
    assert spy.seen == []
