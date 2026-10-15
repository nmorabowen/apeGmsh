"""#2035 (G8b): multi-stage ``.out`` reads through one ``from_recorders`` call.

``Results.from_recorders(..., stages=live.stages)`` transcodes every stage a
``LiveRecorders`` run completed into one native file, one stage each with its
name and kind, each read from the files prefixed ``<name>__``; the cache key
covers every stage.  Oracles: synthetic files with closed-form values, and on
a live stock run (gravity ``LoadControl``, then ``Newmark`` at ``dt=0.01``)
``ops.nodeDisp`` at each stage's last step, rtol 1e-10.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.transcoders import RecorderTranscoder

from tests.conftest import _open_model_from_h5, _stub_model_h5_path
from tests.test_results_transcoder import _MockFem, _write_synthetic_node_out


# =====================================================================
# Multi-stage .out reads (synthetic files)
# =====================================================================

def _node_spec(fem: Any) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="r", components=("displacement_x",),
            dt=None, n_steps=None, node_ids=np.array([1, 2]),
        ),),
    )


def _write_two_stages(out: Path) -> None:
    out.mkdir(exist_ok=True)
    _write_synthetic_node_out(
        out / "gravity__r_disp.out", time=[0.5, 1.0], node_ids=[1, 2],
        dofs=[1], value_fn=lambda k, nid, dof: 1.0 * nid + k)
    _write_synthetic_node_out(
        out / "shake__r_disp.out", time=[0.01, 0.02, 0.03], node_ids=[1, 2],
        dofs=[1], value_fn=lambda k, nid, dof: -10.0 * nid - k)


_STAGES = (SimpleNamespace(name="gravity", kind="static"),
           SimpleNamespace(name="shake", kind="transient"))


def _read(out: Path, fem: Any, cache: Path, **kw: Any) -> Results:
    return Results.from_recorders(
        _node_spec(fem), out, fem=fem, cache_root=cache,
        model=_open_model_from_h5(_stub_model_h5_path()), **kw)


def test_one_file_holds_every_live_stage(tmp_path: Path) -> None:
    fem = _MockFem([1, 2])
    _write_two_stages(tmp_path / "out")
    with _read(tmp_path / "out", fem, tmp_path / "c", stages=_STAGES) as r:
        assert [(s.name, s.kind) for s in r.stages] == [
            ("gravity", "static"), ("shake", "transient")]
        grav = r.stage("gravity").nodes.get(component="displacement_x")
        shake = r.stage("shake").nodes.get(component="displacement_x")
    np.testing.assert_allclose(grav.time, [0.5, 1.0])
    np.testing.assert_allclose(grav.values, [[1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(shake.time, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(shake.values[-1], [-12.0, -22.0])


def test_the_cache_key_covers_every_stage_kind(tmp_path: Path) -> None:
    """Same files, one stage relabelled: a new file, not the cached one."""
    fem = _MockFem([1, 2])
    _write_two_stages(tmp_path / "out")
    relabelled = (_STAGES[0], SimpleNamespace(name="shake", kind="static"))
    with _read(tmp_path / "out", fem, tmp_path / "c", stages=_STAGES) as r:
        assert r.stage("shake").kind == "transient"
    with _read(tmp_path / "out", fem, tmp_path / "c", stages=relabelled) as r:
        assert r.stage("shake").kind == "static"


@pytest.mark.parametrize("kw", [
    {"stage_id": "gravity"}, {"stage_name": "x"}, {"stage_kind": "static"},
])
def test_stages_excludes_the_single_stage_arguments(
    tmp_path: Path, kw: "dict[str, Any]",
) -> None:
    fem = _MockFem([1, 2])
    with pytest.raises(TypeError, match="pass stages= or"):
        _read(tmp_path / "out", fem, tmp_path / "c", stages=_STAGES, **kw)


def test_a_stage_kind_the_file_cannot_hold_is_refused(tmp_path: Path) -> None:
    fem = _MockFem([1, 2])
    _write_two_stages(tmp_path / "out")
    bad = (_STAGES[0], SimpleNamespace(name="shake", kind="dynamic"))
    with pytest.raises(ValueError, match="kind must be one of"):
        _read(tmp_path / "out", fem, tmp_path / "c", stages=bad)


@pytest.mark.parametrize("stages, match", [
    ((), "must name each stage once"),
    ((SimpleNamespace(name="a", kind="static"),
      SimpleNamespace(name="a", kind="transient")), "must name each stage once"),
])
def test_stages_must_name_each_stage_once(tmp_path: Path, stages, match) -> None:
    fem = _MockFem([1, 2])
    with pytest.raises(ValueError, match=match):
        _read(tmp_path / "out", fem, tmp_path / "c", stages=stages)


def test_the_transcoder_writes_each_stage_from_its_own_files(tmp_path: Path) -> None:
    """Direct ``RecorderTranscoder(stages=)``: the stage order follows
    ``stages``, and each stage reads only its own prefixed file."""
    fem = _MockFem([1, 2])
    _write_two_stages(tmp_path / "out")
    target = tmp_path / "t.h5"
    RecorderTranscoder(_node_spec(fem), tmp_path / "out", target, fem,
                       stages=_STAGES[::-1]).run()
    with Results.from_native(target, fem=fem, model=_open_model_from_h5(target)) as r:
        assert [(s.name, s.kind) for s in r.stages] == [
            ("shake", "transient"), ("gravity", "static")]
        assert r.stage("gravity").nodes.get(component="displacement_x").values.shape == (2, 2)
        assert r.stage("shake").nodes.get(component="displacement_x").values.shape == (3, 2)


# =====================================================================
# Live, stock openseespy: gravity (static) then Newmark (dt=0.01) on .out
# =====================================================================

H = 3.0
P_GRAV, P_DYN = -50e3, 20e3
N_GRAV, N_DYN, DT = 4, 25, 0.01


def _cantilever_fem() -> Any:
    import gmsh

    from apeGmsh import apeGmsh

    with apeGmsh(model_name="g8_cantilever", verbose=False) as g:
        p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
        p1 = g.model.geometry.add_point(0.0, H, 0.0)
        c = g.model.geometry.add_line(p0, p1)
        gmsh.model.mesh.setTransfiniteCurve(c, 4)
        g.physical.add(1, [c], name="Col")
        g.labels.add(0, [p0], "base")
        g.labels.add(0, [p1], "tip")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


@pytest.mark.live
def test_live_gravity_then_newmark_reads_back_both_stages(tmp_path: Path) -> None:
    """The bridge builds the model in the live domain; the two stages are
    driven with raw ``ops`` calls (#1276: no staged live run)."""
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.opensees.emitter.live import get_ops
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    fem = _cantilever_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Col", transf=tr, A=0.02, E=2.0e11, Iz=3.0e-4)
    ops.fix(label="base", dofs=(1, 1, 1))
    tip = int(ops.nodes.one(label="tip"))
    ops.mass(nodes=[tip], values=(1.0e3, 1.0e3, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=tip, forces=(0.0, P_GRAV, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0 / N_GRAV)
    ops.analysis.Static()
    ops.h5(str(tmp_path / "model.h5"))
    model = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))

    nodes = np.asarray([int(n) for n in fem.nodes.ids])
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="all",
            components=("displacement_x", "displacement_y"),
            dt=None, n_steps=None, node_ids=nodes,
        ),),
    )
    # Build the domain through the bridge with zero steps: analyze(1)
    # would also take the first gravity step outside any stage.
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    ops.build().emit(LiveOpsEmitter(wipe=True))
    osp = get_ops()
    expected: dict[str, np.ndarray] = {}
    out = tmp_path / "out"
    with spec.emit_recorders(str(out), ops=osp, model=model) as live:
        live.begin_stage("gravity", kind="static")
        assert osp.analyze(N_GRAV) == 0
        expected["gravity"] = _disp(osp, nodes)
        live.end_stage()

        osp.loadConst("-time", 0.0)
        osp.wipeAnalysis()
        osp.timeSeries("Constant", 901)
        osp.pattern("Plain", 901, 901)
        osp.load(tip, P_DYN, 0.0, 0.0)
        osp.constraints("Plain")
        osp.numberer("RCM")
        osp.system("BandGeneral")
        osp.test("NormDispIncr", 1e-12, 10)
        osp.algorithm("Linear")
        osp.integrator("Newmark", 0.5, 0.25)
        osp.analysis("Transient")
        live.begin_stage("shake", kind="transient")
        assert osp.analyze(N_DYN, DT) == 0
        expected["shake"] = _disp(osp, nodes)
        live.end_stage()

    res = Results.from_recorders(spec, out, fem=fem, model=model,
                                 cache_root=tmp_path / "cache",
                                 stages=live.stages)
    assert [(s.name, s.kind) for s in res.stages] == [
        ("gravity", "static"), ("shake", "transient")]
    for name, n_steps in (("gravity", N_GRAV), ("shake", N_DYN)):
        st = res.stage(name)
        for k, comp in enumerate(("displacement_x", "displacement_y")):
            slab = st.nodes.get(component=comp, ids=nodes)
            assert np.asarray(slab.values).shape[0] == n_steps
            got = np.asarray(slab.values)[-1]
            order = [int(i) for i in np.asarray(slab.node_ids)]
            want = expected[name][[list(nodes).index(i) for i in order], k]
            np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-14)
    # The transient stage moved the tip off its gravity position.
    t = list(nodes).index(tip)
    assert abs(expected["shake"][t, 0] - expected["gravity"][t, 0]) > 1e-6


def _disp(osp: Any, nodes: np.ndarray) -> np.ndarray:
    return np.array([[osp.nodeDisp(int(n), 1), osp.nodeDisp(int(n), 2)]
                     for n in nodes])
