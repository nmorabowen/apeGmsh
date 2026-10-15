"""#1948 (G8): stage kinds on MPCO, and multi-stage ``.out`` reads.

Kinds.  An ``.mpco`` does not tag its stages; ``Results.from_mpco(...,
model_h5=)`` kinds each stage from the archive's own ``analysis`` record
(``/opensees/stages/stage_NNN/analysis@analysis``).  ``analyze_dt`` does
not decide it: a ``Static`` stage may carry ``dt=dlam`` (silent, the
``ops.stage`` docstring's own spelling) and a ``Transient`` one may omit
``dt``.  Oracle: the archived token, ``Static`` -> ``static``,
``Transient`` / ``VariableTransient`` -> ``transient``; anything else
raises.

Multi-stage ``.out``.  ``Results.from_recorders(..., stages=live.stages)``
transcodes every stage a ``LiveRecorders`` run completed into one file,
one stage each, and the cache key covers each stage's name and kind.
Oracle: synthetic files with closed-form values, and on a live stock run
(gravity ``LoadControl`` then ``Newmark`` ``dt=0.01``) ``ops.nodeDisp`` at
each stage's last step.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results._bind import _ANALYSIS_KIND, _bind_stage_names
from apeGmsh.results.readers._mpco import MPCOReader
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)

from tests.conftest import _open_model_from_h5, _stub_model_h5_path
from tests.test_results_transcoder import _MockFem, _write_synthetic_node_out


# =====================================================================
# Stage kinds on MPCO (bind beside the stage-name pairing)
# =====================================================================

_IDS = np.array([1, 2], dtype=np.int64)


def _write_two_stage_mpco(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        f.create_group("INFO")
        for k in (1, 2):
            on = f.create_group(f"MODEL_STAGE[{k}]/RESULTS/ON_NODES")
            f.create_group(f"MODEL_STAGE[{k}]/RESULTS/ON_ELEMENTS")
            disp = on.create_group("DISPLACEMENT")
            disp.attrs["COMPONENTS"] = np.array([b"Ux,Uy"])
            disp.create_dataset("ID", data=_IDS.reshape(-1, 1))
            for s in range(3):
                ds = disp.create_dataset(f"DATA/STEP_{s}", data=np.zeros((2, 2)))
                ds.attrs["TIME"] = np.array([0.1 * (s + 1)])
    return path


def _program(*stages: tuple[str, "dict[str, Any]"]) -> Any:
    """An ``OpenSeesModel`` stand-in: ``stages()`` as ``StageRecordRO``s."""
    from apeGmsh.opensees._internal.typed_records import StageRecordRO

    records = [StageRecordRO(name=n, analyze_steps=3, analyze_dt=0.1,
                             chain_attrs=chain) for n, chain in stages]
    return SimpleNamespace(stages=lambda: list(records))


def test_kind_table_covers_every_analysis_the_bridge_emits() -> None:
    from apeGmsh.opensees.analysis import analysis

    assert set(_ANALYSIS_KIND) == set(analysis.__all__)


def test_mpco_stages_take_the_archived_analysis_kind(tmp_path: Path) -> None:
    """Both stages carry ``analyze_dt=0.1``; the analysis record decides."""
    model = _program(("gravity", {"analysis": "Static"}),
                     ("shake", {"analysis": "Transient"}))
    with MPCOReader(_write_two_stage_mpco(tmp_path / "r.mpco")) as r:
        assert [s.kind for s in r.stages()] == ["transient", "transient"]
        _bind_stage_names(r, model, model_path="model.h5")
        stages = r.stages()
    assert [(s.name, s.kind) for s in stages] == [
        ("gravity", "static"), ("shake", "transient")]
    assert [s.aliases for s in stages] == [
        ("MODEL_STAGE[1]",), ("MODEL_STAGE[2]",)]


def test_variable_transient_is_transient(tmp_path: Path) -> None:
    model = _program(("a", {"analysis": "VariableTransient"}),
                     ("b", {"analysis": "Static"}))
    with MPCOReader(_write_two_stage_mpco(tmp_path / "r.mpco")) as r:
        _bind_stage_names(r, model, model_path="model.h5")
        assert [s.kind for s in r.stages()] == ["transient", "static"]


@pytest.mark.parametrize("chain, match", [
    ({"analysis": "PFEM"}, "analysis 'PFEM', which has no stage kind"),
    ({}, "archives no analysis record"),
])
def test_unknown_or_missing_analysis_record_raises(
    tmp_path: Path, chain: "dict[str, Any]", match: str,
) -> None:
    model = _program(("a", {"analysis": "Static"}), ("b", chain))
    with MPCOReader(_write_two_stage_mpco(tmp_path / "r.mpco")) as r:
        with pytest.raises(ValueError, match=match):
            _bind_stage_names(r, model, model_path="model.h5")


def test_attach_refuses_a_kind_that_is_not_a_program_stage_kind(
    tmp_path: Path,
) -> None:
    with MPCOReader(_write_two_stage_mpco(tmp_path / "r.mpco")) as r:
        with pytest.raises(ValueError, match=r"\['mode'\] are not program"):
            r.attach_stage_names(["a", "b"], ["static", "mode"])
        with pytest.raises(ValueError, match="2 names and 1 kinds"):
            r.attach_stage_names(["a", "b"], ["static"])


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


def test_the_single_stage_cache_key_covers_its_kind(tmp_path: Path) -> None:
    """Regression: ``stage_kind=`` was not in the key, so a second read of
    the same files with another kind returned the first read's kind."""
    fem = _MockFem([1, 2])
    _write_two_stages(tmp_path / "out")
    for kind in ("static", "transient"):
        with _read(tmp_path / "out", fem, tmp_path / "c",
                   stage_id="gravity", stage_kind=kind) as r:
            assert [s.kind for s in r.stages] == [kind]


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
