"""ADR 0055 Phase 2 — staged-archival READ side (P2.2).

``OpenSeesModel.from_h5`` now loads staged 2.18.0 archives:
``.stages()`` exposes value-form ``StageRecordRO`` records and
``to_h5`` echoes them back through ``H5Emitter.restore_stage_blocks``.
This module pins the read contract:

1. ``.stages()`` reconstructs what the writer persisted.
2. **THE acceptance gate** — ``from_h5 → to_h5 → from_h5`` is
   ``model_hash``-stable and the stages zone is structurally
   identical (store-and-echo).
3. A malformed stages zone fails loud (``MalformedH5Error``), never
   loads as a different staged program.
4. tcl / py / live re-emit of a staged archive fails loud until the
   staged replay lands (P2.3); ``build('h5')`` round-trips.
5. ``ModelData.from_h5`` warns on a staged archive (the laundering
   side door — a re-save would strip the staged program).

Fixtures here use a REAL two-quad :class:`FEMData` (not the writer
tests' ``FEMStub``) because ``OpenSeesModel.from_h5`` reloads the
neutral zone via ``FEMData.from_h5``, which rejects stub-only files.
"""
from __future__ import annotations

from pathlib import Path

import h5py
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
from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.h5.test_h5_stages_writer import (
    _chain,
    _collect_zone,
    _model_hash_of,
)


# ---------------------------------------------------------------------------
# Real-FEMData two-quad fixture (mirrors the writer tests' stub geometry)
# ---------------------------------------------------------------------------


def build_two_quad_fem() -> FEMData:
    """Two stacked quads: ``Rock`` (nodes 1-4) + ``Fill`` (nodes 3-6),
    with ``Base`` (1, 2) and ``FillTop`` (5, 6) node sets."""
    node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=2,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array(
            [[1, 2, 3, 4], [4, 3, 5, 6]], dtype=np.int64,
        ),
    )

    def _sel(ids: "list[int]") -> np.ndarray:
        return np.array(ids, dtype=np.int64)

    def _coords(ids: "list[int]") -> np.ndarray:
        return node_coords[[i - 1 for i in ids]]

    pg = {
        (2, 201): {
            "name": "Rock",
            "node_ids": _sel([1, 2, 3, 4]),
            "node_coords": _coords([1, 2, 3, 4]),
            "element_ids": _sel([1]),
        },
        (2, 202): {
            "name": "Fill",
            "node_ids": _sel([3, 4, 5, 6]),
            "node_coords": _coords([3, 4, 5, 6]),
            "element_ids": _sel([2]),
        },
        (0, 203): {
            "name": "Base",
            "node_ids": _sel([1, 2]),
            "node_coords": _coords([1, 2]),
            "element_ids": np.array([], dtype=np.int64),
        },
        (0, 204): {
            "name": "FillTop",
            "node_ids": _sel([5, 6]),
            "node_coords": _coords([5, 6]),
            "element_ids": np.array([], dtype=np.int64),
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={3: quad_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=6, n_elems=2, bandwidth=3, types=[quad_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


def _real_two_stage_bridge() -> apeSees:
    """The writer tests' 2-stage program over a real FEMData."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))

    with ops.stage(name="construction") as s:
        s.activate(pgs=["Fill"])
        s.fix(pg="FillTop", dofs=(1, 1))
        s.analysis(**_chain(ops))
        s.run(n_increments=5)
    with ops.stage(name="loading") as s:
        s.set_time(2.5)
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts) as p:
            p.load(pg="Fill", forces=(10.0, 0.0))
        s.initial_stress(
            name="insitu", pg="Rock",
            sigma_xx=-1.0e3, sigma_yy=-1.0e3, sigma_zz=-2.0e3,
            ramp_steps=4,
        )
        s.analysis(**_chain(ops))
        s.run(n_increments=3, dt=0.01)
    return ops


def _real_kitchen_sink_bridge() -> apeSees:
    """The writer tests' kitchen-sink stage over a real FEMData."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))

    with ops.stage(name="mutate") as s:
        s.activate(pgs=["Fill"])
        s.support(pg="FillTop", dofs=(1, 0))
        s.mass(pg="FillTop", values=(2.0, 2.0))
        s.remove_sp(pg="Base", dofs=(1,))
        s.remove_element(elements=[1])
        s.damping.rayleigh(alpha_m=0.05)
        s.damping.rayleigh(alpha_m=0.1, on="Fill")
        s.recorder(ops.recorder.Node(
            file="r.out", response="disp", nodes=(5,), dofs=(1, 2),
        ))
        s.set_creep(True)
        s.reset()
        s.activate_absorbing(pg="Fill")
        s.analysis(**_chain(ops))
        s.run(n_increments=2, dt=0.5)
    return ops


# ---------------------------------------------------------------------------
# 1. .stages() reconstructs the persisted program
# ---------------------------------------------------------------------------


def test_stages_accessor_matches_two_stage_fixture(tmp_path: Path) -> None:
    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))

    m = OpenSeesModel.from_h5(str(out))
    stages = m.stages()
    assert len(stages) == 2

    s0, s1 = stages
    assert s0.name == "construction"
    assert s0.analyze_steps == 5
    assert s0.analyze_dt is None            # tri-state: never set
    assert s0.set_time is None
    assert s0.set_creep_on is None
    assert s0.pre_analyze_reset is False
    assert s0.domain_changed is True
    assert s0.activated_pgs == ("Fill",)
    assert sorted(s0.owned_node_ids) == [5, 6]
    assert len(s0.owned_element_ids) == 1
    assert len(s0.fixes) == 2               # FillTop fan-out
    # Padded to the global ndf envelope (inv #3, read side).
    assert all(len(f.dofs) == 2 for f in s0.fixes)
    assert s0.chain_attrs["algorithm"] == "Newton"
    assert s0.chain_attrs["analysis"] == "Static"
    assert s0.patterns == ()
    assert s0.initial_stress == ()

    assert s1.name == "loading"
    assert s1.analyze_steps == 3
    assert s1.analyze_dt == pytest.approx(0.01)
    assert s1.set_time == pytest.approx(2.5)
    assert s1.domain_changed is False
    assert len(s1.patterns) == 1
    assert len(s1.patterns[0].loads) == 4   # Fill fan-out
    assert len(s1.pattern_seq) == 1
    assert len(s1.initial_stress) == 1
    isr = s1.initial_stress[0]
    assert isr.name == "insitu"
    assert isr.pg == "Rock"
    assert isr.sigma_zz == pytest.approx(-2.0e3)
    assert isr.ramp_steps == 4


def test_stages_accessor_kitchen_sink(tmp_path: Path) -> None:
    out = tmp_path / "sink.h5"
    _real_kitchen_sink_bridge().h5(str(out))

    (s,) = OpenSeesModel.from_h5(str(out)).stages()
    assert s.set_creep_on is True
    assert s.pre_analyze_reset is True
    assert sorted(s.remove_sps) == [(1, 1), (2, 1)]
    assert len(s.remove_elements) == 1
    assert len(s.masses) == 2
    # HOLD pattern read back with its sp_holds rows.
    holds = [p for p in s.patterns if p.sp_holds]
    assert len(holds) == 1
    assert sorted(holds[0].sp_holds) == [(5, 1), (6, 1)]
    # Pattern order restored from emit_index stamps.
    assert list(s.pattern_seq) == sorted(s.pattern_seq)
    # Rayleigh + provenance.
    assert s.rayleighs == ((0.05, 0.0, 0.0, 0.0),)
    assert len(s.rayleigh_seq) == 1
    assert len(s.regions) >= 1
    assert len(s.region_seq) == len(s.regions)
    assert len(s.recorders) == 1
    # Absorbing flip — declarative discriminant.
    assert s.activate_absorbing == (("Fill", None),)


# ---------------------------------------------------------------------------
# 2. THE acceptance gate — from_h5 → to_h5 → from_h5 hash-stable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [
    _real_two_stage_bridge, _real_kitchen_sink_bridge,
])
def test_staged_roundtrip_hash_stable(tmp_path: Path, builder) -> None:
    p1 = tmp_path / "m1.h5"
    builder().h5(str(p1))

    m1 = OpenSeesModel.from_h5(str(p1))
    p2 = tmp_path / "m2.h5"
    m1.to_h5(str(p2))
    m2 = OpenSeesModel.from_h5(str(p2))

    # Hash-stable (the store-and-echo guarantee)...
    assert _model_hash_of(p1) == _model_hash_of(p2)
    # ...and the stages zone is structurally identical.
    with h5py.File(str(p1), "r") as f1, h5py.File(str(p2), "r") as f2:
        assert _collect_zone(f1["opensees"]["stages"]) == _collect_zone(
            f2["opensees"]["stages"]
        )
    # The second read agrees with the first.
    assert [s.name for s in m2.stages()] == [s.name for s in m1.stages()]


# ---------------------------------------------------------------------------
# 3. Malformed stages zone fails loud
# ---------------------------------------------------------------------------


def test_malformed_stage_missing_analyze_steps_fails_loud(
    tmp_path: Path,
) -> None:
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))
    with h5py.File(str(out), "r+") as f:
        del f["opensees"]["stages"]["stage_000"].attrs["analyze_steps"]

    with pytest.raises(MalformedH5Error, match="analyze_steps"):
        OpenSeesModel.from_h5(str(out))


def test_malformed_n_stages_mismatch_fails_loud(tmp_path: Path) -> None:
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))
    with h5py.File(str(out), "r+") as f:
        f["opensees"]["stages"].attrs["n_stages"] = 7

    with pytest.raises(MalformedH5Error, match="n_stages"):
        OpenSeesModel.from_h5(str(out))


# ---------------------------------------------------------------------------
# 4. Re-emit targets: tcl/py/live fail loud (P2.3 pending); h5 round-trips
# ---------------------------------------------------------------------------


def test_staged_build_tcl_py_live_fail_loud(tmp_path: Path) -> None:
    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))
    m = OpenSeesModel.from_h5(str(out))
    for target in ("tcl", "py", "live"):
        with pytest.raises(NotImplementedError, match="P2.3"):
            m.build(target)


def test_staged_build_h5_roundtrips(tmp_path: Path) -> None:
    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))
    m = OpenSeesModel.from_h5(str(out))
    rt = tmp_path / "rt.h5"
    m.build("h5", out=str(rt))
    assert _model_hash_of(out) == _model_hash_of(rt)


# ---------------------------------------------------------------------------
# 5. ModelData laundering warning
# ---------------------------------------------------------------------------


def test_model_data_warns_on_staged_archive(tmp_path: Path) -> None:
    from apeGmsh.opensees import ModelData

    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))
    with pytest.warns(UserWarning, match="STAGED"):
        ModelData.from_h5(str(out))
