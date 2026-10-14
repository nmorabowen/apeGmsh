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

Fixtures here use a REAL two-quad :class:`FEMData` (the writer
tests' bridges, shared since #1880) because ``OpenSeesModel.from_h5`` reloads the
neutral zone via ``FEMData.from_h5``, which rejects stub-only files.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.h5._opensees_model_fixtures import build_two_quad_fem
from tests.opensees.h5.test_h5_stages_writer import (
    _build_kitchen_sink_bridge,
    _build_two_stage_bridge,
    _chain,  # noqa: F401 -- re-exported: sibling tests import it from here
    build_material_stage_bridge,
    _collect_zone,
    _model_hash_of,
)


# ---------------------------------------------------------------------------
# The writer tests' bridges: built on a real two-quad FEMData (#1880)
# ---------------------------------------------------------------------------

#: The reader and the writer tests drive the same two programs; one
#: builder each, so the R1b oracle replays each model once (#1797).
_real_two_stage_bridge = _build_two_stage_bridge
_real_kitchen_sink_bridge = _build_kitchen_sink_bridge


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
    # Unconditional stage barrier — even a pure-loading stage.
    assert s1.domain_changed is True
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


def test_staged_build_tcl_py_succeed_live_fails_loud(tmp_path: Path) -> None:
    """ADR 0055 P2.3: tcl/py re-emit a staged archive (deck-equality is
    pinned in test_h5_stages_replay.py); live stays fail-loud
    (LiveOpsEmitter.stage_open raises)."""
    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))
    m = OpenSeesModel.from_h5(str(out))
    assert isinstance(m.build("tcl"), str)
    assert isinstance(m.build("py"), str)
    with pytest.raises(NotImplementedError, match="live"):
        m.build("live")


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


# ---------------------------------------------------------------------------
# 6. Gate-2 fix round — recorder order, equalDOF pads, fail-loud holes,
#    compose-buffers bypass, legacy pin
# ---------------------------------------------------------------------------


def test_stage_recorder_order_survives_numeric_suffix(tmp_path: Path) -> None:
    """``recorder_name`` is ``{kind}_{idx}`` UNPADDED — plain sorted()
    would scramble mixed kinds and ``_10`` before ``_2``.  Drive 12
    same-kind + 1 mixed recorder through a stage bracket and assert
    the read order is the emit order."""
    from types import SimpleNamespace

    from apeGmsh.opensees.emitter import h5_reader
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    e.model(ndm=2, ndf=2)
    e.stage_open("s")
    for i in range(12):
        e.recorder("Node", "-file", f"n{i}.out", "-node", i + 1)
    e.recorder("Element", "-file", "e.out", "-ele", 1)
    e.analyze(steps=1)
    e.stage_close()
    e.set_stage_records([SimpleNamespace(
        name="s", n_increments=1, dt=None, activated_pgs=(),
        initial_stress_records=(), activate_absorbing_records=(),
        update_parameter_records=(), zero_velocity_records=(),
    )])
    out = tmp_path / "rec_order.h5"
    e.write(str(out))

    with h5_reader.open(str(out)) as m:
        (stage,) = m.stages()
    files = [
        next(str(a) for a in r.args if str(a).endswith(".out"))
        for r in stage.recorders
    ]
    assert files == [f"n{i}.out" for i in range(12)] + ["e.out"]


def test_flat_mixed_kind_recorders_roundtrip_hash_stable(
    tmp_path: Path,
) -> None:
    """The flat ``recorders()`` reader shares the numeric-suffix sort —
    a vanilla archive with mixed-kind recorders must order by emit and
    stay hash-stable across ``from_h5 → to_h5``."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.recorder.Node(
        file="n.out", response="disp", nodes=(1,), dofs=(1,),
    )
    ops.recorder.Element(file="e.out", response="stresses", elements=(1,))

    p1 = tmp_path / "m1.h5"
    ops.h5(str(p1))
    m1 = OpenSeesModel.from_h5(str(p1))
    kinds = [r.kind for r in m1.recorders()]
    assert kinds == ["Node", "Element"]      # emit order, not alphabetical
    p2 = tmp_path / "m2.h5"
    m1.to_h5(str(p2))
    assert _model_hash_of(p1) == _model_hash_of(p2)


def test_stage_equal_dof_pads_trimmed(tmp_path: Path) -> None:
    """equalDOF dofs are a 1-based dof LIST — the compound's zero pads
    must not surface as dof-0 entries on the .stages() API."""
    from types import SimpleNamespace

    from apeGmsh.opensees.emitter import h5_reader
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.stage_open("s")
    e.equalDOF(1, 2, 1, 2, 3)   # narrower than ndf=6
    e.analyze(steps=1)
    e.stage_close()
    e.set_stage_records([SimpleNamespace(
        name="s", n_increments=1, dt=None, activated_pgs=(),
        initial_stress_records=(), activate_absorbing_records=(),
        update_parameter_records=(), zero_velocity_records=(),
    )])
    out = tmp_path / "eqdof.h5"
    e.write(str(out))

    with h5_reader.open(str(out)) as m:
        (stage,) = m.stages()
    (eq,) = stage.equal_dofs
    assert eq.dofs == (1, 2, 3)              # pads trimmed


@pytest.mark.parametrize("mutate, match", [
    (lambda f: f["opensees"]["stages"]["stage_000"].attrs.__delitem__(
        "name"), "missing name"),
    (lambda f: f["opensees"]["stages"]["stage_001"]["patterns"][
        list(f["opensees"]["stages"]["stage_001"]["patterns"])[0]
    ].attrs.__delitem__("emit_index"), "missing emit_index"),
])
def test_malformed_attr_strips_fail_loud(
    tmp_path: Path, mutate, match,
) -> None:
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))
    with h5py.File(str(out), "r+") as f:
        mutate(f)
    with pytest.raises(MalformedH5Error, match=match):
        OpenSeesModel.from_h5(str(out))


def test_malformed_empty_stages_group_fails_loud(tmp_path: Path) -> None:
    """A stages group with zero children must NOT silently flatten the
    archive into the flat replay path."""
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    out = tmp_path / "staged.h5"
    _real_two_stage_bridge().h5(str(out))
    with h5py.File(str(out), "r+") as f:
        stages = f["opensees"]["stages"]
        for child in list(stages):
            del stages[child]
        stages.attrs["n_stages"] = 0
    with pytest.raises(MalformedH5Error, match="no\\s+stage_NNN"):
        OpenSeesModel.from_h5(str(out))


def test_malformed_region_missing_provenance_fails_loud(
    tmp_path: Path,
) -> None:
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    out = tmp_path / "sink.h5"
    _real_kitchen_sink_bridge().h5(str(out))
    with h5py.File(str(out), "r+") as f:
        g = f["opensees"]["stages"]["stage_000"]["regions"]
        del g[list(g)[0]].attrs["emit_index"]
    with pytest.raises(MalformedH5Error, match="tag or emit_index"):
        OpenSeesModel.from_h5(str(out))


def test_malformed_absorbing_without_discriminant_fails_loud(
    tmp_path: Path,
) -> None:
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    out = tmp_path / "sink.h5"
    _real_kitchen_sink_bridge().h5(str(out))
    with h5py.File(str(out), "r+") as f:
        ab = f["opensees"]["stages"]["stage_000"]["activate_absorbing"][
            "absorb_000"
        ]
        del ab.attrs["pg"]
    with pytest.raises(MalformedH5Error, match="neither pg"):
        OpenSeesModel.from_h5(str(out))


def test_from_compose_buffers_unattached_stage_bypass_raises() -> None:
    """Freezing captured-but-unattached buckets would silently drop the
    declarative complement — mirror the _write_stages bypass guard."""
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    e.model(ndm=2, ndf=2)
    e.stage_open("s")
    e.analyze(steps=1)
    e.stage_close()
    # NO set_stage_records.
    with pytest.raises(RuntimeError, match="set_stage_records"):
        OpenSeesModel.from_compose_buffers(
            build_two_quad_fem(), e, snapshot_id="",
        )


def test_vanilla_archive_stages_empty_and_flat_build_works(
    tmp_path: Path,
) -> None:
    """Legacy pin: a non-staged archive yields stages() == () and the
    flat tcl build still renders (the truthiness gate keys on it)."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    out = tmp_path / "vanilla.h5"
    ops.h5(str(out))

    m = OpenSeesModel.from_h5(str(out))
    assert m.stages() == ()
    deck = m.build("tcl")
    assert isinstance(deck, str) and "element quad" in deck


# ---------------------------------------------------------------------------
# SSI-2.E ``update_material_stage`` round trip (schema 2.21.0)
# ---------------------------------------------------------------------------


def test_stages_accessor_reads_back_material_stage_flips(
    tmp_path: Path,
) -> None:
    out = tmp_path / "flip.h5"
    ops = build_material_stage_bridge(build_two_quad_fem())
    ops.h5(str(out))

    gravity, push = OpenSeesModel.from_h5(str(out)).stages()
    assert gravity.update_material_stages == ()
    assert push.update_material_stages == (
        (ops.tag_for(ops._names["sand_a"]), 1),
        (ops.tag_for(ops._names["sand_b"]), 1),
    )


def test_material_stage_flip_survives_h5_echo(tmp_path: Path) -> None:
    """``from_h5 -> to_h5 -> from_h5`` keeps the flip (store-and-echo)
    and the deck replay re-emits it."""
    out = tmp_path / "flip.h5"
    echo = tmp_path / "flip_echo.h5"
    build_material_stage_bridge(build_two_quad_fem()).h5(str(out))

    m = OpenSeesModel.from_h5(str(out))
    m.to_h5(str(echo))
    assert _model_hash_of(str(echo)) == _model_hash_of(str(out))

    _, push = OpenSeesModel.from_h5(str(echo)).stages()
    assert len(push.update_material_stages) == 2
    assert {v for _t, v in push.update_material_stages} == {1}
