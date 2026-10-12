"""K1-8 (#1465, opensees 2.27.0): the archive stores every parameter tag.

The T ruling on #1465 (option A):

- an archive at or above 2.27.0 stores the tags the build planned, per
  record: ``initial_stress/stress_NNN@param_tags`` (global and stage), the
  ``(rank, pid)`` rows of each ``activate_absorbing/absorb_NNN`` and of each
  ``update_parameter/update_NNN``. Its replay reads them and never re-mints:
  the planted-fallback tests make both re-mint entries raise and show such an
  archive still replays to the bridge's own deck, unmasked;
- an archive below 2.27.0 keeps re-minting through the waived entries,
  behind the version check in ``compose.py``: the same planted fallback
  fails it, and unplanted it replays to the bridge's deck;
- the former refusals archive: a stage's ``s.update_parameter`` and
  ``s.zero_velocities`` round-trip, and their decks equal the bridge's.

A pre-2.27 archive is made the way a 2.26 writer left one: the tags
stripped and the opensees stamp set one minor below ``PARAM_TAGS_FROM``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel
from apeGmsh.opensees._internal.build import stage_emit as build_mod
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.h5 import H5Emitter

from tests.fixtures.schema import OPENSEES_PARAM_TAGS_FROM
from tests.opensees.h5.test_h5_initial_stress import _build_frame
from tests.opensees.h5.test_h5_stages_reader import (
    _chain,
    _real_kitchen_sink_bridge,
    _real_two_stage_bridge,
    build_two_quad_fem,
)
from tests.opensees.h5.test_h5_stages_replay import _strip_comments


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _update_then_stress_bridge() -> apeSees:
    """Stage 1 updates a parameter (tag 1); stage 2 ramps an initial stress
    (tags 2-4), flips the absorbing elements (tag 5) and zeroes the
    velocities. A replay that re-minted the stage-2 tags from a fresh
    allocator would number them 1-4, since no re-mint entry knows the
    update's tag: only the stored tags give the bridge's deck back."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="soften") as s:
        s.update_parameter("E", 2.0e6, pg="Rock", material=mat)
        s.analysis(**_chain(ops))
        s.run(n_increments=2)
    with ops.stage(name="shake") as s:
        s.activate(pgs=["Fill"])
        s.initial_stress(
            name="insitu", pg="Rock",
            sigma_xx=-1.0e3, sigma_yy=-1.0e3, sigma_zz=-2.0e3,
            ramp_steps=4,
        )
        s.activate_absorbing(pg="Fill")
        s.zero_velocities()
        s.analysis(**_chain(ops))
        s.run(n_increments=3, dt=0.01)
    return ops


def _flat_quad_initial_stress_bridge() -> apeSees:
    """A flat (unstaged) model with two global initial stresses (tags 1-3,
    4-6), a load pattern and an analysis: every line of its deck replays."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    ops.initial_stress(
        name="rock", pg="Rock",
        sigma_xx=-1.0e3, sigma_yy=-1.0e3, sigma_zz=-2.0e3, ramp_steps=4,
    )
    ops.initial_stress(
        name="fill", pg="Fill",
        sigma_xx=-5.0e2, sigma_yy=-5.0e2, sigma_zz=-1.0e3, ramp_steps=4,
    )
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="FillTop", forces=(1.0, 0.0))
    # No global chain: a flat replay emits it after the patterns, which
    # the bridge does not (a category-order difference K1-8 does not own).
    return ops


def _zero_some_bridge() -> apeSees:
    """One stage zeroing two named nodes (the explicit ``nodes=`` form)."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="static") as s:
        s.zero_velocities(nodes=[3, 4])
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    return ops


def _bridge_deck(factory: Callable[[], apeSees], tmp_path: Path) -> str:
    p = tmp_path / "bridge.tcl"
    factory().tcl(str(p), progress=False)
    return p.read_text(encoding="utf-8")


def _archive(factory: Callable[[], apeSees], tmp_path: Path) -> Path:
    p = tmp_path / "model.h5"
    factory().h5(str(p))
    return p


def _replayed(path: Path) -> str:
    deck = OpenSeesModel.from_h5(str(path)).build("tcl")
    assert isinstance(deck, str)
    return deck


def _pre_param_tags_version() -> str:
    """The minor just below ``PARAM_TAGS_FROM``: what a 2.26 writer stamped."""
    major, minor, _patch = (int(p) for p in OPENSEES_PARAM_TAGS_FROM.split("."))
    return f"{major}.{minor - 1}.0"


def downgrade_to_pre_param_tags(path: Path) -> None:
    """Make ``path`` the archive a pre-2.27 writer would have left: no
    ``param_tags`` anywhere, the opensees stamp one minor below
    ``PARAM_TAGS_FROM``. Refuses an archive holding a record that writer
    refused (an ``s.update_parameter`` or ``s.zero_velocities``)."""
    with h5py.File(path, "r+") as f:
        attrs: list[Any] = []
        datasets: list[Any] = []

        def visit(name: str, obj: Any) -> None:
            assert not name.endswith(("update_parameter", "zero_velocities")), (
                f"{name}: a pre-2.27 writer refused this verb")
            if "param_tags" in obj.attrs:
                attrs.append(obj)
            if isinstance(obj, h5py.Group) and isinstance(
                    obj.get("param_tags"), h5py.Dataset):
                datasets.append(obj)

        f["opensees"].visititems(visit)
        assert attrs or datasets, "the archive stores no param_tags"
        for obj in attrs:
            del obj.attrs["param_tags"]
        for obj in datasets:
            del obj["param_tags"]
        f["meta"].attrs["opensees_schema_version"] = _pre_param_tags_version()


@pytest.fixture
def planted_fallback(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Both parameter re-mint entries raise; returns the calls made."""
    calls: list[str] = []

    def planted(name: str) -> Callable[..., Any]:
        def fail(*_a: Any, **_k: Any) -> Any:
            calls.append(name)
            raise AssertionError(f"planted fallback: {name} re-minted")
        return fail

    for name in ("replay_initial_stress_global", "replay_activate_absorbing"):
        monkeypatch.setattr(build_mod, name, planted(name))
    return calls


_PARAM_FIXTURES = [
    pytest.param(_flat_quad_initial_stress_bridge, id="flat_initial_stress"),
    pytest.param(_real_two_stage_bridge, id="staged_initial_stress"),
    pytest.param(_real_kitchen_sink_bridge, id="staged_absorbing"),
    pytest.param(_update_then_stress_bridge, id="update_then_stress"),
]


# ---------------------------------------------------------------------------
# 1. The writer stores the tags the build planned
# ---------------------------------------------------------------------------


def test_flat_archive_stores_the_global_ramp_tags(tmp_path: Path) -> None:
    p = _archive(_flat_quad_initial_stress_bridge, tmp_path)
    with h5py.File(p) as f:
        grp = f["opensees/initial_stress"]
        assert grp["stress_000"].attrs["param_tags"].tolist() == [1, 2, 3]
        assert grp["stress_001"].attrs["param_tags"].tolist() == [4, 5, 6]
    model = OpenSeesModel.from_h5(str(p))
    assert model._initial_stress_param_tags == ((1, 2, 3), (4, 5, 6))
    # The frame of the original Phase-1 tests stores them too.
    (tmp_path / "frame").mkdir()
    p2 = _archive(lambda: _build_frame(with_initial_stress=True),
                  tmp_path / "frame")
    with h5py.File(p2) as f:
        assert f["opensees/initial_stress/stress_000"].attrs[
            "param_tags"].tolist() == [1, 2, 3]


def test_staged_archive_stores_every_record_tag(tmp_path: Path) -> None:
    p = _archive(_update_then_stress_bridge, tmp_path)
    with h5py.File(p) as f:
        s0, s1 = f["opensees/stages/stage_000"], f["opensees/stages/stage_001"]
        upd = s0["update_parameter/update_000"]
        assert upd["param_tags"][:].tolist() == [[-1, 1]]
        assert (upd.attrs["name"], float(upd.attrs["value"]),
                int(upd.attrs["mat_tag"]), upd.attrs["pg"]) == (
            "E", 2.0e6, 1, "Rock")
        stress = s1["initial_stress/stress_000"]
        assert stress.attrs["param_tags"].tolist() == [2, 3, 4]
        absorb = s1["activate_absorbing/absorb_000"]
        assert absorb["param_tags"][:].tolist() == [[-1, 5]]
        zero = s1["zero_velocities"]
        assert list(zero) == ["zero_000"] and "nodes" not in zero["zero_000"]
    (st0, st1) = OpenSeesModel.from_h5(str(p)).stages()
    assert st0.update_parameter_param_tags == (((-1, 1),),)
    assert st1.initial_stress_param_tags == ((2, 3, 4),)
    assert st1.absorbing_param_tags == (((-1, 5),),)
    assert [z.nodes for z in st1.zero_velocities] == [None]


def test_explicit_zero_velocity_nodes_round_trip(tmp_path: Path) -> None:
    p = _archive(_zero_some_bridge, tmp_path)
    with h5py.File(p) as f:
        nodes = f["opensees/stages/stage_000/zero_velocities/zero_000/nodes"]
        assert nodes[:].tolist() == [3, 4]
    (st,) = OpenSeesModel.from_h5(str(p)).stages()
    assert [z.nodes for z in st.zero_velocities] == [(3, 4)]


# ---------------------------------------------------------------------------
# 2. A >= 2.27 archive replays its stored tags and never re-mints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _PARAM_FIXTURES)
def test_archive_replays_the_bridge_deck_unmasked(
    factory: Callable[[], apeSees], tmp_path: Path,
    planted_fallback: list[str],
) -> None:
    """``tcl() == from_h5(h5()).build('tcl')`` with no parameter-tag mask,
    and with both re-mint entries planted to raise."""
    bridge = _strip_comments(_bridge_deck(factory, tmp_path))
    replayed = _strip_comments(_replayed(_archive(factory, tmp_path)))
    assert replayed == bridge
    assert planted_fallback == []


def test_update_then_stress_shows_why_the_tags_are_stored(
    tmp_path: Path,
) -> None:
    """The bridge numbers stage 2's ramps 2-4 and its flip 5, after the
    stage-1 update's 1; the replay writes exactly those."""
    deck = _replayed(_archive(_update_then_stress_bridge, tmp_path))
    assert "updateParameter 1 2000000.0" in deck or (
        "updateParameter 1 2000000" in deck)
    for tag in (2, 3, 4, 5):
        assert f"parameter {tag}" in deck


def test_zero_velocities_replay_after_reset_before_analyze(
    tmp_path: Path,
) -> None:
    bridge = _strip_comments(_bridge_deck(_zero_some_bridge, tmp_path))
    replayed = _strip_comments(_replayed(_archive(_zero_some_bridge, tmp_path)))
    assert replayed == bridge
    assert sum("setNodeVel" in ln for ln in replayed) == 4
    assert sum("setNodeAccel" in ln for ln in replayed) == 4


# ---------------------------------------------------------------------------
# 3. Both sides of the version check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _PARAM_FIXTURES[:3])
def test_pre_2_27_archive_re_mints_through_the_waived_entries(
    factory: Callable[[], apeSees], tmp_path: Path,
    planted_fallback: list[str],
) -> None:
    """Below 2.27.0 the replay takes the waived re-mint path: planted, it
    raises there."""
    p = _archive(factory, tmp_path)
    downgrade_to_pre_param_tags(p)
    with pytest.raises(AssertionError, match="planted fallback"):
        _replayed(p)
    assert planted_fallback


@pytest.mark.parametrize("factory", _PARAM_FIXTURES[:3])
def test_pre_2_27_archive_still_replays_the_bridge_deck(
    factory: Callable[[], apeSees], tmp_path: Path,
) -> None:
    bridge = _strip_comments(_bridge_deck(factory, tmp_path))
    p = _archive(factory, tmp_path)
    downgrade_to_pre_param_tags(p)
    assert _strip_comments(_replayed(p)) == bridge


def test_a_2_27_archive_missing_its_tags_refuses(tmp_path: Path) -> None:
    """Stripping the tags but keeping the 2.27 stamp is a malformed archive,
    not a pre-2.27 one: the replay refuses rather than re-mint."""
    p = _archive(_real_two_stage_bridge, tmp_path)
    with h5py.File(p, "r+") as f:
        del f["opensees/stages/stage_001/initial_stress/stress_000"].attrs[
            "param_tags"]
    with pytest.raises(ValueError, match="stored param_tags"):
        _replayed(p)


def test_replay_without_a_version_refuses() -> None:
    """A caller replaying a parameter-bearing record without saying which
    archive it came from refuses rather than guess."""
    from apeGmsh.opensees._internal.compose import _archives_param_tags

    with pytest.raises(ValueError, match="opensees_version"):
        _archives_param_tags(None)


# ---------------------------------------------------------------------------
# 4. The rewrite (to_h5) stores the tags
# ---------------------------------------------------------------------------


def test_rewrite_of_a_2_27_archive_is_hash_stable(tmp_path: Path) -> None:
    from apeGmsh.opensees._internal.lineage import compute_model_hash

    src = _archive(_update_then_stress_bridge, tmp_path)
    out = tmp_path / "rewrite.h5"
    OpenSeesModel.from_h5(str(src)).to_h5(str(out))
    with h5py.File(src) as a, h5py.File(out) as b:
        assert compute_model_hash("", a["opensees"]) == compute_model_hash(
            "", b["opensees"])


def test_rewrite_of_a_pre_2_27_archive_stores_the_re_minted_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rewrite stamps 2.27.0, so it stores the tags the old archive's
    replay mints; its own replay then mints nothing."""
    src = _archive(_real_kitchen_sink_bridge, tmp_path)
    with h5py.File(src) as f:
        stored = f["opensees/stages/stage_000/activate_absorbing/absorb_000/"
                   "param_tags"][:].tolist()
    downgrade_to_pre_param_tags(src)
    out = tmp_path / "rewrite.h5"
    OpenSeesModel.from_h5(str(src)).to_h5(str(out))
    with h5py.File(out) as f:
        assert f["opensees/stages/stage_000/activate_absorbing/absorb_000/"
                 "param_tags"][:].tolist() == stored
    for name in ("replay_initial_stress_global", "replay_activate_absorbing"):
        monkeypatch.setattr(build_mod, name, _raise)
    bridge = _strip_comments(_bridge_deck(_real_kitchen_sink_bridge, tmp_path))
    assert _strip_comments(_replayed(out)) == bridge


def _raise(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("planted fallback")


# ---------------------------------------------------------------------------
# 5. The writer refuses what it cannot attribute
# ---------------------------------------------------------------------------


def _stage_record(**kw: Any) -> Any:
    from types import SimpleNamespace

    base: dict[str, Any] = dict(
        name="s1", n_increments=1, dt=None, activated_pgs=(),
        initial_stress_records=(), activate_absorbing_records=(),
        update_parameter_records=(), zero_velocity_records=(), profile=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _one_stage(e: H5Emitter, body: Callable[[H5Emitter], None]) -> None:
    e.stage_open("s1")
    body(e)
    e.analyze(steps=1)
    e.stage_close()


def test_a_record_without_a_flip_refuses() -> None:
    """Two absorbing records but one flip tag (a record whose target holds
    no element): the tag cannot be attributed, so the archive refuses."""
    from apeGmsh.opensees._internal.build.records import (
        ActivateAbsorbingRecord,
    )

    e = H5Emitter()
    _one_stage(e, lambda e: e.flip_element_stage(7, (1,)))
    recs = (ActivateAbsorbingRecord(pg="A", elements=None),
            ActivateAbsorbingRecord(pg="B", elements=None))
    with pytest.raises(RuntimeError) as info:
        e.set_stage_records([_stage_record(activate_absorbing_records=recs)])
    msg = str(info.value)
    # The message names the stage, the verb and the targets.
    assert "stage 's1'" in msg and "s.activate_absorbing" in msg
    assert "pg='A'" in msg and "pg='B'" in msg
    assert "holds no element" in msg


def test_an_update_record_without_a_call_names_its_target() -> None:
    from apeGmsh.opensees._internal.build.records import UpdateParameterRecord

    e = H5Emitter()
    _one_stage(e, lambda e: None)
    rec = UpdateParameterRecord(
        name="xPerm", value=1.0, pg=None, elements=(9,), mat_tag=None)
    with pytest.raises(RuntimeError) as info:
        e.set_stage_records([_stage_record(update_parameter_records=(rec,))])
    msg = str(info.value)
    assert "stage 's1'" in msg and "s.update_parameter" in msg
    assert "elements=[9]" in msg


def test_partitioned_flips_keep_every_rank_row() -> None:
    """A partitioned emit writes a record's tag once per rank holding its
    elements; the record stores every ``(rank, pid)`` row, ascending tags in
    record order."""
    from apeGmsh.opensees._internal.build.records import (
        ActivateAbsorbingRecord,
    )

    def body(e: H5Emitter) -> None:
        for rank, calls in ((0, [(4, (1,)), (5, (2,))]), (1, [(5, (3,))])):
            e.partition_open(rank)
            for pid, eles in calls:
                e.flip_element_stage(pid, eles)
            e.partition_close()

    e = H5Emitter()
    _one_stage(e, body)
    recs = (ActivateAbsorbingRecord(pg="A", elements=None),
            ActivateAbsorbingRecord(pg="B", elements=None))
    e.set_stage_records([_stage_record(activate_absorbing_records=recs)])
    (blk,) = e._stage_blocks
    assert blk.absorbing_param_tags == (((0, 4),), ((0, 5), (1, 5)))


def test_an_update_call_that_disagrees_with_its_record_refuses() -> None:
    from apeGmsh.opensees._internal.build.records import UpdateParameterRecord

    e = H5Emitter()
    _one_stage(e, lambda e: e.update_parameter(3, (1,), ("E", 1), 9.0))
    rec = UpdateParameterRecord(
        name="E", value=2.0, pg="Rock", elements=None, mat_tag=1)
    with pytest.raises(RuntimeError, match="diverged"):
        e.set_stage_records([_stage_record(update_parameter_records=(rec,))])


def test_zeroing_without_a_record_refuses() -> None:
    e = H5Emitter()
    _one_stage(e, lambda e: e.set_node_vel(1, 1, 0.0))
    with pytest.raises(RuntimeError, match="zero_velocities"):
        e.set_stage_records([_stage_record()])


def test_a_nonzero_node_velocity_refuses() -> None:
    e = H5Emitter()
    e.stage_open("s1")
    with pytest.raises(RuntimeError, match="nonzero"):
        e.set_node_vel(1, 1, 0.5)


def test_a_ramp_with_no_record_refuses() -> None:
    e = H5Emitter()
    e.step_hook_ramp(
        "ghost", targets=((1, -1.0), (2, -1.0), (3, -1.0)),
        n_steps_to_full=1.0,
    )
    with pytest.raises(RuntimeError, match="step_hook_ramp calls named"):
        e.set_initial_stress_records([])


def test_the_absorbing_rows_dtype_is_int64(tmp_path: Path) -> None:
    p = _archive(_real_kitchen_sink_bridge, tmp_path)
    with h5py.File(p) as f:
        ds = f["opensees/stages/stage_000/activate_absorbing/absorb_000/"
               "param_tags"]
        assert ds.dtype == np.int64 and ds.shape == (1, 2)


# ---------------------------------------------------------------------------
# 6. A pre-2.27 archive with an empty-target absorbing flip still re-saves
# ---------------------------------------------------------------------------


def _absorb_and_stress_bridge() -> apeSees:
    """One stage: a stage initial stress and an absorbing flip on ``Fill``."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="shake") as s:
        s.initial_stress(
            name="insitu", pg="Rock",
            sigma_xx=-1.0e3, sigma_yy=-1.0e3, sigma_zz=-2.0e3, ramp_steps=4,
        )
        s.activate_absorbing(pg="Fill")
        s.analysis(**_chain(ops))
        s.run(n_increments=2, dt=0.01)
    return ops


def _prepend_empty_absorbing(path: Path) -> None:
    """Give the archive's stage an ``s.activate_absorbing(elements=[])``
    record ahead of its ``pg="Fill"`` one, as a pre-2.27 writer archived
    it (an empty ``elements`` dataset; the declaration rows column grows
    with it, sharing the existing row)."""
    with h5py.File(path, "r+") as f:
        absorb = f["opensees/stages/stage_000/activate_absorbing"]
        absorb.move("absorb_000", "absorb_001")
        g = absorb.create_group("absorb_000")
        g.create_dataset("elements", data=np.zeros(0, dtype=np.int64))
        key = "opensees/decls/rows/stages/stage_000/activate_absorbing"
        if key in f:
            col = f[key][()]
            del f[key]
            f.create_dataset(key, data=np.concatenate([col[:1], col]))


def test_pre_2_27_empty_target_absorbing_archive_re_saves(
    tmp_path: Path,
) -> None:
    """Main archived an empty-target flip record; such an old file must
    keep re-saving (the T ruling on #1465). The rewrite stores an empty run
    for that record and the flip's tag for the other, and its deck equals
    the old archive's own replay and the bridge's deck."""
    bridge = _strip_comments(_bridge_deck(_absorb_and_stress_bridge, tmp_path))
    src = _archive(_absorb_and_stress_bridge, tmp_path)
    downgrade_to_pre_param_tags(src)
    _prepend_empty_absorbing(src)
    old = OpenSeesModel.from_h5(str(src))
    assert [e for _pg, e in old.stages()[0].activate_absorbing] == [(), None]
    old_deck = _strip_comments(old.build("tcl"))
    assert old_deck == bridge

    out = tmp_path / "rewrite.h5"
    old.to_h5(str(out))
    with h5py.File(out) as f:
        absorb = f["opensees/stages/stage_000/activate_absorbing"]
        assert absorb["absorb_000/param_tags"].shape == (0, 2)
        assert absorb["absorb_001/param_tags"][:].tolist() == [[-1, 4]]
        assert f["opensees/stages/stage_000/initial_stress/stress_000"].attrs[
            "param_tags"].tolist() == [1, 2, 3]
    new = OpenSeesModel.from_h5(str(out))
    assert new.stages()[0].absorbing_param_tags == ((), ((-1, 4),))
    assert _strip_comments(new.build("tcl")) == bridge
