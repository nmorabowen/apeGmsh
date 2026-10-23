"""#2464: the stage-close route checks run again at emit, on the final model.

A stage close checks only the elements registered so far
(``stage._builder.check_stage_routes``), while the deck emits every element
spec, one registered after the stage included, before that stage's flip
lines.  So an element registered late slipped past each route check: a
FourNodeQuad on a flipped sand emitted a bare ``updateMaterialStage`` with
no sweep, the #2445 wrong answer.  The emit now runs the same checks on
every registered primitive (``stage_validate.validate_stage_routes``).

The oracle is the stage close itself.  Each case is built twice, the same
model with one element registered before the stage (the stage close
refuses it) and after it (the stage close cannot see it).  The late model
must pass the stage close and be refused at emit, on every target, with
the stage close's own message, character for character.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.stage import _builder

from tests.opensees.h5.test_h5_stage_route_replay import (
    _DP,
    _host,
    _md,
    _ops,
    _stage,
)

#: ``late`` registers the case's last element after the stage named here.
Build = Callable[[bool], apeSees]


def _multi_element_on_a_late_quad(late: bool) -> apeSees:
    """#2445: a plastic flip of a sand that a FourNodeQuad also carries."""
    ops = _ops()
    sand = _md(ops, "sand")
    _host(ops, "A", sand)
    if not late:
        _host(ops, "Q", sand)
    _stage(ops, "push", lambda s: s.update_material_stage(
        materials=[sand], stage=1))
    if late:
        _host(ops, "Q", sand)
    return ops


def _late_carrier_left_out(late: bool) -> apeSees:
    """#2445: a materialState flip that misses a second carrier."""
    ops = _ops()
    sand = _md(ops, "sand")
    _host(ops, "A", sand)
    if not late:
        _host(ops, "B", sand)
    _stage(ops, "push", lambda s: s.update_parameter(
        "materialState", 1.0, pg="A", material=sand))
    if late:
        _host(ops, "B", sand)
    return ops


def _late_tag_blind_neighbour(late: bool) -> apeSees:
    """#2458: a materialState over a DruckerPrager element."""
    ops = _ops()
    sand = _md(ops, "sand")
    clay = ops.nDMaterial.DruckerPrager(name="clay", **_DP)
    _host(ops, "A", sand)
    if not late:
        _host(ops, "B", clay)
    _stage(ops, "push", lambda s: s.update_parameter(
        "materialState", 1.0, elements=[1, 2], material=sand))
    if late:
        _host(ops, "B", clay)
    return ops


def _late_stress_density_beside_the_flip(late: bool) -> apeSees:
    """#2458: an updateMaterialStage beside a class that takes it whatever
    the tag.  No bridge primitive carries that class yet, so a stand-in of
    its name does (as in ``test_tag_blind_stage_parameter.py``)."""
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    ops = _ops()
    sand = _md(ops, "sand")
    stand_in = type("StressDensityModel", (ElasticIsotropic,), {})
    sdm = ops.register(stand_in(E=1e5, nu=0.3))
    _host(ops, "A", sand)
    if not late:
        ops.element.SixNodeTri(pg="B", thickness=1.0, material=sdm)
    _stage(ops, "push", lambda s: s.update_material_stage(
        materials=[sand], stage=1))
    if late:
        ops.element.SixNodeTri(pg="B", thickness=1.0, material=sdm)
    return ops


def _materialstate_through_a_late_quad(late: bool) -> apeSees:
    """#2460: a materialState whose targets include a FourNodeQuad carrier.
    Stage 0, so the multi-element plastic check does not answer first."""
    ops = _ops()
    sand = _md(ops, "sand")
    _host(ops, "A", sand)
    if not late:
        _host(ops, "P", sand)
    _stage(ops, "push", lambda s: s.update_parameter(
        "materialState", 0.0, elements=[1, 5], material=sand))
    if late:
        _host(ops, "P", sand)
    return ops


def _flip_then_activate_a_late_carrier(
    late: bool, *, own_material: bool = False, mode: "str | None" = None,
    system: str = "UmfPack",
) -> apeSees:
    """Sand on ``A`` flipped plastic, then ``C`` activated: #2445 when ``C``
    carries the same sand, the fork constructor reset (#2443) when it
    carries its own ManzariDafalias on a fork-pinned bridge.  Late, ``C``'s
    element is registered after the activating stage closes."""
    ops = _ops(mode)
    sand = _md(ops, "sand")
    on_c = _md(ops, "sand_c") if own_material else sand
    _host(ops, "A", sand)
    if not late:
        _host(ops, "C", on_c)
    _stage(ops, "plastic", lambda s: s.update_material_stage(
        materials=[sand], stage=1))
    _stage(ops, "install", lambda s: s.activate(pgs=["C"]), system=system)
    if late:
        _host(ops, "C", on_c)
    return ops


CASES: dict[str, tuple[Build, str]] = {
    "multi-element flip on a late FourNodeQuad": (
        _multi_element_on_a_late_quad, "but FourNodeQuad carries it"),
    "materialState leaves a late carrier out": (
        _late_carrier_left_out, "leaves 1 of those elements out"),
    "late tag-blind neighbour": (
        _late_tag_blind_neighbour,
        "carry DruckerPrager (nDMaterial tag 2), whose setParameter takes "
        "materialState without checking the tag"),
    "late stress-density class beside updateMaterialStage": (
        _late_stress_density_beside_the_flip,
        "SixNodeTri on pg 'B' (StressDensityModel) takes "
        "updateMaterialStage without checking the material tag"),
    "materialState through a late FourNodeQuad": (
        _materialstate_through_a_late_quad,
        "targets FourNodeQuad elements, which are not measured"),
    "activation of a late carrier after the plastic flip": (
        _flip_then_activate_a_late_carrier,
        "SixNodeTri on pg 'C' carries nDMaterial tag 1, whose stage flag "
        "an earlier stage left plastic"),
    "fork constructor reset by a late carrier": (
        lambda late: _flip_then_activate_a_late_carrier(
            late, own_material=True, mode="fork", system="Pardiso"),
        "reset the class-wide C++ static ManzariDafalias::mElastFlag to 0"),
}


def _quiet(fn: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _stage_close_message(build: Build) -> str:
    with pytest.raises(BridgeError) as exc:
        _quiet(lambda: build(False))
    return str(exc.value)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_early_twin_is_refused_at_stage_close(case: str) -> None:
    build, phrase = CASES[case]
    assert phrase in _stage_close_message(build)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_late_model_passes_the_stage_close(case: str) -> None:
    """The gap: every stage closes, so only the emit can refuse it."""
    build, _phrase = CASES[case]
    ops = _quiet(lambda: build(True))
    assert ops._stage_records


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("target", ["tcl", "py", "h5"])
def test_the_late_model_is_refused_at_emit_with_the_stage_close_message(
    case: str, target: str, tmp_path: Path,
) -> None:
    build, phrase = CASES[case]
    expected = _stage_close_message(build)
    ops = _quiet(lambda: build(True))
    out = tmp_path / f"model.{target}"
    with pytest.raises(BridgeError) as exc:
        _quiet(lambda: getattr(ops, target)(str(out)))
    assert str(exc.value) == expected
    assert phrase in str(exc.value)
    assert not out.exists()


def test_the_emit_check_reads_the_model_the_stage_close_reads() -> None:
    """Parity of the two models: the emit's ``RouteModel`` of the final
    primitives equals, spec for spec, the stage close's of the same
    bridge."""
    from apeGmsh.opensees._internal.build.stage_validate import (
        validate_stage_routes,
    )
    from tests.opensees.h5.test_h5_stage_route_replay import _facts

    seen: list[Any] = []

    def capture(name: str, model: Any, **_kw: Any) -> None:
        seen.append(model)

    for case in sorted(CASES):
        build, _phrase = CASES[case]
        ops = _quiet(lambda: build(True))
        built = ops.build()
        with pytest.MonkeyPatch.context() as m:
            m.setattr(_builder, "check_stage_routes", capture)
            validate_stage_routes(
                stage_records=built.stage_records,
                primitives=built.primitives, tag_for=built.tag_for,
                fem=built.fem, fork=lambda: built.fork_pinned)
        live = _builder._StageBuilder(ops, "probe")._route_model()
        assert seen, case
        for model in seen:
            assert _facts(model) == _facts(live), case
            assert dict(model.nd_class) == dict(live.nd_class), case
            assert model.fork == live.fork, case
        seen.clear()


def test_a_late_carrier_the_sweep_reaches_passes() -> None:
    """No false refusal: a second SixNodeTri on the flipped sand,
    registered after the stage, is swept with the first one."""
    ops = _ops()
    sand = _md(ops, "sand")
    _host(ops, "A", sand)
    _stage(ops, "push", lambda s: s.update_material_stage(
        materials=[sand], stage=1))
    _host(ops, "B", sand)
    tcl = TclEmitter()
    _quiet(lambda: ops.build().emit(tcl))
    lines = tcl.lines()
    tri = [ln.split()[2] for ln in lines if ln.startswith("element tri6n")]
    sweep = [ln for ln in lines if ln.startswith("setParameter")]
    assert len(tri) == 2 and len(sweep) == 1
    listed = sweep[0].split("-ele ")[1].split(" materialState")[0].split()
    assert sorted(listed) == sorted(tri)


# ---------------------------------------------------------------------------
# The fork signal: the emit knows the fork when the replay does
# ---------------------------------------------------------------------------


def _fork_reset(late: bool, mode: "str | None") -> apeSees:
    """The fork-reset model on a fork-only chain (``system Pardiso``)."""
    return _flip_then_activate_a_late_carrier(
        late, own_material=True, mode=mode, system="Pardiso")


@pytest.mark.parametrize("target", ["tcl", "py", "h5"])
@pytest.mark.parametrize("late", [False, True], ids=["early", "late"])
@pytest.mark.parametrize("mode", [None, "auto"])
def test_a_fork_only_chain_is_known_to_run_the_fork(
    mode: "str | None", late: bool, target: str, tmp_path: Path,
) -> None:
    """An unpinned bridge whose stage chain declares a fork-only type
    writes a deck that runs only on the fork, and its archive's replay
    reads ``"fork"`` off that chain.  So the emit refuses the fork reset
    (#2443) as the replay does, with the fork-pinned stage close's
    message.  The stage close cannot see the chain's binary, so it passes
    both orders."""
    expected = _stage_close_message(lambda lt: _fork_reset(lt, "fork"))
    ops = _quiet(lambda: _fork_reset(late, mode))
    out = tmp_path / f"model.{target}"
    with pytest.raises(BridgeError) as exc:
        _quiet(lambda: getattr(ops, target)(str(out)))
    assert str(exc.value) == expected
    assert "#2444" in expected
    assert not out.exists()


@pytest.mark.parametrize("system", ["Pardiso", "UmfPack"])
def test_the_emit_reads_the_fork_off_the_chain_as_the_replay_does(
    system: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Oracle of the fork signal: ``stage_chains_need_fork`` on the built
    model equals the ``"fork"`` the archive's replay requires.  The
    archive is written with the route checks off, so the refused model
    can be written at all."""
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.opensees._internal.build.stage_validate import (
        stage_chains_need_fork,
    )
    from tests.opensees.h5.test_h5_stage_route_replay import _write

    def build() -> apeSees:
        return _flip_then_activate_a_late_carrier(
            True, own_material=True, system=system)

    built = _quiet(lambda: build().build())
    emit = stage_chains_need_fork(
        built.stage_records, tag_for=built.tag_for,
        global_chain=lambda em: built._maybe_auto_emit_constraint_handler(
            em, list(built.primitives)))
    path = _write(build, tmp_path / "model.h5", monkeypatch, pre_fix=True)
    replay = "fork" in OpenSeesModel.from_h5(str(path)).replay_requires()
    assert emit == replay == (system == "Pardiso")


def test_a_stock_chain_on_an_unpinned_bridge_passes() -> None:
    """The same model on a stock chain is not known to run the fork, at
    the emit as at the replay (``own_material_on_stock`` there)."""
    ops = _quiet(lambda: _flip_then_activate_a_late_carrier(
        True, own_material=True))
    _quiet(lambda: ops.build().emit(TclEmitter()))


# ---------------------------------------------------------------------------
# The partitioned emit runs the same check
# ---------------------------------------------------------------------------


def _partitioned(late: bool) -> apeSees:
    """:func:`_multi_element_on_a_late_quad` on two partitions."""
    from tests.opensees.fixtures.fem_stub import (
        _PartitionSetStub,
        _PartitionStub,
    )

    ops = _multi_element_on_a_late_quad(late)
    nodes = {1: range(1, 7), 2: range(7, 13), 3: range(13, 17),
             4: range(17, 21), 5: range(21, 25), 6: range(25, 31)}
    object.__setattr__(ops.fem, "partitions", _PartitionSetStub([
        _PartitionStub(id=k, element_ids=tuple(eles), node_ids=tuple(
            sorted({n for e in eles for n in nodes[e]})))
        for k, eles in enumerate(([1, 3], [2, 4, 5, 6]))]))
    return ops


PARTITIONED: dict[str, Callable[[apeSees, Path], Any]] = {
    "tcl": lambda o, d: o.tcl(str(d / "m.tcl")),
    "py": lambda o, d: o.py(str(d / "m.py")),
    "h5": lambda o, d: o.h5(str(d / "m.h5")),
    "tcl per_rank": lambda o, d: o.tcl(str(d / "m.tcl"), per_rank=True),
    "tcl per_rank stream": lambda o, d: o.tcl(
        str(d / "m.tcl"), per_rank=True, stream=True),
}


@pytest.mark.parametrize("target", sorted(PARTITIONED))
def test_the_partitioned_emit_refuses_a_late_carrier(
    target: str, tmp_path: Path,
) -> None:
    from apeGmsh.opensees._internal.build.ownership import is_partitioned

    expected = _stage_close_message(_partitioned)
    ops = _quiet(lambda: _partitioned(True))
    assert is_partitioned(ops.fem)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(BridgeError) as exc:
        _quiet(lambda: PARTITIONED[target](ops, out))
    assert str(exc.value) == expected
    assert "but FourNodeQuad carries it" in expected
    assert not list(out.iterdir())
