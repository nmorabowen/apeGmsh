"""The stage-close route checks run again at emit, on the final model.

A stage close checks only the elements registered so far
(``stage._builder.check_stage_routes``), while the deck emits every element
spec, one registered after the stage included, before that stage's flip
lines.  So an element registered late slipped past each route check: a
FourNodeQuad on a flipped sand emitted a bare ``updateMaterialStage`` with
no sweep, the wrong answer of #2445.  The emit now runs the same checks on
every registered primitive (``stage_validate.validate_stage_routes``).

The oracle is the stage close itself.  Each case is built twice, the same
model with one element registered before the stage (the stage close
refuses it) and after it (the stage close cannot see it).  The late model
must pass the stage close and be refused at emit, on every target, flat
and partitioned, with the stage close's own message, character for
character.

#2528 then lifts a false refusal: a flip counts only the carriers alive
in the domain when it runs (``RouteSpec.alive_at``), so a carrier that a
later stage activates is that stage's to reach, and the emit reads each
spec's true owning stage.  The oracles are the deck itself (each flip
reaches exactly the carriers alive at it), the ownership pass
(``compute_stage_ownership``), and, live, the per-element reference.
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


def _flip_then_activate_a_late_carrier(late: bool) -> apeSees:
    """#2445: sand on ``A`` flipped plastic, then ``C``, which carries the
    same sand, activated.  Late, ``C``'s element is registered after the
    activating stage closes."""
    ops = _ops()
    sand = _md(ops, "sand")
    _host(ops, "A", sand)
    if not late:
        _host(ops, "C", sand)
    _stage(ops, "plastic", lambda s: s.update_material_stage(
        materials=[sand], stage=1))
    _stage(ops, "install", lambda s: s.activate(pgs=["C"]))
    if late:
        _host(ops, "C", sand)
    return ops


def _flip_then_activate_a_late_second_material(late: bool) -> apeSees:
    """#2534: sand on ``A`` flipped plastic, then ``C``, which carries a
    second ManzariDafalias the flip never named, activated.  The flag is
    the class's, so ``C`` enters plastic with no Elastic2Plastic(), as a
    carrier of the sand itself does."""
    ops = _ops()
    sand = _md(ops, "sand")
    sand_c = _md(ops, "sand_c")
    _host(ops, "A", sand)
    if not late:
        _host(ops, "C", sand_c)
    _stage(ops, "plastic", lambda s: s.update_material_stage(
        materials=[sand], stage=1))
    _stage(ops, "install", lambda s: s.activate(pgs=["C"]))
    if late:
        _host(ops, "C", sand_c)
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
    "activation of a late second material after the plastic flip": (
        _flip_then_activate_a_late_second_material,
        "SixNodeTri on pg 'C' carries nDMaterial tag 2, whose stage flag "
        "an earlier stage left plastic (ManzariDafalias::mElastFlag"),
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
                fem=built.fem)
        live = _builder._StageBuilder(ops, "probe")._route_model()
        assert seen, case
        for model in seen:
            assert _facts(model) == _facts(live), case
            assert dict(model.nd_class) == dict(live.nd_class), case
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
# The partitioned emit runs the same check
# ---------------------------------------------------------------------------


def _partitioned(build: Build) -> Build:
    """``build`` on two partitions."""
    from tests.opensees.fixtures.fem_stub import (
        _PartitionSetStub,
        _PartitionStub,
    )

    def on_two(late: bool) -> apeSees:
        ops = build(late)
        nodes = {1: range(1, 7), 2: range(7, 13), 3: range(13, 17),
                 4: range(17, 21), 5: range(21, 25), 6: range(25, 31)}
        object.__setattr__(ops.fem, "partitions", _PartitionSetStub([
            _PartitionStub(id=k, element_ids=tuple(eles), node_ids=tuple(
                sorted({n for e in eles for n in nodes[e]})))
            for k, eles in enumerate(([1, 3], [2, 4, 5, 6]))]))
        return ops
    return on_two


PARTITIONED: dict[str, Callable[[apeSees, Path], Any]] = {
    "tcl": lambda o, d: o.tcl(str(d / "m.tcl")),
    "py": lambda o, d: o.py(str(d / "m.py")),
    "h5": lambda o, d: o.h5(str(d / "m.h5")),
    "tcl per_rank": lambda o, d: o.tcl(str(d / "m.tcl"), per_rank=True),
    "tcl per_rank stream": lambda o, d: o.tcl(
        str(d / "m.tcl"), per_rank=True, stream=True),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("target", sorted(PARTITIONED))
def test_the_partitioned_emit_refuses_a_late_carrier(
    target: str, case: str, tmp_path: Path,
) -> None:
    from apeGmsh.opensees._internal.build.ownership import is_partitioned

    build, phrase = CASES[case]
    expected = _stage_close_message(_partitioned(build))
    ops = _quiet(lambda: _partitioned(build)(True))
    assert is_partitioned(ops.fem)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(BridgeError) as exc:
        _quiet(lambda: PARTITIONED[target](ops, out))
    assert str(exc.value) == expected
    assert phrase in expected
    assert not list(out.iterdir())


# ---------------------------------------------------------------------------
# #2528: a flip counts the carriers alive at it
# ---------------------------------------------------------------------------
#
# Stage 'push' flips the sand of pg A, every element alive there, by
# materialState over A.  pg C carries the same sand and is activated by
# 'install'.  The check used to count C at 'push' too, whichever stage
# activates it, and refused 'push' for leaving C out.  C is not in the
# domain at 'push'; it is 'install''s to reach.  The model is
# ``alive_at_the_flip`` (an R1b model of the replay tests, which replays
# byte for byte); the cases where 'install' does not reach C stay refused
# (the replay tests' REFUSED table, on the stage close, the emit and the
# replay).

#: ``(reflip, register)`` of every passing variant.
ALIVE: tuple[tuple[str, str], ...] = (
    ("sweep", "between"), ("sweep", "after"),
    ("materialState", "between"), ("materialState", "after"),
)


@pytest.mark.parametrize(("reflip", "register"), ALIVE)
@pytest.mark.parametrize("target", ["tcl", "py", "h5"])
def test_a_flip_of_every_alive_carrier_passes(
    reflip: str, register: str, target: str, tmp_path: Path,
) -> None:
    from tests.opensees.h5.test_h5_stage_route_replay import (
        _replay,
        alive_at_the_flip,
    )

    ops = _quiet(lambda: alive_at_the_flip(reflip=reflip, register=register))
    assert [st.name for st in ops._stage_records] == ["push", "install"]
    out = tmp_path / f"model.{target}"
    _quiet(lambda: getattr(ops, target)(str(out)))
    assert out.exists()
    if target == "h5":
        _replay(out, "tcl")


@pytest.mark.parametrize(("reflip", "register"), ALIVE)
@pytest.mark.parametrize("target", sorted(PARTITIONED))
def test_the_partitioned_emit_passes_it_too(
    reflip: str, register: str, target: str, tmp_path: Path,
) -> None:
    from tests.opensees.h5.test_h5_stage_route_replay import alive_at_the_flip

    ops = _quiet(lambda: _partitioned(
        lambda _late: alive_at_the_flip(reflip=reflip, register=register))(
            True))
    out = tmp_path / "out"
    out.mkdir()
    _quiet(lambda: PARTITIONED[target](ops, out))
    assert list(out.iterdir())


def _flip_reaches(
    lines: "list[str]",
) -> "list[tuple[int, list[int], list[int]]]":
    """``(material, reached, alive)`` per plastic ``materialState`` flip of
    a flat Tcl deck: the elements it updates, and the elements carrying
    the material already in the domain when it runs."""
    carriers: dict[int, list[int]] = {}
    flips: list[tuple[int, list[int], list[int]]] = []
    group: dict[int, list[tuple[int, int]]] = {}
    for ln in lines:
        tok = ln.split()
        if tok[:2] == ["element", "tri6n"]:
            carriers.setdefault(int(tok[-1]), []).append(int(tok[2]))
        elif tok[:1] == ["setParameter"] and "materialState" in tok:
            assert tok[1] == "-val" and tok[3] == "-ele", ln
            mat = int(tok[-1])
            if float(tok[2]) >= 1:
                flips.append((mat, sorted(int(e) for e in tok[4:-2]),
                              sorted(carriers.get(mat, []))))
        elif tok[:1] == ["addToParameter"]:
            assert tok[2] == "element" and tok[4] == "materialState", ln
            group.setdefault(int(tok[1]), []).append(
                (int(tok[5]), int(tok[3])))
        elif tok[:1] == ["updateParameter"] and int(tok[1]) in group:
            rows = group.pop(int(tok[1]))
            (mat,) = {m for m, _e in rows}
            if float(tok[2]) >= 1:
                flips.append((mat, sorted(e for _m, e in rows),
                              sorted(carriers.get(mat, []))))
    return flips


@pytest.mark.parametrize(("reflip", "register"), ALIVE)
def test_each_flip_of_the_deck_reaches_every_carrier_alive_at_it(
    reflip: str, register: str,
) -> None:
    """The deck oracle on the stub: each plastic flip of the emitted deck
    updates exactly the elements carrying the material that are in the
    domain when it runs.  The flip at 'push' reaches A, before C exists;
    the flip at 'install' runs after C's element line and reaches A and
    C.  So every element of the sand gets its own Elastic2Plastic() when
    it turns plastic, the per-element answer (#2445)."""
    from tests.opensees.h5.test_h5_stage_route_replay import alive_at_the_flip

    tcl = TclEmitter()
    _quiet(lambda: alive_at_the_flip(
        reflip=reflip, register=register).build().emit(tcl))
    flips = _flip_reaches(tcl.lines())
    assert [(len(reached), reached == alive)
            for _m, reached, alive in flips] == [(1, True), (2, True)]


def test_the_early_twin_is_still_refused_at_the_stage_close() -> None:
    """The stage close knows only the stages closed so far.  When 'push'
    closes with C registered and no stage activating it, C emits
    globally, as far as the program then says, so the flip must reach
    it: the stage close refuses, fail closed.  Registering C after 'push'
    (``register="between"``) is the same model, and it passes."""
    from tests.opensees.h5.test_h5_stage_route_replay import alive_at_the_flip

    with pytest.raises(BridgeError) as exc:
        _quiet(lambda: alive_at_the_flip(register="early"))
    msg = str(exc.value)
    assert msg.startswith("Stage 'push': ") and "leaves 1 of those" in msg


def test_a_flip_of_a_carrier_not_yet_in_the_domain_is_refused_on_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed where counting alive carriers would open a seam: a
    pre-fix archive whose 'push' flips C's element alone, before 'install'
    activates C.  The update reaches nothing at 'push'.  The multi-element
    check no longer counts C there, so the reach check reads the alive
    targets, and the replay refuses by name."""
    from tests.opensees.h5.test_h5_stage_route_replay import (
        _replay,
        _write,
        alive_at_the_flip,
    )

    path = _write(lambda: alive_at_the_flip(push="host"),
                  tmp_path / "pre_fix.h5", monkeypatch, pre_fix=True)
    with pytest.raises(BridgeError) as exc:
        _replay(path)
    msg = str(exc.value)
    assert msg.startswith("Stage 'push': s.update_parameter(")
    assert ("it targets no element in the domain (SixNodeTri on pg 'C' "
            "carries it, but a later stage activates it)") in msg


def test_the_owners_are_the_ownership_pass_owners() -> None:
    """The owner oracle: the ``RouteSpec.owner`` the emit and the stage
    close read is, spec for spec, the stage ``compute_stage_ownership``
    assigns the element to (the pass that places each element in the
    deck).  Before #2528 the emit's owners were all -1."""
    from apeGmsh.opensees._internal.build.ownership import (
        compute_stage_ownership,
    )
    from apeGmsh.opensees._internal.build.stage_validate import (
        validate_stage_routes,
    )
    from apeGmsh.opensees._internal.types import Element

    from tests.opensees.h5.test_h5_stage_route_replay import (
        _flip_then_activate,
        alive_at_the_flip,
    )

    builds: "list[Callable[[], apeSees]]" = [
        alive_at_the_flip,
        lambda: alive_at_the_flip(reflip="materialState", register="after"),
        lambda: _flip_then_activate(reflip=True),
    ]
    for build in builds:
        ops = _quiet(build)
        built = ops.build()
        elements = [p for p in built.primitives if isinstance(p, Element)]
        owner, _nodes = compute_stage_ownership(
            built.stage_records, elements, built.fem)
        seen: list[Any] = []
        with pytest.MonkeyPatch.context() as m:
            m.setattr(_builder, "check_stage_routes",
                      lambda _name, model, **_kw: seen.append(model))
            validate_stage_routes(
                stage_records=built.stage_records,
                primitives=built.primitives, tag_for=built.tag_for,
                fem=built.fem)
        assert seen
        expected = [owner.get(id(e), -1) for e in elements]
        assert [spec.owner for spec in seen[0].specs] == expected
        assert set(expected) == {-1, 1}
        close = _builder._StageBuilder(ops, "probe")._route_model()
        assert [spec.owner for spec in close.specs] == expected


# ---------------------------------------------------------------------------
# Live: the deck gives the per-element reference answer
# ---------------------------------------------------------------------------

#: C's first node in the stub (A 1-6, B 7-12, Q 13-20, P 21-24, C 25-30).
_C0 = 24
_EDGE = ((2, 1 / 6), (3, 1 / 6), (5, 4 / 6))


def _alive_run(tmp_path: Path, *, shared: bool, wrong: bool = False,
               ) -> "list[float]":
    """The #2528 model under load, run as the bridge's openseespy deck on
    the resolved backend; every node's displacement.

    A (global) is consolidated elastic, flipped plastic at 'push' by
    ``materialState`` over pg A while C, which carries the same sand, is
    not alive, and loaded.  'install' activates C, flips the group
    elastic and consolidates C; 'shear' sweeps the group plastic and
    loads both.  ``shared``: one sand on A and C (the model whose 'push'
    was refused); else each its own material and its own
    ``updateMaterialStage``, the per-element reference.  ``wrong``: the
    shared model with 'shear' flipping A alone, as a deck written with
    the route checks off: C turns plastic with no Elastic2Plastic() of
    its own (#2445)."""
    from tests.opensees.h5.test_h5_stage_route_replay import _MD, _ops
    from tests.opensees.integration.test_md_stage_route_elements import (
        _run_bridge_deck,
    )

    ops = _ops()
    names = ("sand",) if shared else ("sand_a", "sand_c")
    mats = [_quiet(lambda n=n: ops.nDMaterial.ManzariDafalias(name=n, **_MD))
            for n in names]
    mat_a, mat_c = mats[0], mats[-1]
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=mat_a)
    ops.fix(nodes=[1, *range(7, _C0 + 1)], dofs=(1, 1))
    ops.fix(nodes=[2, 4], dofs=(0, 1))
    ops.fix(nodes=[3, 6], dofs=(1, 0))

    def chain() -> dict[str, object]:
        return {
            "test": ops.test.NormUnbalance(tol=1e-8, max_iter=100),
            "algorithm": ops.algorithm.Newton(),
            "integrator": ops.integrator.LoadControl(dlam=0.05),
            "constraints": ops.constraints.Transformation(),
            "numberer": ops.numberer.RCM(),
            "system": ops.system.UmfPack(),
            "analysis": ops.analysis.Static(),
        }

    def loads(s: Any, rows: "list[tuple[int, float, float]]") -> None:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for first, fx, fy in rows:
                for n, w in _EDGE:
                    p.load(node=first + n, forces=(fx * w, fy * w))

    def stage(name: str, body: "Callable[[Any], None]") -> None:
        with ops.stage(name=name) as s:
            body(s)
            s.analysis(**chain())
            s.run(n_increments=20)

    def push(s: Any) -> None:
        if shared:
            s.update_parameter("materialState", 1.0, pg="A",
                               material=mat_a)
        else:
            s.update_material_stage(materials=[mat_a], stage=1)
        loads(s, [(0, 0.0, -40.0)])

    def install(s: Any) -> None:
        s.activate(pgs=["C"])
        s.fix(nodes=[_C0 + 1], dofs=(1, 1))
        s.fix(nodes=[_C0 + 2, _C0 + 4], dofs=(0, 1))
        s.fix(nodes=[_C0 + 3, _C0 + 6], dofs=(1, 0))
        s.update_material_stage(materials=mats, stage=0)
        loads(s, [(_C0, -100.0, -100.0)])

    def shear(s: Any) -> None:
        if wrong:
            s.update_parameter("materialState", 1.0, pg="A",
                               material=mat_a)
        else:
            s.update_material_stage(materials=mats, stage=1)
        loads(s, [(0, 0.0, -20.0), (_C0, 0.0, -40.0)])

    stage("gravity", lambda s: (
        s.update_material_stage(materials=[mat_a], stage=0),
        loads(s, [(0, -100.0, -100.0)])))
    stage("push", push)
    ops.element.SixNodeTri(pg="C", thickness=1.0, material=mat_c)
    stage("install", install)
    stage("shear", shear)
    name = "wrong" if wrong else "shared" if shared else "reference"
    live = _quiet(lambda: _run_bridge_deck(ops, tmp_path / f"{name}.py"))
    try:
        return [d for n in sorted(live.getNodeTags())
                for d in live.nodeDisp(n)]
    finally:
        live.wipe()


@pytest.mark.live
def test_the_deck_gives_the_per_element_reference_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the model whose 'push' was refused runs to the answer
    of the deck that gives A and C their own materials and their own
    ``updateMaterialStage``, bit for bit (measured on stock openseespy
    3.7.1 and the fork build 79e062367).  The oracle is not blind: the
    same model with C left uninitialised (checks off) moves C by about
    3 % of the peak displacement."""
    from apeGmsh.opensees import apesees as _apesees

    pytest.importorskip("openseespy.opensees")
    ref = _alive_run(tmp_path, shared=False)
    got = _alive_run(tmp_path, shared=True)
    scale = max(abs(v) for v in ref)
    assert scale > 1e-3
    assert got == ref
    monkeypatch.setattr(_builder, "check_stage_routes", lambda *a, **k: None)
    monkeypatch.setattr(_apesees, "validate_stage_routes", lambda **k: None)
    monkeypatch.setattr(_apesees, "validate_class_wide_stage_flags",
                        lambda **k: None)
    bad = _alive_run(tmp_path, shared=True, wrong=True)
    assert max(abs(a - b) for a, b in zip(bad, ref)) > 1e-2 * scale


# ---------------------------------------------------------------------------
# #2534: late carriers are counted per class-wide group
# ---------------------------------------------------------------------------
#
# The flag the plastic flip at 'plastic' sets is ManzariDafalias's
# class-wide static, so a carrier of a second ManzariDafalias activated
# after it enters plastic with no Elastic2Plastic(), as a carrier of the
# flipped sand does (the CASES row above).  A flip of the whole group at
# the activating stage is the way through; a group whose flip runs nothing
# per element is not counted.


def _group_reflip_at_the_activation() -> apeSees:
    """The refused model with 'install' flipping the whole group again:
    its sweep runs after C's element line and reaches A and C."""
    ops = _ops()
    sand = _md(ops, "sand")
    sand_c = _md(ops, "sand_c")
    _host(ops, "A", sand)
    _host(ops, "C", sand_c)
    _stage(ops, "plastic", lambda s: s.update_material_stage(
        materials=[sand], stage=1))
    _stage(ops, "install", lambda s: (
        s.activate(pgs=["C"]),
        s.update_material_stage(materials=[sand, sand_c], stage=1)))
    return ops


@pytest.mark.parametrize("target", ["tcl", "py", "h5"])
def test_a_flip_of_the_group_at_the_activation_passes(
    target: str, tmp_path: Path,
) -> None:
    from tests.opensees.h5.test_h5_stage_route_replay import _replay

    ops = _quiet(_group_reflip_at_the_activation)
    out = tmp_path / f"model.{target}"
    _quiet(lambda: getattr(ops, target)(str(out)))
    assert out.exists()
    if target == "h5":
        _replay(out, "tcl")


def test_a_group_whose_flip_runs_nothing_per_element_is_not_counted(
) -> None:
    """PM4Sand's ``updateParameter`` for ``updateMaterialStage`` and
    ``materialState`` sets the class-wide ``me2p`` and nothing else
    (``PM4Sand.cpp``; PM4Silt and SAniSandMS alike), so a flip at the
    activation would change nothing in the new element: a late carrier of
    a second PM4Sand gets the per-element answer, and passes."""
    ops = _ops()
    sand = ops.nDMaterial.PM4Sand(name="sand", Dr=0.55, G0=677.0,
                                  hpo=0.4, rho=1.7)
    sand_c = ops.nDMaterial.PM4Sand(name="sand_c", Dr=0.55, G0=677.0,
                                    hpo=0.4, rho=1.7)
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=sand)
    ops.element.SixNodeTri(pg="C", thickness=1.0, material=sand_c)
    _quiet(lambda: _stage(ops, "plastic", lambda s: s.update_material_stage(
        materials=[sand], stage=1)))
    _quiet(lambda: _stage(ops, "install", lambda s: s.activate(pgs=["C"])))
    tcl = TclEmitter()
    _quiet(lambda: ops.build().emit(tcl))
    assert any(ln.startswith("element tri6n") for ln in tcl.lines())


# Live: the per-element reference.  Raw commands on the resolved backend,
# one quad per material, so no first-element-only sweep is in play: A
# (material 1) is consolidated elastic, flipped plastic and loaded; then
# quad C is built and loaded with A.  Variants: "late" (C on material 2,
# no flip after it: the refused model), "same" (C on material 1, no flip:
# #2445's refused model), "reference" (C's own updateMaterialStage after
# it is built, the per-element reference) and "group" (both materials
# flipped after C is built, what the refusal asks for).

_MD_ARGS = (125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7, 100.0, 0.01,
            7.05, 0.968, 1.1, 0.704, 3.5, 4.0, 600.0, 1.42)


def _late_second_material_run(
    variant: str,
) -> "tuple[list[int], list[float]]":
    """Analyze codes and every node's displacement of ``variant``.  The
    run sets stage 0 before it analyzes, so the class-wide static an
    earlier run left in this process does not leak in."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()

    def quad(tag: int, x0: float, mat: int) -> "list[int]":
        nn = [4 * (tag - 1) + k for k in (1, 2, 3, 4)]
        for n, (dx, y) in zip(nn, ((0, 0), (1, 0), (1, 1), (0, 1))):
            ops.node(n, x0 + dx, float(y))
            ops.fix(n, int(dx == 0), int(y == 0))
        ops.element("quad", tag, *nn, 1.0, "PlaneStrain", mat)
        return nn

    def load(series: int, nn: "list[int]", fx: float, fy: float) -> None:
        ops.timeSeries("Linear", series)
        ops.pattern("Plain", series, series)
        for n in (nn[1], nn[2]):
            ops.load(n, fx, 0.0)
        for n in (nn[2], nn[3]):
            ops.load(n, 0.0, fy)

    def analyze(n: int) -> int:
        ops.system("UmfPack")
        ops.numberer("RCM")
        ops.constraints("Transformation")
        ops.test("NormUnbalance", 1e-8, 100)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0 / n)
        ops.analysis("Static")
        code = int(ops.analyze(n))
        ops.loadConst("-time", 0.0)
        ops.wipeAnalysis()
        return code

    try:
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        ops.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
        ops.nDMaterial("ManzariDafalias", 2, *_MD_ARGS)
        a = quad(1, 0.0, 1)
        ops.updateMaterialStage("-material", 1, "-stage", 0)
        load(1, a, -50.0, -50.0)
        codes = [analyze(10)]
        ops.updateMaterialStage("-material", 1, "-stage", 1)
        load(2, a, 0.0, -8.0)
        codes.append(analyze(20))
        c = quad(2, 2.0, 1 if variant == "same" else 2)
        if variant == "reference":
            ops.updateMaterialStage("-material", 2, "-stage", 1)
        elif variant == "group":
            ops.updateMaterialStage("-material", 1, "-stage", 1)
            ops.updateMaterialStage("-material", 2, "-stage", 1)
        load(3, c, -50.0, -50.0)
        codes.append(analyze(20))
        load(4, c, 0.0, -8.0)
        load(5, a, 0.0, -2.0)
        codes.append(analyze(20))
        return codes, [float(d) for n in sorted(ops.getNodeTags())
                       for d in ops.nodeDisp(n)]
    finally:
        ops.wipe()


def _check_the_per_element_reference() -> None:
    runs = {v: _late_second_material_run(v)
            for v in ("reference", "late", "same", "group")}
    assert all(codes == [0, 0, 0, 0] for codes, _d in runs.values()), runs
    ref = runs["reference"][1]
    scale = max(abs(v) for v in ref)
    assert scale > 1e-3
    # What the refusal asks for gives the per-element answer, bit for bit.
    assert runs["group"][1] == ref
    # The refused second-material carrier is the refused same-tag one.
    assert runs["late"][1] == runs["same"][1]
    # And it is not the per-element answer: C (nodes 5-8) moves.
    off = max(abs(x - y) for x, y in zip(runs["late"][1][8:], ref[8:]))
    assert off > 1e-5 * scale, off


@pytest.mark.live
def test_a_late_second_material_is_not_the_per_element_answer() -> None:
    """#2534 on the bound backend.  Stock openseespy 3.7.1: C is off the
    reference by 4.9e-07 on a 3.41e-03 peak (the Pmin stress seed of a
    fresh element), bit for bit the refused same-tag answer.  The fork
    build 79e062367, whose ManzariDafalias constructor resets the static
    (#2443), is off by 1.8e-03.  Flipping the group after the activation
    gives the reference on both."""
    pytest.importorskip("openseespy.opensees")
    _check_the_per_element_reference()


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_a_late_second_material_is_not_the_per_element_answer_on_the_fork(
) -> None:
    """``live-fork``: the same oracle at FORK_PIN, which carries the
    constructor fix (43b235bbb), so C behaves as on stock."""
    _check_the_per_element_reference()
