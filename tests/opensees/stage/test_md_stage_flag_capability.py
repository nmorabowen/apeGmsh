"""#2518: the ManzariDafalias constructor-reset refusal lifts on a fork
whose ``ladrunoSchema`` v1 lists ``mdStageFlagPreserved``.

On a Ladruno fork build without commit ``43b235bbb`` the constructor of
``ManzariDafalias`` / ``LadrunoSANISAND`` sets their class-wide stage flag
back to elastic, so a model that builds an element or material after a
stage flipped the group, and solves before flipping it again, runs the
whole group elastic with no warning (#2443, #2444).  Fork PR #1017 makes
the fix visible: ``ladrunoSchema`` v1 lists ``mdStageFlagPreserved``.  The
refusal (``stage_emit.MD_STATIC_RESET_GUARD``) now lifts exactly there, on
every route, and keeps refusing by name everywhere else:

* the deck route and the ``model.h5`` replay emit a guard that asks the
  running build (``tests/opensees/integration/test_deck_md_static_guard.py``,
  ``tests/opensees/h5/test_h5_stage_route_replay.py``);
* the live route asks the engine it drives, before the solve (here).

The live tests run the two-phase model as the bridge's openseespy deck on
the bound build: where the build passes the guard, the late element
changes nothing (the answer of the model that builds it before the flip);
where it does not, the deck stops by name, and the run the guard stopped
is the wrong answer.
"""
from __future__ import annotations

import types
import warnings
from pathlib import Path
from typing import Any, Callable

import pytest

from apeGmsh.opensees._internal.build import stage_emit
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.stage_emit import (
    MD_STATIC_RESET_GUARD,
    MD_STATIC_RESET_MESSAGE,
    engine_has_capability,
    fork_guard_lines,
)
from apeGmsh.opensees._target import backend_info_of
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

from tests.opensees.integration.test_deck_md_static_guard import (
    SCHEMA_CASES,
    schema_json,
)

_SHA = "6050d78e105bd4fc3a15ddb8ee521aa861a4dafe"
_MD_ARGS = (125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7, 100.0, 0.01,
            7.05, 0.968, 1.1, 0.704, 3.5, 4.0, 600.0, 1.42)


# ---------------------------------------------------------------------------
# The live route, on stub engines
# ---------------------------------------------------------------------------


class _Engine:
    """A stub OpenSees module that records every call.  ``build`` and
    ``schema`` add the fork's ``ladrunoBuild`` / ``ladrunoSchema`` (absent
    when ``None``); ``schema_raises`` makes the second fail.  Every other
    ``ladruno*`` and dunder name is absent, so ``hasattr`` reads stock."""

    def __init__(self, *, build: "str | None" = None,
                 schema: "str | None" = None,
                 schema_raises: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        if build is not None:
            self.ladrunoBuild = lambda: build
        if schema_raises:
            def boom() -> str:
                raise RuntimeError("boom")
            self.ladrunoSchema = boom
        elif schema is not None:
            self.ladrunoSchema = lambda: schema

    def __getattr__(self, name: str) -> Any:
        if name.startswith(("ladruno", "__")):
            raise AttributeError(name)

        def call(*args: Any) -> Any:
            self.calls.append((name, args))
            return [1.0] if name == "eigen" else 0
        return call

    def solved(self) -> bool:
        return any(name in ("analyze", "eigen") for name, _a in self.calls)


def _live(engine: _Engine) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = engine  # type: ignore[assignment]
    le._real_ops = engine  # type: ignore[assignment]
    le._in_partition = False
    le._step_hooks_registered = False
    le._fork_verified_types = set()
    le._partition_warned = True
    return le


def _flip_then_build(le: LiveOpsEmitter, *, solve: str = "analyze",
                     reflip: bool = False, route: str = "stage") -> None:
    """Quad 1 on ManzariDafalias 1, a plastic flip, then quad 2 on an
    elastic material built after it, then the solve: the reset pattern.
    The flip has one carrier, so the live #2445 refusal stays out of it."""
    le.model(ndm=2, ndf=2)
    le.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
    le.nDMaterial("ElasticIsotropic", 2, 1e5, 0.3)
    le.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
    if route == "stage":
        le.update_material_stage(1, 1)
    else:
        le.update_parameter(9, (1,), ("materialState", 1), 1.0)
    le.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 2)
    if reflip:
        le.update_material_stage(1, 1)
    if solve == "analyze":
        le.analyze(steps=1)
    else:
        le.eigen(1)


def _stub_cases() -> "dict[str, tuple[_Engine, bool]]":
    cases = {
        f"fork, {name}": (_Engine(build=_SHA, schema=schema), ok)
        for name, (schema, ok) in SCHEMA_CASES.items()}
    cases.update({
        "stock": (_Engine(), True),
        "fork, no ladrunoSchema": (_Engine(build=_SHA), False),
        "fork, ladrunoSchema raises": (
            _Engine(build=_SHA, schema_raises=True), False),
        # A fork build outside git answers "unknown": BackendInfo reads it
        # as stock with schema None, and it is still refused.
        "fork stamped unknown, v1 lists it": (
            _Engine(build="unknown", schema=schema_json()), False),
        # A schema whose build disagrees with ladrunoBuild is unknown.
        "fork, schema of another build": (
            _Engine(build=_SHA, schema=schema_json(build="0" * 40)), False),
    })
    return cases


@pytest.mark.parametrize("solve", ["analyze", "eigen"])
@pytest.mark.parametrize("route", ["stage", "parameter"])
def test_the_live_route_asks_the_engine(solve: str, route: str) -> None:
    """The oracle on the live route: the key present passes and solves;
    absent (v0, no schema, schema None, every near miss) refuses before
    the solve, with the deck's message."""
    for name, (engine, ok) in _stub_cases().items():
        le = _live(engine)
        if ok:
            _flip_then_build(le, solve=solve, route=route)
            assert engine.solved(), name
            continue
        with pytest.raises(BridgeError) as err:
            _flip_then_build(le, solve=solve, route=route)
        assert str(err.value) == MD_STATIC_RESET_MESSAGE, name
        assert not engine.solved(), name


#: Stub engines the live route refuses and the deck guard passes: the
#: schema lists the key, but the bridge's reader (``_target``) reads it as
#: unknown (``schema is None``) because its ``build`` is not the
#: ``ladrunoBuild`` sha, and the live route never passes a fork on an
#: unknown schema.  The deck cannot read a schema that strictly in Tcl;
#: it takes the build's own promise.  Live is never the looser of the two.
_LIVE_ONLY_REFUSES = frozenset({
    "fork stamped unknown, v1 lists it", "fork, schema of another build"})


def test_the_live_route_agrees_with_the_deck_guard() -> None:
    """One question on both routes: the live verdict on each stub engine is
    the py deck guard's verdict on it, except where only the live route's
    stricter reader refuses."""
    code = compile("\n".join(fork_guard_lines(MD_STATIC_RESET_GUARD, "py")),
                   "<guard>", "exec")
    for name, (engine, ok) in _stub_cases().items():
        assert engine_has_capability(
            MD_STATIC_RESET_GUARD, backend_info_of(engine), engine) is ok, name
        try:
            exec(code, {"ops": engine})
            deck = True
        except SystemExit:
            deck = False
        assert deck is (ok or name in _LIVE_ONLY_REFUSES), name


@pytest.mark.parametrize("schema", [None, schema_json(version=0, caps=None)])
def test_no_refusal_where_every_build_agrees(schema: "str | None") -> None:
    """A fork without the key still solves when nothing could reset: a
    flip of the group after the construction, no construction after the
    flip, or an elastic flip."""
    engine = _Engine(build=_SHA, schema=schema)
    _flip_then_build(_live(engine), reflip=True)
    assert engine.solved()
    for stage in (0, 1):
        engine = _Engine(build=_SHA, schema=schema)
        le = _live(engine)
        le.model(ndm=2, ndf=2)
        le.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
        le.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
        le.update_material_stage(1, stage)
        if stage == 0:
            le.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 1)
        le.analyze(steps=1)
        assert engine.solved()


def test_a_suppressed_rank_block_neither_resets_nor_heals() -> None:
    """Inside a non-zero partition block the live emitter drives no
    engine: a construction there resets nothing, and a flip there heals
    nothing (fail closed)."""
    engine = _Engine(build=_SHA)
    le = _live(engine)
    le.model(ndm=2, ndf=2)
    le.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
    le.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
    le.update_material_stage(1, 1)
    le.partition_open(1)
    le.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 2)
    le.partition_close()
    le.analyze(steps=1)
    assert engine.solved()

    engine = _Engine(build=_SHA)
    le = _live(engine)
    le.model(ndm=2, ndf=2)
    le.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
    le.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)
    le.update_material_stage(1, 1)
    le.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 2)
    le.partition_open(1)
    le.update_material_stage(1, 1)
    le.partition_close()
    with pytest.raises(BridgeError, match="mdStageFlagPreserved"):
        le.analyze(steps=1)


def test_the_watch_is_one_class_on_both_routes() -> None:
    """The deck emitters' carrier table and the live emitter share
    :class:`stage_emit.StageFlagResetWatch`, so the two cannot count a
    reset differently."""
    le = _live(_Engine())
    assert isinstance(le._md_reset_watch(), stage_emit.StageFlagResetWatch)
    carriers = stage_emit.MaterialStageCarriers()
    assert isinstance(carriers._reset, stage_emit.StageFlagResetWatch)


# ---------------------------------------------------------------------------
# Live: the two-phase model as the bridge's deck on the bound build
# ---------------------------------------------------------------------------

#: C's first node in the replay stub (A 1-6, B 7-12, Q 13-20, P 21-24,
#: C 25-30); every element has its own nodes, so A and C never touch.
_C0 = 24
_EDGE = ((2, 1 / 6), (3, 1 / 6), (5, 4 / 6))


def _quiet(fn: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _two_phase(variant: str) -> apeSees:
    """Sand A (ManzariDafalias, SixNodeTri) consolidated elastic at
    'gravity', flipped plastic and pushed at 'push'; at 'install' an
    elastic SixNodeTri C, on its own nodes, is loaded; at 'shear' A is
    pushed again, plastic.

    ``late``: C is activated at 'install', after the flip, and no stage
    flips the group again: the reset pattern, guarded.  ``reference``: C
    is built at the start, so nothing is built after the flip; C never
    touches A, so A's answer is what a build that keeps the flag gives.
    ``elastic``: ``late`` with A flipped back to elastic at 'install', what
    a build that resets the flag does to A."""
    from tests.opensees.h5.test_h5_stage_route_replay import _MD, _ops

    ops = _ops()
    sand = _quiet(lambda: ops.nDMaterial.ManzariDafalias(name="sand", **_MD))
    soft = ops.nDMaterial.ElasticIsotropic(name="soft", E=2e4, nu=0.3)
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=sand)
    ops.fix(nodes=[1, *range(7, _C0 + 1)], dofs=(1, 1))
    ops.fix(nodes=[2, 4], dofs=(0, 1))
    ops.fix(nodes=[3, 6], dofs=(1, 0))

    def fix_c(at: Any) -> None:
        at.fix(nodes=[_C0 + 1], dofs=(1, 1))
        at.fix(nodes=[_C0 + 2, _C0 + 4], dofs=(0, 1))
        at.fix(nodes=[_C0 + 3, _C0 + 6], dofs=(1, 0))

    if variant == "reference":
        ops.element.SixNodeTri(pg="C", thickness=1.0, material=soft)
        fix_c(ops)

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

    def install(s: Any) -> None:
        if variant != "reference":
            s.activate(pgs=["C"])
            fix_c(s)
        if variant == "elastic":
            s.update_material_stage(materials=[sand], stage=0)
        loads(s, [(_C0, -50.0, -50.0)])

    stage("gravity", lambda s: (
        s.update_material_stage(materials=[sand], stage=0),
        loads(s, [(0, -100.0, -100.0)])))
    stage("push", lambda s: (
        s.update_material_stage(materials=[sand], stage=1),
        loads(s, [(0, 0.0, -40.0)])))
    if variant != "reference":
        ops.element.SixNodeTri(pg="C", thickness=1.0, material=soft)
    stage("install", install)
    stage("shear", lambda s: loads(s, [(0, 0.0, -20.0)]))
    return ops


def _a_disp(ops: apeSees, path: Path) -> "list[float]":
    """Run ``ops`` as the bridge's openseespy deck on the resolved backend;
    A's node displacements (nodes 1-6)."""
    from tests.opensees.integration.test_md_stage_route_elements import (
        _run_bridge_deck,
    )

    live = _quiet(lambda: _run_bridge_deck(ops, path))
    try:
        return [d for n in range(1, 7) for d in live.nodeDisp(n)]
    finally:
        live.wipe()


def _check_the_two_phase_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apeGmsh.opensees.emitter.live import get_backend_info, get_ops

    guard = fork_guard_lines(MD_STATIC_RESET_GUARD, "py")
    for variant, guarded in (("late", 1), ("reference", 0), ("elastic", 0)):
        path = tmp_path / f"{variant}_deck.py"
        _quiet(lambda: _two_phase(variant).py(str(path)))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines.count(guard[0]) == guarded, variant

    ref = _a_disp(_two_phase("reference"), tmp_path / "reference.py")
    scale = max(abs(v) for v in ref)
    assert scale > 1e-3
    # The oracle is not blind: A run elastic after 'push' is off the
    # reference by far more than the tolerance below.
    soft = _a_disp(_two_phase("elastic"), tmp_path / "elastic.py")
    assert max(abs(a - b) for a, b in zip(soft, ref)) > 1e-3 * scale

    passes = engine_has_capability(
        MD_STATIC_RESET_GUARD, get_backend_info(), get_ops())
    if passes:
        got = _a_disp(_two_phase("late"), tmp_path / "late.py")
        assert max(abs(a - b) for a, b in zip(got, ref)) <= 1e-8 * scale
    else:
        with pytest.raises(SystemExit) as stop:
            _a_disp(_two_phase("late"), tmp_path / "late.py")
        assert stop.value.code == MD_STATIC_RESET_MESSAGE
    _check_the_raw_deck(passes, monkeypatch)


def _raw_deck(*, add: bool) -> "list[str]":
    """An openseespy deck written by the emitter alone, as the replay or a
    direct driver writes one: quad 1 on ManzariDafalias 1, consolidated
    elastic (hydrostatic 100), flipped plastic; with ``add``, a new
    ManzariDafalias 2 and a quad 2 on it (own nodes, fully fixed), whose
    constructor is the one a stale fork resets the flag in; then +2 on
    quad 1's top.  The bridge's stage close refuses this late carrier
    (#2534); the emitter's guard is the check that stands behind it."""
    from apeGmsh.opensees.emitter.py import PyEmitter

    em = PyEmitter()
    em.model(ndm=2, ndf=2)
    em.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
    for n, (x, y) in zip((1, 2, 3, 4), ((0, 0), (1, 0), (1, 1), (0, 1))):
        em.node(n, float(x), float(y))
        em.fix(n, int(x == 0), int(y == 0))
    em.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStrain", 1)

    def solve(series: int, rows: "list[tuple[int, float, float]]",
              steps: int) -> None:
        em.timeSeries("Linear", series)
        em.pattern_open("Plain", series, series)
        for n, fx, fy in rows:
            em.load(n, fx, fy)
        em.pattern_close()
        em.constraints("Transformation")
        em.numberer("RCM")
        em.system("UmfPack")
        em.test("NormUnbalance", 1e-8, 100)
        em.algorithm("Newton")
        em.integrator("LoadControl", 1.0 / steps)
        em.analysis("Static")
        em.analyze(steps=steps)

    em.stage_open("consolidate")
    em.update_material_stage(1, 0)
    solve(1, [(2, -50.0, 0.0), (3, -50.0, -50.0), (4, 0.0, -50.0)], 10)
    em.stage_close()
    em.stage_open("push")
    em.update_material_stage(1, 1)
    if add:
        em.nDMaterial("ManzariDafalias", 2, *_MD_ARGS)
        for n, (x, y) in zip((5, 6, 7, 8), ((2, 0), (3, 0), (3, 1), (2, 1))):
            em.node(n, float(x), float(y))
            em.fix(n, 1, 1)
        em.element("quad", 2, 5, 6, 7, 8, 1.0, "PlaneStrain", 2)
    solve(2, [(3, 0.0, -2.0), (4, 0.0, -2.0)], 10)
    em.stage_close()
    return em.lines()


def _run_raw(lines: "list[str]") -> "list[float]":
    """Run a deck of :func:`_raw_deck` on the resolved backend; quad 1's
    node displacements."""
    from apeGmsh.opensees.emitter.live import get_ops

    header = "import openseespy.opensees as ops"
    assert lines.count(header) == 1
    src = "\n".join(ln for ln in lines if ln != header)
    live = get_ops()
    try:
        _quiet(lambda: exec(compile(src, "<raw deck>", "exec"),  # noqa: S102
                            {"__name__": "deck", "ops": live}))
        return [d for n in (1, 2, 3, 4) for d in live.nodeDisp(n)]
    finally:
        live.wipe()


def _check_the_raw_deck(passes: bool, monkeypatch: pytest.MonkeyPatch,
                        ) -> None:
    """The constructor-reset deck itself: where the build passes the guard,
    building a ManzariDafalias element after the flip leaves quad 1 on the
    answer of the deck that builds none; where it does not, the deck stops
    by name, and the same deck unguarded is off that answer (the reset)."""
    guard = fork_guard_lines(MD_STATIC_RESET_GUARD, "py")
    ref_lines, add_lines = _raw_deck(add=False), _raw_deck(add=True)
    assert ref_lines.count(guard[0]) == 0
    assert add_lines.count(guard[0]) == 1
    ref = _run_raw(ref_lines)
    scale = max(abs(v) for v in ref)
    assert scale > 1e-4
    if passes:
        got = _run_raw(add_lines)
        assert max(abs(a - b) for a, b in zip(got, ref)) <= 1e-8 * scale
        return
    with pytest.raises(SystemExit) as stop:
        _run_raw(add_lines)
    assert stop.value.code == MD_STATIC_RESET_MESSAGE
    monkeypatch.setattr(stage_emit.MaterialStageCarriers,
                        "solve_fork_guards", lambda self: ())
    bad = _run_raw(_raw_deck(add=True))
    assert max(abs(a - b) for a, b in zip(bad, ref)) > 1e-3 * scale


@pytest.mark.live
def test_the_two_phase_model_on_the_bound_openseespy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``live-stock``: stock keeps the flag and has no ``ladrunoBuild``, so
    the late element's deck runs and gives the reference answer."""
    pytest.importorskip("openseespy.opensees")
    _check_the_two_phase_model(tmp_path, monkeypatch)


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_the_two_phase_model_on_the_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``live-fork``: at FORK_PIN, whose ``ladrunoSchema`` v1 lists
    ``mdStageFlagPreserved``, the guarded deck runs and A matches the
    reference.  On a fork build without the key (the local 79e062367) the
    deck stops by name, and the same deck unguarded is off the
    reference."""
    from apeGmsh.opensees.emitter.live import get_backend_info

    assert get_backend_info().kind == "fork"
    _check_the_two_phase_model(tmp_path, monkeypatch)


def test_the_stub_schema_is_the_forks_shape() -> None:
    """The stub answer parses as the bridge's own reader parses a v1
    schema, with the capability read on ``info.schema``."""
    info = backend_info_of(types.SimpleNamespace(
        ladrunoBuild=lambda: _SHA, ladrunoSchema=lambda: schema_json()))
    assert info.kind == "fork" and info.schema is not None
    assert info.schema.schema_version == 1
    assert "mdStageFlagPreserved" in info.schema.capabilities
