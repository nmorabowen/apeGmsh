"""The build floor of a 3-D partial or trapezoidal ``-beamUniform`` (#2565).

Every 3-D beam element the bridge emits honours ``eleLoad -type
-beamUniform wy wz wx aL bL [wyb wzb wxb]`` only from fork commit
``34df942c7`` (fork WP-098 ``9d3c3216d`` for the vanilla elastic,
Timoshenko and displacement beams and ``LadrunoDispBeamColumn``, WP-200 for
``LadrunoIMKBeam``).  Fork build 79e062367 keeps only the start values on
``elasticBeamColumn`` and drops the whole load on the other three while
``analyze`` returns 0 (#2550).  The live, Tcl and py routes refuse every
build not measured to carry the floor.  The numeric oracle, which fails
on 79e062367 and passes at ``FORK_PIN``, is
``tests/opensees/live/test_beam_uniform_3d_trapezoid_live.py`` (#2561).
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.stage_emit import (
    BEAM3D_PARTIAL_LOAD_FORK_COMMIT,
    BEAM3D_PARTIAL_LOAD_GUARD,
    BEAM3D_PARTIAL_LOAD_MESSAGE,
    FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD,
    fork_build_has_beam3d_partial_load,
    fork_guard_lines,
    is_beam3d_partial_load,
)
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from tests.opensees.unit.test_beam_uniform_load import _beam_fem_data

_PIN = "bdb69cea5a9fb17f474cb526e745ac5efb89b0be"
#: The maintainer's venv build, an ancestor of fork WP-098.
_STALE_FORK = "79e0623670c61c768357a01dff1477a4a0ef7e06"

_HEAD = ("-ele", 7, 8, "-type", "-beamUniform")
#: The three 3-D forms the bridge emits (pattern._beam_uniform_args).
_FULL = _HEAD + (-10.0, -4.0, 1.0)
_PARTIAL = _HEAD + (-10.0, -4.0, 1.0, 0.25, 0.75)
_TRAPEZOID = _HEAD + (-10.0, -4.0, 1.0, 0.25, 0.75, -20.0, -8.0, 2.0)
_FLOORED = {"partial": _PARTIAL, "trapezoid": _TRAPEZOID}


def _pin_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    for line in (root / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_table_lists_the_pin_and_not_the_stale_build() -> None:
    carries = {sha for sha, ok in FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD.items()
               if ok}
    assert _pin_sha() in carries, "FORK_PIN carries 34df942c7; measure a bump"
    assert BEAM3D_PARTIAL_LOAD_FORK_COMMIT in carries
    assert _STALE_FORK not in carries
    assert FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD[_STALE_FORK] is False
    assert set(BEAM3D_PARTIAL_LOAD_GUARD.builds) == carries
    assert BEAM3D_PARTIAL_LOAD_GUARD.stock_runs is False
    assert not fork_build_has_beam3d_partial_load(None)
    assert not fork_build_has_beam3d_partial_load("")
    assert not fork_build_has_beam3d_partial_load("c" * 40)
    assert fork_build_has_beam3d_partial_load(_PIN)


@pytest.mark.parametrize(
    "args, ndm, floored",
    [
        (_FULL, 3, False),
        (_PARTIAL, 3, True),
        (_TRAPEZOID, 3, True),
        # 2-D forms are outside this floor.
        (_HEAD + (-10.0, 0.0, 0.25, 0.75), 2, False),
        (_HEAD + (-10.0, 0.0, 0.25, 0.75, -20.0, 0.0), 2, False),
        # Other load types, and a -beamUniform with no -type, never are.
        (("-ele", 7, "-type", "-selfWeight", 0.0, 0.0, -1.0, 9.0), 3, False),
        (("-ele", 7, "-type", "-surfaceLoad"), 3, False),
        (("-ele", 7, "-beamUniform", 1.0, 2.0, 3.0, 4.0), 3, False),
        (("-ele", 7, "-type"), 3, False),
    ],
)
def test_the_predicate(
    args: tuple[Any, ...], ndm: int, floored: bool,
) -> None:
    assert is_beam3d_partial_load(args, ndm) is floored


# ---------------------------------------------------------------------------
# The live route
# ---------------------------------------------------------------------------


class _StockOps:
    """Stock-shaped fake: no ``ladrunoBuild``; records eleLoad calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def eleLoad(self, *args: Any) -> None:  # noqa: N802
        self.calls.append(args)


class _ForkOps(_StockOps):
    def __init__(self, build: str) -> None:
        super().__init__()
        self._build = build

    def ladrunoBuild(self) -> str:  # noqa: N802
        return self._build


def _live(ops: object, ndm: int | None = 3) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[assignment]
    le._in_partition = False
    le._model_ndm = ndm
    return le


@pytest.mark.parametrize("which", sorted(_FLOORED))
@pytest.mark.parametrize(
    "ops", [_StockOps(), _ForkOps(_STALE_FORK), _ForkOps("c" * 40),
            _ForkOps("")],
    ids=["stock", "stale-fork", "unmeasured-fork", "empty-answer"],
)
def test_live_route_refuses_below_the_floor(
    ops: _StockOps, which: str,
) -> None:
    ops.calls.clear()
    with pytest.raises(RuntimeError) as err:
        _live(ops).eleLoad(*_FLOORED[which])
    assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE
    # Refused before the call reaches the interpreter.
    assert ops.calls == []


def test_live_route_refuses_when_ndm_is_unknown() -> None:
    """Without a ``model`` call the live route reads the load as 3-D."""
    ops = _ForkOps(_STALE_FORK)
    with pytest.raises(RuntimeError, match="34df942c7"):
        _live(ops, ndm=None).eleLoad(*_TRAPEZOID)
    assert ops.calls == []


@pytest.mark.parametrize("which", sorted(_FLOORED))
@pytest.mark.parametrize(
    "build", sorted(sha for sha, ok in
                    FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD.items() if ok))
def test_live_route_passes_on_a_listed_build(which: str, build: str) -> None:
    ops = _ForkOps(build)
    _live(ops).eleLoad(*_FLOORED[which])
    assert ops.calls == [_FLOORED[which]]


def test_a_full_span_3d_load_runs_on_any_fork_build() -> None:
    ops = _ForkOps(_STALE_FORK)
    _live(ops).eleLoad(*_FULL)
    assert ops.calls == [_FULL]


def test_the_stock_refusal_still_names_the_fork() -> None:
    """The message keeps the words the stock gate always raised
    (``tests/opensees/unit/test_beam_uniform_load.py``)."""
    assert "requires the Ladruno fork" in BEAM3D_PARTIAL_LOAD_MESSAGE


# ---------------------------------------------------------------------------
# The deck routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
def test_a_floored_deck_checks_the_build_once(emitter: Any) -> None:
    dialect = "tcl" if emitter is TclEmitter else "py"
    guard = fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, dialect)
    em = emitter()
    em.model(ndm=3, ndf=6)
    em.eleLoad(*_PARTIAL)
    em.eleLoad(*_TRAPEZOID)
    lines = em.lines()
    assert lines.count(guard[0]) == 1
    at = lines.index(guard[0])
    assert lines[at:at + len(guard)] == guard
    assert at < next(i for i, ln in enumerate(lines) if "eleLoad" in ln)


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(
    "ndm, args",
    [(3, _FULL), (2, _HEAD + (-10.0, 0.0, 0.25, 0.75, -20.0, 0.0))],
    ids=["3d-full-span", "2d-trapezoid"],
)
def test_an_unfloored_deck_has_no_guard(
    emitter: Any, ndm: int, args: tuple[Any, ...],
) -> None:
    em = emitter()
    em.model(ndm=ndm, ndf=6 if ndm == 3 else 3)
    em.eleLoad(*args)
    assert not any("ladrunoBuild" in ln for ln in em.lines())


def test_the_tcl_guard_on_stub_builds() -> None:
    tkinter = pytest.importorskip("tkinter")
    tcl = tkinter.Tcl()
    guard = "\n".join(fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, "tcl"))
    with pytest.raises(tkinter.TclError) as err:
        tcl.eval(guard)  # stock: no ladrunoBuild, stops
    assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE
    for answer, ok in ((_PIN, True), (f" {_PIN}\n", True),
                       (_STALE_FORK, False), ("", False), ("0" * 40, False)):
        tcl.eval(f"proc ladrunoBuild {{}} {{ return {{{answer}}} }}")
        if ok:
            tcl.eval(guard)
            continue
        with pytest.raises(tkinter.TclError) as err:
            tcl.eval(guard)
        assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE, answer


def test_the_py_guard_on_stub_builds() -> None:
    code = "\n".join(fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, "py"))
    with pytest.raises(SystemExit) as err:
        exec(code, {"ops": types.SimpleNamespace()})  # stock stops
    assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE
    for answer, ok in ((_PIN, True), (_STALE_FORK, False), ("", False)):
        engine = {"ops": types.SimpleNamespace(
            ladrunoBuild=lambda answer=answer: answer)}
        if ok:
            exec(code, engine)
            continue
        with pytest.raises(SystemExit) as err:
            exec(code, engine)
        assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE, answer


# ---------------------------------------------------------------------------
# Through the bridge
# ---------------------------------------------------------------------------


def _bridge(**load: Any) -> apeSees:
    ops = apeSees(_beam_fem_data(ndm=3))
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Beams", transf=t, A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9,
        J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.beam_uniform(pg="Beams", **load)
    return ops


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(
    "load, guarded",
    [
        ({"wy": -10.0, "wz": -4.0}, False),
        ({"wy": -10.0, "wz": -4.0, "span": (0.25, 0.75)}, True),
        ({"wy": (-10.0, -20.0), "wz": (-4.0, -8.0)}, True),
    ],
    ids=["full-span", "partial", "trapezoid"],
)
def test_the_bridge_deck_carries_the_guard(
    emitter: Any, load: dict[str, Any], guarded: bool,
) -> None:
    dialect = "tcl" if emitter is TclEmitter else "py"
    guard = fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, dialect)
    em = emitter()
    _bridge(**load).build().emit(em)
    lines = em.lines()
    assert (guard[0] in lines) is guarded
