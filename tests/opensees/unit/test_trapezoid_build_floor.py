"""The build floor of a 3-D partial or trapezoidal ``-beamUniform`` (#2565).

The load is ``eleLoad -type -beamUniform wy wz wx aL bL [wyb wzb wxb]``.

- ``forceBeamColumn`` honours it from upstream OpenSees commit
  ``f4c791df6``. Every stock build that reports 3.7.0 or later carries
  that commit, and so does every fork build in the table.
- Every other 3-D beam element the bridge emits honours it only from fork
  commit ``34df942c7``. Fork WP-098 ``9d3c3216d`` fixes the vanilla
  elastic, Timoshenko and displacement beams and ``LadrunoDispBeamColumn``.
  WP-200 fixes ``LadrunoIMKBeam``. ``LadrunoForceBeamColumn`` first exists
  after the floor.
- Fork build 79e062367 keeps only the start values on ``elasticBeamColumn``
  and drops the whole load on three others, while ``analyze`` returns 0
  (#2550).

A load that reaches any element but a ``forceBeamColumn`` needs a fork
build measured to carry the floor. A load on ``forceBeamColumn`` alone
needs a listed fork build, or a stock build that reports 3.7.0 or later.
Both rules hold on the live, Tcl and py routes.

The numeric oracle is
``tests/opensees/live/test_beam_uniform_3d_trapezoid_live.py`` (#2561). It
fails on 79e062367 and passes at ``FORK_PIN``.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.stage_emit import (
    BEAM3D_PARTIAL_LOAD_FLOOR_ELEMENTS,
    BEAM3D_PARTIAL_LOAD_FORK_COMMIT,
    BEAM3D_PARTIAL_LOAD_GUARD,
    BEAM3D_PARTIAL_LOAD_MESSAGE,
    BEAM3D_PARTIAL_LOAD_SAFE_ELEMENTS,
    FORCE_BEAM_PARTIAL_LOAD_COMMIT,
    FORCE_BEAM_PARTIAL_LOAD_GUARD,
    FORCE_BEAM_PARTIAL_LOAD_MESSAGE,
    FORCE_BEAM_PARTIAL_LOAD_STOCK_VERSION,
    FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD,
    FORK_BUILD_HAS_FORCE_BEAM_PARTIAL_LOAD,
    Beam3dPartialLoadTargets,
    ForkBuildAllowlist,
    fork_build_has_beam3d_partial_load,
    fork_build_has_force_beam_partial_load,
    fork_guard_lines,
    is_beam3d_partial_load,
    parse_ops_version,
    stock_version_has_force_beam_partial_load,
)
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from tests.opensees.unit.test_beam_uniform_load import _beam_fem_data

_PIN = "6ee6a04fdc9e8a4699b9c483284151f80b48f701"
#: The maintainer's venv build: an ancestor of fork WP-098, a descendant
#: of upstream f4c791df6.
_STALE_FORK = "79e0623670c61c768357a01dff1477a4a0ef7e06"
_UNLISTED = "c" * 40

#: The tags the cases load: 7 is the element under test, 8 a second one.
_HEAD = ("-ele", 7, "-type", "-beamUniform")
#: The three 3-D forms the bridge emits (pattern._beam_uniform_args).
_W = (-10.0, -4.0, 1.0)
_FULL = _HEAD + _W
_PARTIAL = _HEAD + _W + (0.25, 0.75)
_TRAPEZOID = _HEAD + _W + (0.25, 0.75, -20.0, -8.0, 2.0)
_FLOORED = {"partial": _PARTIAL, "trapezoid": _TRAPEZOID}
#: The element argv after the tag: nodes, then anything (never read).
_ELE_ARGS = (1, 2, 1.0)

#: ``version`` answers as both interpreters return them: ``OPS_VERSION``
#: verbatim (the venv's openseespy 3.7.1.2 over fork build 79e062367
#: answers ``'3.8.0'``), with what a Tcl result may carry around it.
_OLD_VERSIONS = ["3.6.0", "3.5.1", "3.0.0", "2.5.0", " 3.6.0\n"]
_NEW_VERSIONS = ["3.7.0", "3.7.1", "3.8.0", "3.10.0", "4.0.0", " 3.8.0\n",
                 "3.7.0-dev"]
_BAD_VERSIONS = ["", "unknown", "3.7", "v3.8.0", "three", "3.x.0"]


def _pin_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    for line in (root / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def test_the_floor_table_lists_the_pin_and_not_the_stale_build() -> None:
    carries = {sha for sha, ok in FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD.items()
               if ok}
    assert _pin_sha() in carries, "FORK_PIN carries 34df942c7; measure a bump"
    assert BEAM3D_PARTIAL_LOAD_FORK_COMMIT in carries
    assert FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD[_STALE_FORK] is False
    assert set(BEAM3D_PARTIAL_LOAD_GUARD.builds) == carries
    assert BEAM3D_PARTIAL_LOAD_GUARD.stock_runs is False
    assert BEAM3D_PARTIAL_LOAD_GUARD.stock_min_version is None
    assert not fork_build_has_beam3d_partial_load(None)
    assert not fork_build_has_beam3d_partial_load("")
    assert not fork_build_has_beam3d_partial_load(_UNLISTED)
    assert fork_build_has_beam3d_partial_load(_PIN)


def test_the_force_beam_table() -> None:
    """Every row carries upstream f4c791df6 (GitHub compare on the fork,
    79e062367 included: ahead 3472, behind 0).  No listed fork build
    predates it, so the fork side of the check refuses only unlisted
    builds."""
    assert FORCE_BEAM_PARTIAL_LOAD_COMMIT.startswith("f4c791df6")
    carries = {sha for sha, ok in
               FORK_BUILD_HAS_FORCE_BEAM_PARTIAL_LOAD.items() if ok}
    assert _pin_sha() in carries, "FORK_PIN carries f4c791df6; measure a bump"
    assert set(FORK_BUILD_HAS_FORCE_BEAM_PARTIAL_LOAD) == set(
        FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD)
    assert all(FORK_BUILD_HAS_FORCE_BEAM_PARTIAL_LOAD.values())
    assert fork_build_has_force_beam_partial_load(_STALE_FORK)
    assert set(FORCE_BEAM_PARTIAL_LOAD_GUARD.builds) == carries
    assert FORCE_BEAM_PARTIAL_LOAD_GUARD.stock_runs is False
    assert FORCE_BEAM_PARTIAL_LOAD_GUARD.stock_min_version == (3, 7, 0)
    assert FORCE_BEAM_PARTIAL_LOAD_STOCK_VERSION == (3, 7, 0)
    assert not fork_build_has_force_beam_partial_load(None)
    assert not fork_build_has_force_beam_partial_load(_UNLISTED)


@pytest.mark.parametrize("text, parsed", [
    ("3.8.0", (3, 8, 0)), (" 3.7.0\n", (3, 7, 0)), ("3.10.0", (3, 10, 0)),
    ("3.7.0-dev", (3, 7, 0)), ("3.7", None), ("v3.8.0", None), ("", None),
    ("unknown", None), (None, None), (3.8, None),
])
def test_parse_ops_version(text: object, parsed: object) -> None:
    assert parse_ops_version(text) == parsed


@pytest.mark.parametrize("text", _OLD_VERSIONS + _BAD_VERSIONS + [None])
def test_a_stock_version_below_the_threshold_or_unreadable_refuses(
    text: object,
) -> None:
    assert not stock_version_has_force_beam_partial_load(text)


@pytest.mark.parametrize("text", _NEW_VERSIONS)
def test_a_stock_version_at_or_after_the_threshold_passes(text: str) -> None:
    assert stock_version_has_force_beam_partial_load(text)


def test_the_element_lists() -> None:
    """``LOAD_TAG_Beam3dPartialUniformLoad`` branches per 3-D element
    source, counted on the fork's GitHub history:

    =====================  =========  =====  ==========  =====
    element                79e062367  9d3c   34df (floor) pin
    =====================  =========  =====  ==========  =====
    ElasticBeam3d          1 (no tail) 1     1           1
    ElasticTimoshenko3d    0          1      1           1
    DispBeamColumn3d       0          1      1           1
    ForceBeamColumn3d      2          2      2           2
    LadrunoDispBeam3d      0          1      1           1
    LadrunoIMKBeam         0          0      1           1
    LadrunoForceBeam3d     absent     absent absent      2
    =====================  =========  =====  ==========  =====
    """
    assert BEAM3D_PARTIAL_LOAD_SAFE_ELEMENTS == {"forceBeamColumn"}
    assert BEAM3D_PARTIAL_LOAD_FLOOR_ELEMENTS == {
        "elasticBeamColumn", "ElasticTimoshenkoBeam", "dispBeamColumn",
        "LadrunoDispBeamColumn", "LadrunoIMKBeam", "LadrunoForceBeamColumn"}
    assert not BEAM3D_PARTIAL_LOAD_SAFE_ELEMENTS & (
        BEAM3D_PARTIAL_LOAD_FLOOR_ELEMENTS)


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
def test_the_load_form(
    args: tuple[Any, ...], ndm: int, floored: bool,
) -> None:
    assert is_beam3d_partial_load(args, ndm) is floored


def _targets(*elements: tuple[str, int]) -> Beam3dPartialLoadTargets:
    t = Beam3dPartialLoadTargets()
    for ele_type, tag in elements:
        t.note_element(ele_type, tag)
    return t


@pytest.mark.parametrize("ele_type", sorted(BEAM3D_PARTIAL_LOAD_FLOOR_ELEMENTS))
def test_every_listed_element_needs_the_floor(ele_type: str) -> None:
    t = _targets((ele_type, 7))
    assert t.guard_for(_PARTIAL, 3) is BEAM3D_PARTIAL_LOAD_GUARD
    assert t.guard_for(_TRAPEZOID, 3) is BEAM3D_PARTIAL_LOAD_GUARD
    assert t.guard_for(_FULL, 3) is None


def test_a_force_beam_needs_the_force_beam_fix() -> None:
    t = _targets(("forceBeamColumn", 7), ("forceBeamColumn", 8))
    assert t.guard_for(_PARTIAL, 3) is FORCE_BEAM_PARTIAL_LOAD_GUARD
    assert t.guard_for(
        ("-ele", 7, 8) + _TRAPEZOID[2:], 3) is FORCE_BEAM_PARTIAL_LOAD_GUARD
    assert t.guard_for(_FULL, 3) is None


@pytest.mark.parametrize(
    "elements, args",
    [
        # A mix: one needing element among the targets is enough.
        ((("forceBeamColumn", 7), ("elasticBeamColumn", 8)),
         ("-ele", 7, 8) + _PARTIAL[2:]),
        # A tag never created, and an unknown element token (fail closed).
        ((("forceBeamColumn", 8),), _PARTIAL),
        ((("mixedBeamColumn", 7),), _PARTIAL),
        # A tag re-created as another element.
        ((("forceBeamColumn", 7), ("dispBeamColumn", 7)), _PARTIAL),
        # A target list other than -ele t1 t2 ...
        ((("forceBeamColumn", 7),),
         ("-range", 7, 7, "-type", "-beamUniform") + _W + (0.25, 0.75)),
        ((("forceBeamColumn", 7),), ("-ele", "-type", "-beamUniform")
         + _W + (0.25, 0.75)),
    ],
    ids=["mixed", "unknown-tag", "unknown-token", "recreated", "range",
         "no-tags"],
)
def test_the_floor_is_needed(
    elements: tuple[tuple[str, int], ...], args: tuple[Any, ...],
) -> None:
    assert _targets(*elements).guard_for(args, 3) is BEAM3D_PARTIAL_LOAD_GUARD


def test_a_removed_force_beam_needs_the_floor() -> None:
    t = _targets(("forceBeamColumn", 7))
    t.note_removed(7)
    assert t.guard_for(_PARTIAL, 3) is BEAM3D_PARTIAL_LOAD_GUARD


def test_a_versioned_guard_needs_stock_refused() -> None:
    bad = ForkBuildAllowlist(builds=(_PIN,), message="m", stock_runs=True,
                             stock_min_version=(3, 7, 0))
    with pytest.raises(BridgeError, match="stock_runs=False"):
        fork_guard_lines(bad, "tcl")


# ---------------------------------------------------------------------------
# The live route
# ---------------------------------------------------------------------------


class _StockOps:
    """Stock-shaped fake: no ``ladrunoBuild``; records every call."""

    def __init__(self, version: str | None = "3.8.0") -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._version = version

    def version(self) -> str:
        if self._version is None:
            raise RuntimeError("no version command")
        return self._version

    def element(self, *args: Any) -> None:
        self.calls.append(("element", *args))

    def eleLoad(self, *args: Any) -> None:  # noqa: N802
        self.calls.append(("eleLoad", *args))

    def loads(self) -> list[tuple[Any, ...]]:
        return [c[1:] for c in self.calls if c[0] == "eleLoad"]


class _ForkOps(_StockOps):
    def __init__(self, build: str) -> None:
        super().__init__("3.8.0")
        self._build = build

    def ladrunoBuild(self) -> str:  # noqa: N802
        return self._build


def _live(ops: object, ndm: int | None = 3) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[assignment]
    le._in_partition = False
    le._fork_verified_types = set()
    le._model_ndm = ndm
    # The fork-only element probe is recorded, not run.
    le._element_fork_gated = (  # type: ignore[method-assign]
        lambda ele_type, tag, args, key: None)
    return le


_BUILDS = [_StockOps(), _ForkOps(_STALE_FORK), _ForkOps(_UNLISTED),
           _ForkOps("")]
_BUILD_IDS = ["stock", "stale-fork", "unmeasured-fork", "empty-answer"]


@pytest.mark.parametrize("which", sorted(_FLOORED))
@pytest.mark.parametrize(
    "ele_type", sorted(BEAM3D_PARTIAL_LOAD_FLOOR_ELEMENTS))
@pytest.mark.parametrize("ops", _BUILDS, ids=_BUILD_IDS)
def test_live_route_refuses_below_the_floor(
    ops: _StockOps, ele_type: str, which: str,
) -> None:
    ops.calls.clear()
    le = _live(ops)
    le.element(ele_type, 7, *_ELE_ARGS)
    with pytest.raises(RuntimeError) as err:
        le.eleLoad(*_FLOORED[which])
    assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE
    # Refused before the call reaches the interpreter.
    assert ops.loads() == []


def test_live_route_refuses_an_unknown_target() -> None:
    """A tag the emitter never created is read as needing the floor."""
    ops = _ForkOps(_STALE_FORK)
    with pytest.raises(RuntimeError, match="34df942c7"):
        _live(ops).eleLoad(*_TRAPEZOID)
    assert ops.loads() == []


def test_live_route_refuses_when_ndm_is_unknown() -> None:
    """Without a ``model`` call the live route reads the load as 3-D."""
    ops = _ForkOps(_STALE_FORK)
    le = _live(ops, ndm=None)
    le.element("elasticBeamColumn", 7, *_ELE_ARGS)
    with pytest.raises(RuntimeError, match="34df942c7"):
        le.eleLoad(*_TRAPEZOID)
    assert ops.loads() == []


@pytest.mark.parametrize("which", sorted(_FLOORED))
@pytest.mark.parametrize(
    "build", sorted(sha for sha, ok in
                    FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD.items() if ok))
def test_live_route_passes_on_a_listed_build(which: str, build: str) -> None:
    ops = _ForkOps(build)
    le = _live(ops)
    le.element("elasticBeamColumn", 7, *_ELE_ARGS)
    le.eleLoad(*_FLOORED[which])
    assert ops.loads() == [_FLOORED[which]]


def _force_beam_load(ops: object, which: str) -> tuple[LiveOpsEmitter,
                                                        tuple[Any, ...]]:
    le = _live(ops)
    le.element("forceBeamColumn", 7, *_ELE_ARGS)
    le.element("forceBeamColumn", 8, *_ELE_ARGS)
    return le, ("-ele", 7, 8) + _FLOORED[which][2:]


@pytest.mark.parametrize("which", sorted(_FLOORED))
@pytest.mark.parametrize(
    "build", sorted(sha for sha, ok in
                    FORK_BUILD_HAS_FORCE_BEAM_PARTIAL_LOAD.items() if ok))
def test_live_route_passes_a_force_beam_on_a_listed_build(
    which: str, build: str,
) -> None:
    """79e062367 is below the floor yet carries the forceBeamColumn fix."""
    ops = _ForkOps(build)
    le, args = _force_beam_load(ops, which)
    le.eleLoad(*args)
    assert ops.loads() == [args]


@pytest.mark.parametrize("which", sorted(_FLOORED))
def test_live_route_refuses_a_force_beam_on_an_unlisted_build(
    which: str,
) -> None:
    ops = _ForkOps(_UNLISTED)
    le, args = _force_beam_load(ops, which)
    with pytest.raises(RuntimeError) as err:
        le.eleLoad(*args)
    assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE
    assert ops.loads() == []


@pytest.mark.parametrize("version", _OLD_VERSIONS + _BAD_VERSIONS + [None])
def test_live_route_refuses_a_force_beam_on_old_stock(
    version: str | None,
) -> None:
    ops = _StockOps(version)
    le, args = _force_beam_load(ops, "trapezoid")
    with pytest.raises(RuntimeError) as err:
        le.eleLoad(*args)
    assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE
    assert ops.loads() == []


@pytest.mark.parametrize("version", _NEW_VERSIONS)
def test_live_route_keeps_its_stock_refusal_for_a_force_beam(
    version: str,
) -> None:
    """On stock that carries the fix, the live stock refusal of every
    partial form is unchanged: it raises its own message."""
    ops = _StockOps(version)
    le, args = _force_beam_load(ops, "partial")
    with pytest.raises(RuntimeError) as err:
        le.eleLoad(*args)
    assert "requires the Ladruno fork" in str(err.value)
    assert str(err.value) not in (BEAM3D_PARTIAL_LOAD_MESSAGE,
                                  FORCE_BEAM_PARTIAL_LOAD_MESSAGE)
    assert ops.loads() == []


def test_a_full_span_3d_load_runs_on_any_fork_build() -> None:
    ops = _ForkOps(_STALE_FORK)
    le = _live(ops)
    le.element("elasticBeamColumn", 7, *_ELE_ARGS)
    le.eleLoad(*_FULL)
    assert ops.loads() == [_FULL]


class _NoOps:
    """The partition stand-in: records every call, answers no build."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *a: self.calls.append((name, *a))


def _partitioned(real: object) -> tuple[LiveOpsEmitter, _NoOps]:
    stand_in = _NoOps()
    le = _live(stand_in)
    le._real_ops = real  # type: ignore[assignment]
    le._in_partition = True
    return le, stand_in


@pytest.mark.parametrize("real_build, refused", [
    (_STALE_FORK, True), (_UNLISTED, True), (_PIN, False)])
def test_the_partitioned_live_route_reads_the_real_build(
    real_build: str, refused: bool,
) -> None:
    """Inside a partition block the build is the real module's
    (``_real_ops``), not the stand-in's, which answers no sha."""
    real = _ForkOps(real_build)
    le, stand_in = _partitioned(real)
    le.element("dispBeamColumn", 7, *_ELE_ARGS)
    if refused:
        with pytest.raises(RuntimeError) as err:
            le.eleLoad(*_TRAPEZOID)
        assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE
        assert ("eleLoad", *_TRAPEZOID) not in stand_in.calls
    else:
        le.eleLoad(*_TRAPEZOID)
        assert ("eleLoad", *_TRAPEZOID) in stand_in.calls
    assert real.loads() == []


@pytest.mark.parametrize("real, refused", [
    (_ForkOps(_STALE_FORK), False), (_ForkOps(_UNLISTED), True),
    (_StockOps("3.6.0"), True), (_StockOps(None), True),
    (_StockOps("3.8.0"), False)],
    ids=["listed-fork", "unlisted-fork", "old-stock", "no-version",
         "new-stock"])
def test_the_partitioned_live_route_checks_a_force_beam(
    real: _StockOps, refused: bool,
) -> None:
    """The stock gate is skipped inside a partition block, so the version
    check is what stops old stock there."""
    le, stand_in = _partitioned(real)
    le.element("forceBeamColumn", 7, *_ELE_ARGS)
    if refused:
        with pytest.raises(RuntimeError) as err:
            le.eleLoad(*_TRAPEZOID)
        assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE
        assert ("eleLoad", *_TRAPEZOID) not in stand_in.calls
    else:
        le.eleLoad(*_TRAPEZOID)
        assert ("eleLoad", *_TRAPEZOID) in stand_in.calls


def test_the_stock_refusal_still_names_the_fork() -> None:
    """The message keeps the words the stock gate always raised
    (``tests/opensees/unit/test_beam_uniform_load.py``)."""
    assert "requires the Ladruno fork" in BEAM3D_PARTIAL_LOAD_MESSAGE


# ---------------------------------------------------------------------------
# The deck routes
# ---------------------------------------------------------------------------


def _dialect(emitter: Any) -> str:
    return "tcl" if emitter is TclEmitter else "py"


def _guard(guard: ForkBuildAllowlist, emitter: Any) -> list[str]:
    return fork_guard_lines(guard, _dialect(emitter))  # type: ignore[arg-type]


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(
    "ele_type", sorted(BEAM3D_PARTIAL_LOAD_FLOOR_ELEMENTS))
def test_a_floored_deck_checks_the_build_once(
    emitter: Any, ele_type: str,
) -> None:
    guard = _guard(BEAM3D_PARTIAL_LOAD_GUARD, emitter)
    em = emitter()
    em.model(ndm=3, ndf=6)
    em.element(ele_type, 7, *_ELE_ARGS)
    em.eleLoad(*_PARTIAL)
    em.eleLoad(*_TRAPEZOID)
    lines = em.lines()
    assert lines.count(guard[0]) == 1
    at = lines.index(guard[0])
    assert lines[at:at + len(guard)] == guard
    assert at < next(i for i, ln in enumerate(lines) if "eleLoad" in ln)


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
def test_a_force_beam_deck_checks_the_force_beam_fix(emitter: Any) -> None:
    floor = _guard(BEAM3D_PARTIAL_LOAD_GUARD, emitter)
    force = _guard(FORCE_BEAM_PARTIAL_LOAD_GUARD, emitter)
    em = emitter()
    em.model(ndm=3, ndf=6)
    em.element("forceBeamColumn", 7, *_ELE_ARGS)
    em.eleLoad(*_PARTIAL)
    em.eleLoad(*_TRAPEZOID)
    lines = em.lines()
    assert lines.count(force[0]) == 1
    at = lines.index(force[0])
    assert lines[at:at + len(force)] == force
    assert at < next(i for i, ln in enumerate(lines) if "eleLoad" in ln)
    assert floor[0] not in lines


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
def test_a_mixed_deck_checks_the_floor(emitter: Any) -> None:
    floor = _guard(BEAM3D_PARTIAL_LOAD_GUARD, emitter)
    force = _guard(FORCE_BEAM_PARTIAL_LOAD_GUARD, emitter)
    em = emitter()
    em.model(ndm=3, ndf=6)
    em.element("forceBeamColumn", 7, *_ELE_ARGS)
    em.element("elasticBeamColumn", 8, *_ELE_ARGS)
    em.eleLoad("-ele", 7, *_PARTIAL[2:])
    assert floor[0] not in em.lines() and force[0] in em.lines()
    em.eleLoad("-ele", 7, 8, *_PARTIAL[2:])
    assert em.lines().count(floor[0]) == 1


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(
    "ndm, ele_type, args",
    [(3, "elasticBeamColumn", _FULL),
     (3, "forceBeamColumn", _FULL),
     (2, "elasticBeamColumn",
      _HEAD + (-10.0, 0.0, 0.25, 0.75, -20.0, 0.0))],
    ids=["3d-full-span", "force-full-span", "2d-trapezoid"],
)
def test_an_unfloored_deck_has_no_guard(
    emitter: Any, ndm: int, ele_type: str, args: tuple[Any, ...],
) -> None:
    em = emitter()
    em.model(ndm=ndm, ndf=6 if ndm == 3 else 3)
    em.element(ele_type, 7, *_ELE_ARGS)
    em.eleLoad(*args)
    assert not any("ladrunoBuild" in ln for ln in em.lines())


def _ranked_deck(em: TclEmitter, ele_type: str) -> None:
    """One beam on each of ranks 0 and 1, then each rank's block loads its
    own beam, as a partitioned build emits it."""
    em.model(ndm=3, ndf=6)
    for rank, tag in ((0, 7), (1, 8)):
        em.partition_open(rank)
        em.element(ele_type, tag, *_ELE_ARGS)
        em.partition_close()
    for rank, tag in ((0, 7), (1, 8)):
        em.partition_open(rank)
        em.eleLoad("-ele", tag, *_TRAPEZOID[2:])
        em.partition_close()


_TCL_FLOOR = fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, "tcl")
_TCL_FORCE = fork_guard_lines(FORCE_BEAM_PARTIAL_LOAD_GUARD, "tcl")


@pytest.mark.parametrize("per_rank", [False, True])
@pytest.mark.parametrize("ele_type, guard", [
    ("elasticBeamColumn", _TCL_FLOOR), ("forceBeamColumn", _TCL_FORCE)])
def test_a_streamed_partitioned_deck_checks_in_each_rank_block(
    tmp_path: Path, per_rank: bool, ele_type: str, guard: list[str],
) -> None:
    """Streamed, the deck's start is on disk already, so each rank's block
    checks right before its own eleLoad: a check in rank 0's block never
    runs on rank 1."""
    path = tmp_path / "s.tcl"
    em = TclEmitter()
    em.stream_to(str(path), per_rank=per_rank)
    _ranked_deck(em, ele_type)
    em.stream_finish()
    text = path.read_text(encoding="utf-8").splitlines()
    n = len(guard)
    for rank, tag in ((0, 7), (1, 8)):
        if per_rank:
            lines = (tmp_path / "ranks" / f"rank{rank}_1.tcl").read_text(
                encoding="utf-8").splitlines()
        else:
            start = [i for i, ln in enumerate(text)
                     if ln == f"if {{[getPID] == {rank}}} {{"][1]
            lines = text[start:start + n + 5]
        at = next(i for i, ln in enumerate(lines)
                  if ln.strip().startswith(f"eleLoad -ele {tag} "))
        assert [ln.strip() for ln in lines[at - n:at]] == [
            g.strip() for g in guard], (rank, lines)
        assert sum(ln.strip() == guard[0] for ln in lines) == 1


@pytest.mark.parametrize("ele_type, guard, other", [
    ("LadrunoDispBeamColumn", _TCL_FLOOR, _TCL_FORCE),
    ("forceBeamColumn", _TCL_FORCE, _TCL_FLOOR)])
def test_a_per_rank_deck_checks_in_the_driver(
    tmp_path: Path, ele_type: str, guard: list[str], other: list[str],
) -> None:
    """Buffered and written per rank, the check sits in the driver before
    any rank block, so every rank runs it before its own eleLoad."""
    from apeGmsh.opensees.apesees import _write_per_rank_tcl

    em = TclEmitter()
    _ranked_deck(em, ele_type)
    path = tmp_path / "d.tcl"
    _write_per_rank_tcl(str(path), em.line_buffer(), em.partition_spans())
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = [i for i, ln in enumerate(lines)
              if "[getPID] ==" in ln and "source" in ln]
    at = lines.index(guard[0])
    assert lines[at:at + len(guard)] == guard
    assert lines.count(guard[0]) == 1
    assert other[0] not in lines
    assert len(blocks) == 4 and at < min(blocks)
    for rank, tag in ((0, 7), (1, 8)):
        frag = (tmp_path / "ranks" / f"rank{rank}_1.tcl").read_text(
            encoding="utf-8")
        assert f"eleLoad -ele {tag} " in frag
        assert guard[0] not in frag


def _tcl() -> Any:
    tkinter = pytest.importorskip("tkinter")
    return tkinter, tkinter.Tcl()


def test_the_tcl_floor_guard_on_stub_builds() -> None:
    tkinter, tcl = _tcl()
    guard = "\n".join(_TCL_FLOOR)
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


def test_the_tcl_force_beam_guard_on_stub_stock() -> None:
    tkinter, tcl = _tcl()
    guard = "\n".join(_TCL_FORCE)
    with pytest.raises(tkinter.TclError) as err:
        tcl.eval(guard)  # no version command: unreadable, stops
    assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE
    for answer in _NEW_VERSIONS:
        tcl.eval(f"proc version {{}} {{ return {{{answer}}} }}")
        tcl.eval(guard)
    for answer in _OLD_VERSIONS + _BAD_VERSIONS:
        tcl.eval(f"proc version {{}} {{ return {{{answer}}} }}")
        with pytest.raises(tkinter.TclError) as err:
            tcl.eval(guard)
        assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE, answer


def test_the_tcl_force_beam_guard_on_stub_forks() -> None:
    """A fork build is judged by its sha, whatever version it reports."""
    tkinter, tcl = _tcl()
    guard = "\n".join(_TCL_FORCE)
    tcl.eval("proc version {} { return 3.8.0 }")
    for answer, ok in ((_PIN, True), (_STALE_FORK, True), (f" {_PIN}\n", True),
                       (_UNLISTED, False), ("", False)):
        tcl.eval(f"proc ladrunoBuild {{}} {{ return {{{answer}}} }}")
        if ok:
            tcl.eval(guard)
            continue
        with pytest.raises(tkinter.TclError) as err:
            tcl.eval(guard)
        assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE, answer


def _exec(guard: ForkBuildAllowlist, ops: object) -> None:
    exec("\n".join(fork_guard_lines(guard, "py")), {"ops": ops})


def test_the_py_floor_guard_on_stub_builds() -> None:
    g = BEAM3D_PARTIAL_LOAD_GUARD
    with pytest.raises(SystemExit) as err:
        _exec(g, types.SimpleNamespace(version=lambda: "3.8.0"))  # stock
    assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE
    for answer, ok in ((_PIN, True), (_STALE_FORK, False), ("", False)):
        ops = types.SimpleNamespace(ladrunoBuild=lambda answer=answer: answer)
        if ok:
            _exec(g, ops)
            continue
        with pytest.raises(SystemExit) as err:
            _exec(g, ops)
        assert str(err.value) == BEAM3D_PARTIAL_LOAD_MESSAGE, answer


def test_the_py_force_beam_guard_on_stub_builds() -> None:
    g = FORCE_BEAM_PARTIAL_LOAD_GUARD
    with pytest.raises(SystemExit) as err:
        _exec(g, types.SimpleNamespace())  # no version(): unreadable
    assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE
    for answer in _NEW_VERSIONS:
        _exec(g, types.SimpleNamespace(version=lambda answer=answer: answer))
    for answer in _OLD_VERSIONS + _BAD_VERSIONS:
        with pytest.raises(SystemExit) as err:
            _exec(g, types.SimpleNamespace(
                version=lambda answer=answer: answer))
        assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE, answer
    for build, ok in ((_PIN, True), (_STALE_FORK, True), (_UNLISTED, False)):
        ops = types.SimpleNamespace(
            version=lambda: "3.8.0", ladrunoBuild=lambda build=build: build)
        if ok:
            _exec(g, ops)
            continue
        with pytest.raises(SystemExit) as err:
            _exec(g, ops)
        assert str(err.value) == FORCE_BEAM_PARTIAL_LOAD_MESSAGE, build


# ---------------------------------------------------------------------------
# Through the bridge
# ---------------------------------------------------------------------------


def _bridge(element: str, **load: Any) -> apeSees:
    ops = apeSees(_beam_fem_data(ndm=3))
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    if element == "elasticBeamColumn":
        ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.01, E=200e9, Iz=1e-4, Iy=1e-4,
            G=80e9, J=1e-4,
        )
    else:
        sec = ops.section.Elastic(
            E=200e9, A=0.01, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
        ops.element.forceBeamColumn(pg="Beams", transf=t, integration=integ)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.beam_uniform(pg="Beams", **load)
    return ops


_LOADS = {
    "full-span": {"wy": -10.0, "wz": -4.0},
    "partial": {"wy": -10.0, "wz": -4.0, "span": (0.25, 0.75)},
    "trapezoid": {"wy": (-10.0, -20.0), "wz": (-4.0, -8.0),
                  "wx": (1.0, 2.0), "span": (0.25, 0.75)},
}


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(
    "element, load, which",
    [
        ("elasticBeamColumn", "full-span", None),
        ("elasticBeamColumn", "partial", "floor"),
        ("elasticBeamColumn", "trapezoid", "floor"),
        ("forceBeamColumn", "full-span", None),
        ("forceBeamColumn", "partial", "force"),
        ("forceBeamColumn", "trapezoid", "force"),
    ],
)
def test_the_bridge_deck_carries_the_guard(
    emitter: Any, element: str, load: str, which: str | None,
) -> None:
    floor = _guard(BEAM3D_PARTIAL_LOAD_GUARD, emitter)
    force = _guard(FORCE_BEAM_PARTIAL_LOAD_GUARD, emitter)
    em = emitter()
    _bridge(element, **_LOADS[load]).build().emit(em)
    lines = em.lines()
    assert any("-beamUniform" in ln for ln in lines)
    assert (floor[0] in lines) is (which == "floor")
    assert (force[0] in lines) is (which == "force")
