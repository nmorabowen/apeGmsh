"""The build floor of a 3-D partial or trapezoidal ``-beamUniform`` (#2565).

The load is ``eleLoad -type -beamUniform wy wz wx aL bL [wyb wzb wxb]``.
``forceBeamColumn`` honours it on stock OpenSees and on every fork build.
Every other 3-D beam element the bridge emits honours it only from fork
commit ``34df942c7``:

- fork WP-098 ``9d3c3216d`` fixes the vanilla elastic, Timoshenko and
  displacement beams and ``LadrunoDispBeamColumn``;
- WP-200 fixes ``LadrunoIMKBeam``;
- ``LadrunoForceBeamColumn`` first exists after the floor.

Fork build 79e062367 keeps only the start values on ``elasticBeamColumn``
and drops the whole load on three of them while ``analyze`` returns 0
(#2550).

So a load that reaches any element but a ``forceBeamColumn`` needs a
build measured to carry the floor, on the live, Tcl and py routes alike.
The numeric oracle, which fails on 79e062367 and passes at ``FORK_PIN``,
is ``tests/opensees/live/test_beam_uniform_3d_trapezoid_live.py`` (#2561).
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.stage_emit import (
    BEAM3D_PARTIAL_LOAD_FLOOR_ELEMENTS,
    BEAM3D_PARTIAL_LOAD_FORK_COMMIT,
    BEAM3D_PARTIAL_LOAD_GUARD,
    BEAM3D_PARTIAL_LOAD_MESSAGE,
    BEAM3D_PARTIAL_LOAD_SAFE_ELEMENTS,
    FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD,
    Beam3dPartialLoadTargets,
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


def _pin_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    for line in (root / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


# ---------------------------------------------------------------------------
# The tables
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
    assert not fork_build_has_beam3d_partial_load(_UNLISTED)
    assert fork_build_has_beam3d_partial_load(_PIN)


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
    assert t.needs_floor(_PARTIAL, 3)
    assert t.needs_floor(_TRAPEZOID, 3)
    assert not t.needs_floor(_FULL, 3)


def test_a_force_beam_needs_no_floor() -> None:
    t = _targets(("forceBeamColumn", 7), ("forceBeamColumn", 8))
    assert not t.needs_floor(_PARTIAL, 3)
    assert not t.needs_floor(("-ele", 7, 8) + _TRAPEZOID[2:], 3)


@pytest.mark.parametrize(
    "elements, args",
    [
        # A mix: one needing element among the targets is enough.
        ((("forceBeamColumn", 7), ("elasticBeamColumn", 8)),
         ("-ele", 7, 8) + _PARTIAL[2:]),
        # A tag never created, and an unknown element token (fail closed).
        ((("forceBeamColumn", 8),), _PARTIAL),
        ((("mixedBeamColumn", 7),), _PARTIAL),
        # A tag re-created as another element, or removed.
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
    assert _targets(*elements).needs_floor(args, 3)


def test_a_removed_force_beam_needs_the_floor() -> None:
    t = _targets(("forceBeamColumn", 7))
    t.note_removed(7)
    assert t.needs_floor(_PARTIAL, 3)


# ---------------------------------------------------------------------------
# The live route
# ---------------------------------------------------------------------------


class _StockOps:
    """Stock-shaped fake: no ``ladrunoBuild``; records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def element(self, *args: Any) -> None:
        self.calls.append(("element", *args))

    def eleLoad(self, *args: Any) -> None:  # noqa: N802
        self.calls.append(("eleLoad", *args))

    def loads(self) -> list[tuple[Any, ...]]:
        return [c[1:] for c in self.calls if c[0] == "eleLoad"]


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


@pytest.mark.parametrize("which", sorted(_FLOORED))
@pytest.mark.parametrize("build", [_STALE_FORK, _UNLISTED])
def test_live_route_passes_a_force_beam_on_an_unlisted_build(
    which: str, build: str,
) -> None:
    ops = _ForkOps(build)
    le = _live(ops)
    le.element("forceBeamColumn", 7, *_ELE_ARGS)
    le.element("forceBeamColumn", 8, *_ELE_ARGS)
    args = ("-ele", 7, 8) + _FLOORED[which][2:]
    le.eleLoad(*args)
    assert ops.loads() == [args]


def test_live_route_keeps_its_stock_refusal_for_a_force_beam() -> None:
    """The live stock refusal of any partial form is unchanged: stock
    openseespy is refused with its own message, not the floor's."""
    ops = _StockOps()
    le = _live(ops)
    le.element("forceBeamColumn", 7, *_ELE_ARGS)
    with pytest.raises(RuntimeError) as err:
        le.eleLoad(*_PARTIAL)
    assert "requires the Ladruno fork" in str(err.value)
    assert str(err.value) != BEAM3D_PARTIAL_LOAD_MESSAGE
    assert ops.loads() == []


def test_a_full_span_3d_load_runs_on_any_fork_build() -> None:
    ops = _ForkOps(_STALE_FORK)
    le = _live(ops)
    le.element("elasticBeamColumn", 7, *_ELE_ARGS)
    le.eleLoad(*_FULL)
    assert ops.loads() == [_FULL]


class _NoOps:
    """The partition stand-in: records eleLoad, has no ladrunoBuild."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def __getattr__(self, name: str) -> Any:
        return lambda *a: self.calls.append((name, *a))


@pytest.mark.parametrize("real_build, refused", [
    (_STALE_FORK, True), (_UNLISTED, True), (_PIN, False)])
def test_the_partitioned_live_route_reads_the_real_build(
    real_build: str, refused: bool,
) -> None:
    """Inside a partition block the build is the real module's
    (``_real_ops``), not the stand-in's, which answers no sha."""
    real = _ForkOps(real_build)
    stand_in = _NoOps()
    le = _live(stand_in)
    le._real_ops = real  # type: ignore[assignment]
    le._in_partition = True
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


def test_the_partitioned_live_route_passes_a_force_beam() -> None:
    real = _ForkOps(_STALE_FORK)
    stand_in = _NoOps()
    le = _live(stand_in)
    le._real_ops = real  # type: ignore[assignment]
    le._in_partition = True
    le.element("forceBeamColumn", 7, *_ELE_ARGS)
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


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(
    "ele_type", sorted(BEAM3D_PARTIAL_LOAD_FLOOR_ELEMENTS))
def test_a_floored_deck_checks_the_build_once(
    emitter: Any, ele_type: str,
) -> None:
    guard = fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, _dialect(emitter))
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
def test_a_mixed_deck_checks_the_build(emitter: Any) -> None:
    guard = fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, _dialect(emitter))
    em = emitter()
    em.model(ndm=3, ndf=6)
    em.element("forceBeamColumn", 7, *_ELE_ARGS)
    em.element("elasticBeamColumn", 8, *_ELE_ARGS)
    em.eleLoad("-ele", 7, *_PARTIAL[2:])
    assert guard[0] not in em.lines()
    em.eleLoad("-ele", 7, 8, *_PARTIAL[2:])
    assert em.lines().count(guard[0]) == 1


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(
    "ndm, ele_type, args",
    [(3, "elasticBeamColumn", _FULL),
     (3, "forceBeamColumn", _PARTIAL),
     (3, "forceBeamColumn", _TRAPEZOID),
     (2, "elasticBeamColumn",
      _HEAD + (-10.0, 0.0, 0.25, 0.75, -20.0, 0.0))],
    ids=["3d-full-span", "force-partial", "force-trapezoid", "2d-trapezoid"],
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


_TCL_GUARD = fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, "tcl")


@pytest.mark.parametrize("per_rank", [False, True])
def test_a_streamed_partitioned_deck_checks_in_each_rank_block(
    tmp_path: Path, per_rank: bool,
) -> None:
    """Streamed, the deck's start is on disk already, so each rank's block
    checks right before its own eleLoad: a check in rank 0's block never
    runs on rank 1."""
    path = tmp_path / "s.tcl"
    em = TclEmitter()
    em.stream_to(str(path), per_rank=per_rank)
    _ranked_deck(em, "elasticBeamColumn")
    em.stream_finish()
    text = path.read_text(encoding="utf-8").splitlines()
    for rank, tag in ((0, 7), (1, 8)):
        if per_rank:
            lines = (tmp_path / "ranks" / f"rank{rank}_1.tcl").read_text(
                encoding="utf-8").splitlines()
        else:
            start = [i for i, ln in enumerate(text)
                     if ln == f"if {{[getPID] == {rank}}} {{"][1]
            lines = text[start:start + 8]
        at = next(i for i, ln in enumerate(lines)
                  if ln.strip().startswith(f"eleLoad -ele {tag} "))
        assert [ln.strip() for ln in lines[at - 3:at]] == [
            g.strip() for g in _TCL_GUARD], (rank, lines)
        assert sum(ln.strip() == _TCL_GUARD[0] for ln in lines) == 1


@pytest.mark.parametrize("per_rank", [False, True])
def test_a_streamed_force_beam_deck_has_no_guard(
    tmp_path: Path, per_rank: bool,
) -> None:
    path = tmp_path / "s.tcl"
    em = TclEmitter()
    em.stream_to(str(path), per_rank=per_rank)
    _ranked_deck(em, "forceBeamColumn")
    em.stream_finish()
    texts = [path.read_text(encoding="utf-8")]
    if per_rank:
        texts += [p.read_text(encoding="utf-8")
                  for p in sorted((tmp_path / "ranks").glob("*.tcl"))]
    assert not any("ladrunoBuild" in t for t in texts)


def test_a_per_rank_deck_checks_in_the_driver(tmp_path: Path) -> None:
    """Buffered and written per rank, the check sits in the driver before
    any rank block, so every rank runs it before its own eleLoad."""
    from apeGmsh.opensees.apesees import _write_per_rank_tcl

    em = TclEmitter()
    _ranked_deck(em, "LadrunoDispBeamColumn")
    path = tmp_path / "d.tcl"
    _write_per_rank_tcl(str(path), em.line_buffer(), em.partition_spans())
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = [i for i, ln in enumerate(lines)
              if "[getPID] ==" in ln and "source" in ln]
    at = lines.index(_TCL_GUARD[0])
    assert lines[at:at + 3] == _TCL_GUARD
    assert lines.count(_TCL_GUARD[0]) == 1
    assert len(blocks) == 4 and at < min(blocks)
    for rank, tag in ((0, 7), (1, 8)):
        frag = (tmp_path / "ranks" / f"rank{rank}_1.tcl").read_text(
            encoding="utf-8")
        assert f"eleLoad -ele {tag} " in frag
        assert _TCL_GUARD[0] not in frag


def test_the_tcl_guard_on_stub_builds() -> None:
    tkinter = pytest.importorskip("tkinter")
    tcl = tkinter.Tcl()
    guard = "\n".join(_TCL_GUARD)
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
    "trapezoid": {"wy": (-10.0, -20.0), "wz": (-4.0, -8.0)},
}


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(
    "element, load, guarded",
    [
        ("elasticBeamColumn", "full-span", False),
        ("elasticBeamColumn", "partial", True),
        ("elasticBeamColumn", "trapezoid", True),
        ("forceBeamColumn", "partial", False),
        ("forceBeamColumn", "trapezoid", False),
    ],
)
def test_the_bridge_deck_carries_the_guard(
    emitter: Any, element: str, load: str, guarded: bool,
) -> None:
    guard = fork_guard_lines(BEAM3D_PARTIAL_LOAD_GUARD, _dialect(emitter))
    em = emitter()
    _bridge(element, **_LOADS[load]).build().emit(em)
    lines = em.lines()
    assert any("-beamUniform" in ln for ln in lines)
    assert (guard[0] in lines) is guarded
