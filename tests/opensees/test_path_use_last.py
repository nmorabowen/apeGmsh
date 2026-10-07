"""#1363 — three silent behaviours of ``timeSeries Path``.

1. ``OPS_PathSeries`` builds a ``time=`` series as a ``PathTimeSeries`` and
   never passes it ``-startTime`` or ``-prependZero`` (stock and fork
   alike), and stock also drops ``-useLast`` there.  ``Path`` refuses those
   combinations at construction, before registration.
2. ``use_last=True`` emits ``-useLast``, which holds the last value past
   the support instead of dropping to 0.
3. A stage whose window runs past the series' last sample, while that
   sample is non-zero and ``use_last`` is off, warns with
   :class:`SeriesEndsMidStageWarning`; it stays silent otherwise.  The
   silent half runs under ``-W error::SeriesEndsMidStageWarning``.

The oracle is OpenSees' own factor at every increment of a real run
(``live``): the series drops mid-stage iff holding its last value past
the support changes some increment's factor.  The held twin is
``-useLast`` for a ``dt=`` series (both builds honour it there) and the
time axis extended with the last value for a ``time=`` series (the one
form both builds read the same).
"""
from __future__ import annotations

import warnings
from pathlib import Path as FsPath
from typing import Any, cast

import pytest

from apeGmsh.opensees import (
    SeriesEndsMidStageWarning,
    SeriesOutsideStageWindowWarning,
)
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.time_series.time_series import Path

from tests.opensees.fixtures.fem_stub import make_two_column_frame


def _ops() -> apeSees:
    ops = apeSees(cast("object", make_two_column_frame()),
                  default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Cols", transf=t, A=0.01, E=200e9,
                                  Iz=1e-4)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    return ops


def _register(ops: apeSees, **kw: Any) -> Path:
    """Declare a ``Path`` through the public facade."""
    return ops.timeSeries.Path(**kw)


def _stage(ops: apeSees, series: Any, *, n: int = 8, h: float = 0.25,
           set_time: float | None = None) -> None:
    with ops.stage(name="s2") as s:
        if set_time is not None:
            s.set_time(set_time)
        with s.pattern(series=series) as p:
            p.load(node=2, forces=(1.0, 0.0, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=h),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.Plain(),
            system=ops.system.BandGeneral(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=n, dt=h)


def _caught(fn: Any) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [x for x in w if issubclass(
        x.category, (SeriesEndsMidStageWarning,
                     SeriesOutsideStageWindowWarning))]


def _ts_args(series: Path) -> tuple[Any, ...]:
    rec = RecordingEmitter()
    series._emit(rec, 1)
    (_, ts_args, _), = rec.calls
    return cast("tuple[Any, ...]", ts_args)


# ---------------------------------------------------------------------------
# 1. time= refuses the flags OpenSees drops on that route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("kw", "match"), [
    ({"start_time": 0.5}, "never passes it -startTime"),
    ({"prepend_zero": True}, "never passes it -prependZero"),
    ({"use_last": True}, "stock OpenSees drops -useLast"),
])
def test_time_route_refuses_dropped_flags(kw: dict[str, Any],
                                          match: str) -> None:
    with pytest.raises(BridgeError, match=match):
        Path(time=(0.0, 1.0), values=(0.0, 1.0), **kw)


def test_time_route_refusal_happens_before_registration() -> None:
    """The public wrapper raises before ``_register``: no Path is left in
    the deck."""
    ops = apeSees(cast("object", make_two_column_frame()),
                  default_orientation=None)
    ops.model(ndm=3, ndf=6)
    with pytest.raises(BridgeError):
        ops.timeSeries.Path(time=(0.0, 1.0), values=(0.0, 1.0),
                            start_time=1.0)
    ops.timeSeries.Linear()
    rec = RecordingEmitter()
    ops.build().emit(rec)
    assert [c[1][0] for c in rec.calls if c[0] == "timeSeries"] == ["Linear"]


@pytest.mark.parametrize("kw", [
    {"start_time": 0.5}, {"prepend_zero": True}, {"use_last": True},
])
def test_dt_route_keeps_the_flags(kw: dict[str, Any]) -> None:
    """``PathSeries`` (the ``dt=`` route) honours all three on both builds."""
    args = _ts_args(Path(dt=0.5, values=(0.0, 1.0), **kw))
    flag = {"start_time": "-startTime", "prepend_zero": "-prependZero",
            "use_last": "-useLast"}[next(iter(kw))]
    assert flag in args


def test_time_route_with_default_flags_is_accepted() -> None:
    args = _ts_args(Path(time=(0.0, 1.0), values=(0.0, 1.0), start_time=0.0))
    assert "-startTime" not in args and "-useLast" not in args


# ---------------------------------------------------------------------------
# 2. use_last emits -useLast, and round-trips through model.h5
# ---------------------------------------------------------------------------


def test_use_last_emits_flag() -> None:
    assert _ts_args(Path(dt=0.25, values=(0.0, 1.0), use_last=True)) == (
        "Path", 1, "-values", 0.0, 1.0, "-dt", 0.25, "-useLast")
    assert "-useLast" not in _ts_args(Path(dt=0.25, values=(0.0, 1.0)))


@pytest.mark.parametrize("use_last", [True, False])
def test_use_last_round_trips_through_h5(tmp_path: FsPath,
                                         use_last: bool) -> None:
    """``/opensees/time_series`` stores the emitted tokens, so the flag
    comes back as ``-useLast`` and a file without it reads as no flag."""
    import apeGmsh.opensees.emitter.h5_reader as reader
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    e.model(ndm=1, ndf=1)
    Path(dt=0.25, values=(0.0, 1.0), use_last=use_last)._emit(e, 7)
    out = tmp_path / "model.h5"
    e.write(str(out))
    m = reader.open(str(out))
    try:
        (ts,) = m.time_series()
    finally:
        m.close()
    assert ts.type_token == "Path" and ts.tag == 7
    assert ("-useLast" in ts.args) is use_last


# ---------------------------------------------------------------------------
# 3. Warn-as-contract: warns iff the window passes the support on a
#    non-zero end value without use_last
# ---------------------------------------------------------------------------

# (series kwargs, stage kwargs, expect SeriesEndsMidStageWarning)
_CASES: list[tuple[dict[str, Any], dict[str, Any], bool]] = [
    # The card's case: a ramp over [0, 1] in a [0, 2] stage.
    ({"dt": 0.25, "values": (0.0, 0.25, 0.5, 0.75, 1.0)}, {}, True),
    ({"time": (0.0, 1.0), "values": (0.0, 1.0)}, {}, True),
    # A dt= series reads 0 AT its last sample's time (incr2 >= size):
    # a window ending exactly there still drops on the last increment.
    ({"dt": 0.25, "values": (0.0, 1.0, 2.0, 3.0, 4.0)}, {"n": 4}, True),
    # prepend_zero shifts every sample one dt later.
    ({"dt": 0.5, "values": (1.0, 1.0), "prepend_zero": True}, {}, True),
    ({"dt": 0.5, "values": (1.0, 1.0), "start_time": 0.5}, {}, True),
    # Silent: use_last holds the last value.
    ({"dt": 0.25, "values": (0.0, 0.25, 0.5, 0.75, 1.0), "use_last": True},
     {}, False),
    # Silent: the series ends on 0, so nothing is dropped.
    ({"dt": 0.25, "values": (0.0, 1.0, 0.0)}, {}, False),
    ({"time": (0.0, 1.0, 1.5), "values": (0.0, 1.0, 0.0)}, {}, False),
    # Silent: the support covers the window.
    ({"time": (0.0, 3.0), "values": (0.0, 3.0)}, {}, False),
    ({"dt": 0.25, "values": tuple(float(i) for i in range(10))}, {}, False),
    # Silent: a time= support ending exactly on the last increment reads
    # its last value there (PathTimeSeries: pseudoTime == time1).
    ({"time": (0.0, 1.0), "values": (0.0, 1.0)}, {"n": 4}, False),
]


def _ids(case: tuple[dict[str, Any], dict[str, Any], bool]) -> str:
    kw, st, expect = case
    return f"{'warns' if expect else 'silent'}-{sorted(kw)}-{st}"


@pytest.mark.parametrize(("series_kw", "stage_kw", "expect"), _CASES,
                         ids=[_ids(c) for c in _CASES])
def test_ends_mid_stage_contract(series_kw: dict[str, Any],
                                 stage_kw: dict[str, Any],
                                 expect: bool) -> None:
    ops = _ops()
    series = _register(ops, **series_kw)
    if not expect:
        _stage(ops, series, **stage_kw)  # silent under -W error::...
        return
    w = _caught(lambda: _stage(ops, series, **stage_kw))
    assert [x.category for x in w] == [SeriesEndsMidStageWarning]
    msg = str(w[0].message)
    assert "'s2'" in msg and "drops its load mid-stage" in msg
    assert ("use_last=True" in msg) is (series.dt is not None)
    assert w[0].filename == __file__


def test_inert_stage_warns_only_outside_window() -> None:
    """A series zero over the whole window gets the #1334 warning alone:
    the two warnings never both fire for one pattern."""
    ops = _ops()
    series = _register(ops, dt=0.25, values=(0.0, 1.0))
    w = _caught(lambda: _stage(ops, series, set_time=5.0))
    assert [x.category for x in w] == [SeriesOutsideStageWindowWarning]


def test_use_last_rescues_an_otherwise_inert_stage() -> None:
    """Past the support with ``use_last`` the factor is the last value, so
    the stage is not inert and neither warning fires."""
    ops = _ops()
    series = _register(ops, dt=0.25, values=(0.0, 1.0), use_last=True)
    assert _caught(lambda: _stage(ops, series, set_time=5.0)) == []


# ---------------------------------------------------------------------------
# Oracle: OpenSees' factor at every increment of a real run
# ---------------------------------------------------------------------------


def _opensees_factors(ts_args: tuple[Any, ...], *, n: int = 8,
                      h: float = 0.25,
                      set_time: float | None = None) -> list[float]:
    """Run a one-spring model under ``LoadControl(h)`` and return the
    pattern's load factor after each increment."""
    from apeGmsh.opensees.emitter.live import get_ops

    try:
        o = get_ops()
    except ImportError as exc:
        pytest.skip(f"no OpenSees backend: {exc}")
    o.wipe()
    o.model("basic", "-ndm", 1, "-ndf", 1)
    o.node(1, 0.0)
    o.node(2, 0.0)
    o.fix(1, 1)
    o.uniaxialMaterial("Elastic", 1, 1.0)
    o.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)
    o.timeSeries(*ts_args)
    o.pattern("Plain", 1, 1)
    o.load(2, 1.0)
    o.system("BandGeneral")
    o.numberer("Plain")
    o.constraints("Plain")
    o.test("NormDispIncr", 1e-12, 10)
    o.algorithm("Linear")
    o.integrator("LoadControl", h)
    o.analysis("Static")
    if set_time is not None:
        o.setTime(set_time)
    out = []
    for _ in range(n):
        assert o.analyze(1) == 0
        out.append(float(o.getLoadFactor(1)))
    o.wipe()
    return out


def _held_twin(series: Path) -> tuple[Any, ...]:
    """The series with its last value held past the support, in the form
    both builds read the same."""
    assert series.values is not None
    if series.time is not None:
        return _ts_args(Path(time=(*series.time, 1e6),
                             values=(*series.values, series.values[-1]),
                             factor=series.factor))
    return (*_ts_args(series), "-useLast")


@pytest.mark.live
@pytest.mark.parametrize(("series_kw", "stage_kw", "expect"), _CASES,
                         ids=[_ids(c) for c in _CASES])
def test_ends_mid_stage_matches_opensees(series_kw: dict[str, Any],
                                         stage_kw: dict[str, Any],
                                         expect: bool) -> None:
    series = Path(**series_kw)
    factors = _opensees_factors(_ts_args(series), **stage_kw)
    held = _opensees_factors(_held_twin(series), **stage_kw)
    drops = factors != held and any(f != 0.0 for f in factors)
    assert drops == (expect and not series.use_last), (factors, held)


@pytest.mark.live
def test_use_last_holds_a_ramp_past_its_support() -> None:
    """The card's live check: a ``dt=`` ramp over [0, 1] in a [0, 2]
    stage drops to 0 after t = 1 without ``-useLast`` and holds 1.0 with
    it, on stock openseespy and on the fork alike."""
    ramp = {"dt": 0.25, "values": (0.0, 0.25, 0.5, 0.75, 1.0)}
    dropped = _opensees_factors(_ts_args(Path(**ramp)))
    held = _opensees_factors(_ts_args(Path(**ramp, use_last=True)))
    assert dropped == [0.25, 0.5, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert held == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
