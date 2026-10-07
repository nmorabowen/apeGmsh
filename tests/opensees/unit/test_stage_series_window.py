"""#1334 — a stage's ``Path`` series that is zero over the stage's whole
pseudo-time window warns when the stage closes.

Every stage closes with ``loadConst -time 0.0``, so a series written on a
global clock across stages reads 0 over a later stage and the deck runs
having applied none of that pattern's loads.

The oracle is OpenSees' own evaluation of the series:
``test_window_semantics_match_opensees`` (live) evaluates the factor at
each increment of a real run, and the warning must fire iff every one of
those factors is zero.  The non-warning cases here are the contract's
silent half: run them with ``-W error::SeriesOutsideStageWindowWarning``.
"""
from __future__ import annotations

import warnings
from typing import Any, cast

import pytest

from apeGmsh.opensees import SeriesOutsideStageWindowWarning
from apeGmsh.opensees.apesees import apeSees

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


def _chain(ops: apeSees, *, integrator: Any = None,
           analysis: Any = None) -> dict[str, object]:
    return {
        "test": ops.test.NormDispIncr(tol=1e-8, max_iter=10),
        "algorithm": ops.algorithm.Newton(),
        "integrator": integrator or ops.integrator.LoadControl(dlam=0.25),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.Plain(),
        "system": ops.system.BandGeneral(),
        "analysis": analysis or ops.analysis.Static(),
    }


def _stage(ops: apeSees, series: Any, *, set_time: float | None = None,
           reset: bool = False, n: int = 4, dt: float | None = 0.25,
           **chain: Any) -> None:
    with ops.stage(name="s2") as s:
        if set_time is not None:
            s.set_time(set_time)
        if reset:
            s.reset()
        with s.pattern(series=series) as p:
            p.load(node=2, forces=(1.0, 0.0, 0.0))
        s.analysis(**_chain(ops, **chain))
        s.run(n_increments=n, dt=dt)


def _caught(fn: Any) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [x for x in w
            if issubclass(x.category, SeriesOutsideStageWindowWarning)]


# ---------------------------------------------------------------------------
# Warns: the series is zero at every pseudo-time the stage reaches
# ---------------------------------------------------------------------------


def test_issue_repro_global_clock_path_warns() -> None:
    """The #1334 reproduction: stage window [0, 1], series on [1, 2]."""
    ops = _ops()
    w = _caught(lambda: _stage(
        ops, ops.timeSeries.Path(time=(1.0, 2.0), values=(0.0, 2.0))))
    assert len(w) == 1
    msg = str(w[0].message)
    assert "'s2'" in msg
    assert "support [1, 2]" in msg
    assert "window [0, 1]" in msg
    assert "s.set_time(1)" in msg
    # The warning names the user's line, not the bridge.
    assert w[0].filename == __file__


@pytest.mark.parametrize("series_kw", [
    # entirely after the window (dt route, start_time shifts the support)
    {"dt": 0.5, "values": (1.0, 2.0, 3.0), "start_time": 2.0},
    # entirely before the window (window moved by set_time below)
    {"time": (0.0, 0.5), "values": (1.0, 1.0)},
    # overlapping support, but zero there: a global history 0 -> 0 -> 2
    {"time": (0.0, 1.0, 2.0), "values": (0.0, 0.0, 2.0)},
    # non-zero only between increments: support ends at 0.2, first
    # increment lands at 0.25
    {"time": (-1.0, 0.2), "values": (1.0, 1.0)},
])
def test_zero_over_window_warns(series_kw: dict[str, Any]) -> None:
    ops = _ops()
    set_time = 5.0 if series_kw.get("time") == (0.0, 0.5) else None
    w = _caught(lambda: _stage(
        ops, ops.timeSeries.Path(**series_kw), set_time=set_time))
    assert len(w) == 1


def test_transient_window_uses_dt() -> None:
    ops = _ops()
    w = _caught(lambda: _stage(
        ops, ops.timeSeries.Path(time=(1.0, 2.0), values=(1.0, 1.0)),
        n=10, dt=0.05,
        integrator=ops.integrator.Newmark(gamma=0.5, beta=0.25),
        analysis=ops.analysis.Transient()))
    assert len(w) == 1
    assert "window [0, 0.5]" in str(w[0].message)


def test_reset_restarts_window_at_zero() -> None:
    """``s.reset()`` before analyze reverts the domain time to 0, so a
    ``set_time`` in the same stage does not move the window."""
    ops = _ops()
    w = _caught(lambda: _stage(
        ops, ops.timeSeries.Path(time=(1.0, 2.0), values=(0.0, 2.0)),
        set_time=1.0, reset=True))
    assert len(w) == 1
    assert "drop s.reset()" in str(w[0].message)


# ---------------------------------------------------------------------------
# Silent: run these with -W error::SeriesOutsideStageWindowWarning
# ---------------------------------------------------------------------------


def test_set_time_fix_is_silent() -> None:
    ops = _ops()
    _stage(ops, ops.timeSeries.Path(time=(1.0, 2.0), values=(0.0, 2.0)),
           set_time=1.0)


@pytest.mark.parametrize("series_kw", [
    {"time": (0.0, 1.0), "values": (0.0, 1.0)},          # matches window
    {"time": (0.5, 3.0), "values": (1.0, 1.0)},          # partial overlap
    {"dt": 0.25, "values": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)},  # dt route
    {"dt": 1.0, "values": (1.0, 2.0), "prepend_zero": True},
])
def test_overlapping_path_is_silent(series_kw: dict[str, Any]) -> None:
    ops = _ops()
    _stage(ops, ops.timeSeries.Path(**series_kw))


def test_dt_window_ending_on_last_sample_warns_mid_stage() -> None:
    """A ``dt=`` series reads 0 at its last sample's time
    (``incr2 >= size``), so a window ending exactly there is not inert
    (no #1334 warning) but drops the load on its last increment (#1363)."""
    from apeGmsh.opensees import SeriesEndsMidStageWarning

    ops = _ops()
    series = ops.timeSeries.Path(dt=0.25, values=(0.0, 1.0, 2.0, 3.0, 4.0))
    with pytest.warns(SeriesEndsMidStageWarning, match="t=1 on"):
        _stage(ops, series)


def test_linear_and_constant_are_silent() -> None:
    ops = _ops()
    _stage(ops, ops.timeSeries.Linear(), set_time=100.0)
    _stage(ops, ops.timeSeries.Constant(), set_time=100.0)


def test_file_path_is_not_checked() -> None:
    """The support of a ``file=`` series lives in a file the deck reads
    at run time; it is not checked."""
    ops = _ops()
    _stage(ops, ops.timeSeries.Path(file="motion.txt", dt=0.01),
           set_time=100.0)


def test_solved_for_advance_is_not_checked() -> None:
    """Displacement control solves for the load-factor (time) advance,
    so no window exists before the run."""
    ops = _ops()
    _stage(ops, ops.timeSeries.Path(time=(10.0, 20.0), values=(1.0, 1.0)),
           integrator=ops.integrator.DisplacementControl(
               node=2, dof=1, dU=0.001))


def test_adaptive_load_control_is_not_checked() -> None:
    """With ``min_lam``/``max_lam`` the increment adapts to the iteration
    count, so the increments' pseudo-times are not known before the run."""
    ops = _ops()
    _stage(ops, ops.timeSeries.Path(time=(10.0, 20.0), values=(1.0, 1.0)),
           integrator=ops.integrator.LoadControl(
               dlam=0.25, num_iter=5, min_lam=0.1, max_lam=0.5))


def test_support_touching_the_last_increment_is_silent() -> None:
    """Support [1, 2], increments at 0.25 .. 1.0: OpenSees applies the
    factor 3 at the last increment, so the pattern is not inert."""
    ops = _ops()
    _stage(ops, ops.timeSeries.Path(time=(1.0, 2.0), values=(3.0, 2.0)))


def test_unknown_analysis_class_raises() -> None:
    """Fail closed: an analysis this check cannot read the clock of is a
    TypeError, not a silent skip."""
    from apeGmsh.opensees._internal.build import StageRecord
    from apeGmsh.opensees.pattern.pattern import Plain
    from apeGmsh.opensees.time_series.time_series import Path

    plain = Plain(series=Path(time=(1.0, 2.0), values=(0.0, 1.0)))
    with pytest.raises(TypeError, match="advances pseudo-time"):
        StageRecord(
            name="odd", initial_stress_records=(), test=None,
            algorithm=None, integrator=None, constraints=None,
            numberer=None, system=None, analysis=None,
            n_increments=4, dt=0.25, pattern_specs=(plain,),
        )


# ---------------------------------------------------------------------------
# Oracle: OpenSees' own factor at every increment of a real run
# ---------------------------------------------------------------------------

# (series kwargs, stage kwargs, expected warning).  Each row runs the
# stage's clock in OpenSees; the warning must fire iff every factor the
# run applies is zero.
_ORACLE_CASES: list[tuple[dict[str, Any], dict[str, Any], bool]] = [
    ({"time": (1.0, 2.0), "values": (0.0, 2.0)}, {}, True),
    ({"dt": 0.5, "values": (1.0, 2.0, 3.0), "start_time": 2.0}, {}, True),
    ({"time": (0.0, 0.5), "values": (1.0, 1.0)}, {"set_time": 5.0}, True),
    ({"time": (0.0, 1.0, 2.0), "values": (0.0, 0.0, 2.0)}, {}, True),
    ({"time": (-1.0, 0.2), "values": (1.0, 1.0)}, {}, True),
    ({"time": (1.0, 2.0), "values": (1.0, 1.0)},
     {"transient": True, "n": 10, "h": 0.05}, True),
    ({"time": (1.0, 2.0), "values": (0.0, 2.0)},
     {"set_time": 1.0, "reset": True}, True),
    # 10 x 0.1 sums to 0.9999999999999999 < 1: the touching factor 3 is
    # never reached, so the stage is inert.
    ({"time": (1.0, 2.0), "values": (3.0, 2.0)}, {"n": 10, "h": 0.1}, True),
    ({"time": (1.0, 2.0), "values": (0.0, 2.0)}, {"set_time": 1.0}, False),
    ({"time": (0.0, 1.0), "values": (0.0, 1.0)}, {}, False),
    ({"time": (0.5, 3.0), "values": (1.0, 1.0)}, {}, False),
    ({"dt": 0.25, "values": (0.0, 1.0, 2.0, 3.0, 4.0)}, {}, False),
    ({"dt": 1.0, "values": (1.0, 2.0), "prepend_zero": True}, {}, False),
    # 4 x 0.25 sums to exactly 1.0: the factor 3 lands on the last step.
    ({"time": (1.0, 2.0), "values": (3.0, 2.0)}, {}, False),
    ({"time": (0.0, 2.0), "values": (1.0, 1.0)},
     {"transient": True, "n": 10, "h": 0.05, "set_time": 1.0}, False),
]


def _opensees_factors(series: Any, *, set_time: float | None = None,
                      reset: bool = False, n: int = 4, h: float = 0.25,
                      transient: bool = False) -> list[float]:
    """Run the stage's clock on a one-spring model in OpenSees and return
    the pattern's load factor after each increment."""
    from apeGmsh.opensees.emitter.live import get_ops
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    try:
        o = get_ops()
    except ImportError as exc:
        pytest.skip(f"no OpenSees backend: {exc}")
    rec = RecordingEmitter()
    series._emit(rec, 1)
    (_, ts_args, _), = rec.calls
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
    if transient:
        o.integrator("Newmark", 0.5, 0.25)
        o.analysis("Transient")
    else:
        o.integrator("LoadControl", h)
        o.analysis("Static")
    if set_time is not None:
        o.setTime(set_time)
    if reset:
        o.reset()
    out = []
    for _ in range(n):
        assert (o.analyze(1, h) if transient else o.analyze(1)) == 0
        out.append(float(o.getLoadFactor(1)))
    o.wipe()
    return out


@pytest.mark.live
@pytest.mark.parametrize(("series_kw", "stage_kw", "expect"), _ORACLE_CASES)
def test_window_semantics_match_opensees(
    series_kw: dict[str, Any], stage_kw: dict[str, Any], expect: bool,
) -> None:
    ops = _ops()
    series = ops.timeSeries.Path(**series_kw)
    factors = _opensees_factors(series, **stage_kw)
    inert = all(f == 0.0 for f in factors)
    assert inert == expect, factors

    kw = dict(stage_kw)
    transient = kw.pop("transient", False)
    h = kw.pop("h", 0.25)
    chain: dict[str, Any] = (
        {"integrator": ops.integrator.Newmark(gamma=0.5, beta=0.25),
         "analysis": ops.analysis.Transient()} if transient
        else {"integrator": ops.integrator.LoadControl(dlam=h)})
    w = _caught(lambda: _stage(ops, series, dt=h, **kw, **chain))
    assert (len(w) == 1) == inert, (factors, [str(x.message) for x in w])
