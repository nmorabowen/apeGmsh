"""Emit and validation tests for the fork flags merged at ``c83ee812e``.

ASD shells ``p_delta=`` (``-pDelta``, fork WP-205), ``eigen_feast(...,
trace_stop=)`` (``-traceStop``, WP-203) and ``response_spectrum_analysis(...,
scale=)`` (``-scale``, WP-202). The numeric oracles are in
``tests/opensees/live/test_ladruno_c83_features_live.py``.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees.element.shell import ASDShellQ4, ASDShellT3
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.section.plate import ElasticMembranePlateSection
from tests.opensees.fixtures.fem_stub import make_two_node_beam


def _section() -> ElasticMembranePlateSection:
    return ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)


def _emit(ele: Any, nodes: tuple[int, ...]) -> tuple[Any, ...]:
    e = RecordingEmitter()
    set_tag_resolver(e, lambda prim: 8)
    set_element_nodes(e, nodes)
    ele._emit(e, tag=20)
    assert len(e.calls) == 1
    return cast("tuple[Any, ...]", e.calls[0][1])


# ---------------------------------------------------------------------------
# ASD shells -pDelta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,nodes",
    [(ASDShellQ4, (41, 42, 43, 44)), (ASDShellT3, (41, 42, 43))],
)
def test_p_delta_emits_the_flag(cls: Any, nodes: tuple[int, ...]) -> None:
    args = _emit(cls(pg="Wall", section=_section(), p_delta=True), nodes)
    assert args == (cls.__name__, 20, *nodes, 8, "-pDelta")


@pytest.mark.parametrize(
    "cls,nodes",
    [(ASDShellQ4, (41, 42, 43, 44)), (ASDShellT3, (41, 42, 43))],
)
def test_default_emits_no_flag(cls: Any, nodes: tuple[int, ...]) -> None:
    args = _emit(cls(pg="Wall", section=_section()), nodes)
    assert "-pDelta" not in args


@pytest.mark.parametrize("cls", [ASDShellQ4, ASDShellT3])
def test_p_delta_with_corotational_is_refused(cls: Any) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        cls(pg="Wall", section=_section(), p_delta=True, corotational=True)


def test_ns_wrapper_threads_p_delta() -> None:
    ops = apeSees(cast(Any, make_two_node_beam()))
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=1.0, nu=0.2, h=0.1)
    assert ops.element.ASDShellQ4(pg="Cols", section=sec, p_delta=True).p_delta
    assert ops.element.ASDShellT3(pg="Cols", section=sec, p_delta=True).p_delta


class _StockOps:
    """Stock-shaped fake: no ``ladrunoBuild``; records element calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def element(self, *args: Any) -> None:
        self.calls.append(args)


class _ForkOps(_StockOps):
    def ladrunoBuild(self) -> str:  # noqa: N802
        return "c" * 40


def _live(ops: object) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[attr-defined]
    le._in_partition = False
    le._fork_verified_types = set()
    return le


def test_live_route_refuses_p_delta_off_the_fork() -> None:
    ops = _StockOps()
    with pytest.raises(RuntimeError, match=r"-pDelta.*Ladruno fork"):
        _live(ops).element("ASDShellQ4", 1, 1, 2, 3, 4, 8, "-pDelta")
    assert ops.calls == []
    # The flag-free element is stock and passes through.
    _live(ops).element("ASDShellQ4", 1, 1, 2, 3, 4, 8)
    assert ops.calls == [("ASDShellQ4", 1, 1, 2, 3, 4, 8)]


def test_live_route_passes_p_delta_on_the_fork() -> None:
    ops = _ForkOps()
    _live(ops).element("ASDShellT3", 1, 1, 2, 3, 8, "-pDelta")
    assert ops.calls == [("ASDShellT3", 1, 1, 2, 3, 8, "-pDelta")]


# ---------------------------------------------------------------------------
# eigen -feast -traceStop
# ---------------------------------------------------------------------------


def test_trace_stop_reaches_every_deck() -> None:
    py, tcl, rec = PyEmitter(), TclEmitter(), RecordingEmitter()
    for e in (py, tcl, rec):
        e.eigen_feast(1.0, 50.0, certify=True, trace_stop=True)
    assert "'-traceStop'" in py.lines()[-1] or '"-traceStop"' in py.lines()[-1]
    assert tcl.lines()[-1].endswith("-certify -traceStop")
    assert rec.calls == [
        ("eigen_feast", (1.0, 50.0), {"certify": True, "trace_stop": True}),
    ]


def test_trace_stop_default_is_unchanged() -> None:
    tcl, rec = TclEmitter(), RecordingEmitter()
    tcl.eigen_feast(1.0, 50.0)
    rec.eigen_feast(1.0, 50.0)
    assert "-traceStop" not in tcl.lines()[-1]
    assert rec.calls == [("eigen_feast", (1.0, 50.0), {"certify": False})]


# ---------------------------------------------------------------------------
# responseSpectrumAnalysis -scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_rsa_scale_must_be_finite_and_positive(bad: float) -> None:
    ops = apeSees(cast(Any, make_two_node_beam()))
    ops.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="scale must be finite and > 0"):
        ops.response_spectrum_analysis(
            1, periods=[0.1, 0.5], accels=[2.0, 1.0],
            combine="SRSS", num_modes=2, scale=bad,
        )
