"""Emit and validation tests for the fork flags merged at ``c83ee812e``.

ASD shells ``p_delta=`` (``-pDelta``, fork WP-205), ``eigen_feast(...,
trace_stop=)`` (``-traceStop``, WP-203), ``response_spectrum_analysis(...,
scale=)`` (WP-202, applied to ``Sa`` in Python) and the build checks that
hold ``-pDelta`` and ``EigenResult.modal_properties`` after FEAST to a build
measured to carry ``c83ee812e``. The numeric oracles are in
``tests/opensees/live/test_ladruno_c83_features_live.py``.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.stage_emit import (
    ASDSHELL_PDELTA_GUARD,
    ASDSHELL_PDELTA_MESSAGE,
    FEAST_MODAL_PROPERTIES_MESSAGE,
    FORK_BUILD_HAS_C83,
    fork_build_has_c83,
    fork_guard_lines,
)
from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._target import backend_info_of
from apeGmsh.opensees.analysis.eigen import EigenResult
from apeGmsh.opensees.element.shell import ASDShellQ4, ASDShellT3
from apeGmsh.opensees.emitter import live
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


_PIN = "c9004f48db4849473c4e277665bc9fd63cb84f87"
#: The maintainer's venv build, which predates c83ee812e.
_STALE_FORK = "79e0623670c61c768357a01dff1477a4a0ef7e06"


def _pin_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    for line in (root / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


def test_the_c83_table_lists_the_pin_and_not_the_stale_build() -> None:
    carries = {sha for sha, ok in FORK_BUILD_HAS_C83.items() if ok}
    assert _pin_sha() in carries, "FORK_PIN carries c83ee812e; measure a bump"
    assert _STALE_FORK not in carries
    assert set(ASDSHELL_PDELTA_GUARD.builds) == carries
    assert ASDSHELL_PDELTA_GUARD.stock_runs is False
    assert not fork_build_has_c83(None)
    assert not fork_build_has_c83("c" * 40)


class _StockOps:
    """Stock-shaped fake: no ``ladrunoBuild``; records element calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def element(self, *args: Any) -> None:
        self.calls.append(args)


class _ForkOps(_StockOps):
    def __init__(self, build: str) -> None:
        super().__init__()
        self._build = build

    def ladrunoBuild(self) -> str:  # noqa: N802
        return self._build


def _live(ops: object) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[attr-defined]
    le._in_partition = False
    le._fork_verified_types = set()
    return le


@pytest.mark.parametrize(
    "ops", [_StockOps(), _ForkOps(_STALE_FORK), _ForkOps("c" * 40)],
    ids=["stock", "stale-fork", "unmeasured-fork"],
)
def test_live_route_refuses_p_delta_below_c83(ops: _StockOps) -> None:
    with pytest.raises(RuntimeError) as err:
        _live(ops).element("ASDShellQ4", 1, 1, 2, 3, 4, 8, "-pDelta")
    assert str(err.value) == ASDSHELL_PDELTA_MESSAGE
    assert ops.calls == []
    # The flag-free element runs on any build.
    _live(ops).element("ASDShellQ4", 1, 1, 2, 3, 4, 8)
    assert ops.calls == [("ASDShellQ4", 1, 1, 2, 3, 4, 8)]


def test_live_route_passes_p_delta_on_a_c83_build() -> None:
    ops = _ForkOps(_PIN)
    _live(ops).element("ASDShellT3", 1, 1, 2, 3, 8, "-pDelta")
    assert ops.calls == [("ASDShellT3", 1, 1, 2, 3, 8, "-pDelta")]


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
def test_a_p_delta_deck_checks_the_build_once(emitter: Any) -> None:
    dialect = "tcl" if emitter is TclEmitter else "py"
    guard = fork_guard_lines(ASDSHELL_PDELTA_GUARD, dialect)
    em = emitter()
    em.element("ASDShellQ4", 1, 1, 2, 3, 4, 8, "-pDelta")
    em.element("ASDShellT3", 2, 1, 2, 3, 8, "-pDelta")
    lines = em.lines()
    assert lines.count(guard[0]) == 1
    at = lines.index(guard[0])
    assert lines[at:at + len(guard)] == guard
    assert at < next(i for i, ln in enumerate(lines) if "ASDShellQ4" in ln)


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
def test_a_deck_without_p_delta_has_no_c83_guard(emitter: Any) -> None:
    em = emitter()
    em.element("ASDShellQ4", 1, 1, 2, 3, 4, 8, "-corotational")
    assert not any("ladrunoBuild" in ln for ln in em.lines())


def test_the_tcl_p_delta_guard_on_stub_builds() -> None:
    tkinter = pytest.importorskip("tkinter")
    tcl = tkinter.Tcl()
    guard = "\n".join(fork_guard_lines(ASDSHELL_PDELTA_GUARD, "tcl"))
    with pytest.raises(tkinter.TclError) as err:
        tcl.eval(guard)  # stock: no ladrunoBuild, stops
    assert str(err.value) == ASDSHELL_PDELTA_MESSAGE
    for answer, ok in ((_PIN, True), (f" {_PIN}\n", True),
                       (_STALE_FORK, False), ("", False), ("0" * 40, False)):
        tcl.eval(f"proc ladrunoBuild {{}} {{ return {{{answer}}} }}")
        if ok:
            tcl.eval(guard)
            continue
        with pytest.raises(tkinter.TclError) as err:
            tcl.eval(guard)
        assert str(err.value) == ASDSHELL_PDELTA_MESSAGE, answer


def test_the_py_p_delta_guard_on_stub_builds() -> None:
    code = compile(
        "\n".join(fork_guard_lines(ASDSHELL_PDELTA_GUARD, "py")),
        "<guard>", "exec")

    def raising() -> str:
        raise RuntimeError("boom")

    with pytest.raises(SystemExit) as err:
        exec(code, {"ops": types.SimpleNamespace()})  # stock stops
    assert err.value.code == ASDSHELL_PDELTA_MESSAGE
    for build, ok in ((lambda: _PIN, True), (lambda: _PIN + "\n", True),
                      (lambda: _STALE_FORK, False), (lambda: "", False),
                      (raising, False)):
        engine = {"ops": types.SimpleNamespace(ladrunoBuild=build)}
        if ok:
            exec(code, engine)
            continue
        with pytest.raises(SystemExit) as err:
            exec(code, engine)
        assert err.value.code == ASDSHELL_PDELTA_MESSAGE


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


class _RsaLive:
    """Stand-in for the live emitter: records the RSA call only."""

    rsa: tuple[Any, ...] = ()
    caps = LiveOpsEmitter.caps

    def __init__(self, *, wipe: bool = True) -> None:
        del wipe

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None

    def eigen(self, num_modes: int, *, solver: str) -> list[float]:
        del solver
        return [4.0, 9.0][:num_modes]

    def modal_properties(self, **_: Any) -> dict[str, list[float]]:
        return {}

    def response_spectrum_analysis(self, *args: Any) -> None:
        type(self).rsa = args


def test_rsa_scale_multiplies_sa_and_emits_no_scale_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact on every build: the fork's -scale is ignored before c83ee812e."""
    monkeypatch.setattr(live, "LiveOpsEmitter", _RsaLive)
    ops = apeSees(cast(Any, make_two_node_beam()))
    ops.model(ndm=3, ndf=6)
    ops.response_spectrum_analysis(
        1, periods=[0.1, 0.5], accels=[2.0, 1.0],
        combine="SRSS", num_modes=2, scale=2.5,
    )
    assert _RsaLive.rsa == (
        1, "-Tn", 0.1, 0.5, "-Sa", 5.0, 2.5, "-combine", "SRSS",
    )
    ops.response_spectrum_analysis(
        1, periods=[0.1, 0.5], accels=[2.0, 1.0],
        combine="SRSS", num_modes=2,
    )
    assert _RsaLive.rsa == (
        1, "-Tn", 0.1, 0.5, "-Sa", 2.0, 1.0, "-combine", "SRSS",
    )


# ---------------------------------------------------------------------------
# EigenResult.modal_properties after eigen_feast
# ---------------------------------------------------------------------------


class _PropsLive:
    def __init__(self) -> None:
        self.calls = 0

    def modal_properties(self, *, unorm: bool = False) -> dict[str, Any]:
        del unorm
        self.calls += 1
        return {}


def _result(feast: bool) -> tuple[EigenResult, _PropsLive]:
    fake = _PropsLive()
    return EigenResult(
        eigenvalues=np.array([4.0, 9.0]), _live=cast(Any, fake),
        _bridge=cast(Any, object()), _ndm=3, _ndf=6,
        _epoch=live.domain_epoch(), _feast=feast,
    ), fake


def _bind_build(monkeypatch: pytest.MonkeyPatch, ops: object) -> None:
    info = backend_info_of(ops)
    monkeypatch.setattr(live, "get_backend_info", lambda: info)


@pytest.mark.parametrize(
    "ops", [_StockOps(), _ForkOps(_STALE_FORK), _ForkOps("c" * 40)],
    ids=["stock", "stale-fork", "unmeasured-fork"],
)
def test_feast_modal_properties_is_refused_below_c83(
    monkeypatch: pytest.MonkeyPatch, ops: object,
) -> None:
    """The fork would end the process, so the call never reaches it."""
    _bind_build(monkeypatch, ops)
    result, fake = _result(feast=True)
    with pytest.raises(RuntimeError) as err:
        result.modal_properties()
    assert str(err.value) == FEAST_MODAL_PROPERTIES_MESSAGE
    assert fake.calls == 0


def test_feast_modal_properties_runs_on_a_c83_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_build(monkeypatch, _ForkOps(_PIN))
    result, fake = _result(feast=True)
    result.modal_properties()
    assert fake.calls == 1


def test_arpack_modal_properties_runs_on_an_older_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_build(monkeypatch, _ForkOps(_STALE_FORK))
    result, fake = _result(feast=False)
    result.modal_properties()
    assert fake.calls == 1
