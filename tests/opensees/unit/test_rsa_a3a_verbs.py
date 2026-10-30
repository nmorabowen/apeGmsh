"""Chain A link A3a: the chain M verbs and ``-requireMass`` on the bridge.

* :class:`ResidualModes` / :class:`RitzVectors` validate as the fork's
  parsers do (WP-213, WP-215);
* ``apeSees.ritz_vectors`` and ``residual_modes`` refuse an MPI (SP)
  model before emitting;
* without chain M the verbs and ``require_mass=`` refuse through the A4
  probe (``ForkTooOldError``) before the model is built;
* the one-shots run their commands in the fixed order (basis, residual,
  ``modalProperties``) and ``require_mass`` reaches
  ``responseSpectrumAnalysis`` as ``-requireMass``.

Every engine here is a stub, so nothing runs OpenSees; the live run is
``tests/opensees/integration_ladruno/test_rsa_a3a_verbs_live.py``.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, cast

import pytest

from apeGmsh.opensees import _target, apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._target import ForkTooOldError
from apeGmsh.opensees.analysis.modal import ResidualModes, RitzVectors
from apeGmsh.opensees.emitter import live

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
    make_two_node_beam,
)

TN = [0.0, 0.5, 2.0]
SA = [3.0, 7.5, 1.0]


# -- primitive validation ----------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"dirs": ()}, {"dirs": (0,)}, {"dirs": (4,)}, {"dirs": (1, 1)},
    {"dirs": (1,), "period": 0.0}, {"dirs": (1,), "period": -0.1},
    {"dirs": (1,), "period": float("inf")}, {"dirs": (1,), "period": "eigen"},
    {"dirs": (1,), "tol": 0.0}, {"dirs": (1,), "tol": 1.0},
])
def test_residual_modes_refuses_what_the_fork_refuses(kw: dict) -> None:
    with pytest.raises(ValueError, match="ResidualModes"):
        ResidualModes(**kw)


@pytest.mark.parametrize("kw", [
    {"n": 0, "dirs": (1,)}, {"n": 2.5, "dirs": (1,)}, {"n": 3},
    {"n": 3, "dirs": (1, 1)}, {"n": 3, "patterns": (2, 2)},
    {"n": 3, "dirs": (1,), "cycles": (1, 1)},
    {"n": 3, "dirs": (1,), "cycles": (-1,)}, {"n": 3, "dirs": (1,), "tol": 2.0},
])
def test_ritz_vectors_refuses_what_the_fork_refuses(kw: dict) -> None:
    with pytest.raises(ValueError, match="RitzVectors"):
        RitzVectors(**kw)


def test_rayleigh_period_and_defaults_emit_nothing() -> None:
    assert ResidualModes(dirs=(2,)).args() == ("-dir", 2)
    assert ResidualModes(dirs=(2,), period=1).args() == ("-dir", 2, "-period", 1.0)


# -- the bridge verbs --------------------------------------------------------

def _stub(*commands: str) -> ModuleType:
    mod = ModuleType("opensees")
    setattr(mod, "ladrunoBuild", lambda: "f" * 40)
    for name in ("eigen", "modalProperties", "responseSpectrumAnalysis", *commands):
        setattr(mod, name, lambda *a: None)
    return mod


OLD = ("modalResponseHistory",)                     # a fork before chain M
NEW = ("modalResponseHistory", "residualModes", "ritzVectors")


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_target, "_RSA_CACHE", {})


def _engine(monkeypatch: pytest.MonkeyPatch, commands: tuple[str, ...]) -> None:
    mod = _stub(*commands)
    monkeypatch.setattr(live, "_get_ops", lambda: mod)


def _bridge(fem: Any = None, *, analysis: bool = True) -> apeSees:
    ops = apeSees(cast("object", fem or make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(pg="Cols", transf=tr, A=0.01, E=200e9,
                                  Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(100.0, 100.0, 100.0, 0.0, 0.0, 0.0))
    if analysis:
        ops.constraints.Transformation()
        ops.numberer.RCM()
        ops.system.BandGeneral()
        ops.test.NormDispIncr(tol=1e-10, max_iter=10)
        ops.algorithm.Linear()
        ops.integrator.LoadControl(dlam=1.0)
        ops.analysis.Static()
    return ops


def _no_build(ops: apeSees, monkeypatch: pytest.MonkeyPatch) -> None:
    def build() -> None:
        raise AssertionError("the verb built the model before refusing")
    monkeypatch.setattr(ops, "build", build)


class _FakeLive:
    """Stands in for ``LiveOpsEmitter``: records the one-shot's steps."""

    steps: list[tuple[str, tuple[Any, ...]]] = []

    def __init__(self, *, wipe: bool) -> None:
        _FakeLive.steps = []

    def eigen(self, n: int, *, solver: str) -> list[float]:
        self.steps.append(("eigen", (n, solver)))
        return [1.0] * n

    def command(self, verb: str, *args: Any) -> None:
        self.steps.append((verb, args))

    def modal_properties(self, **kw: Any) -> dict[str, list[float]]:
        self.steps.append(("modalProperties", ()))
        return {"eigenLambda": [4.0, 9.0, 100.0]}

    def response_spectrum_analysis(self, direction: int, *args: Any) -> None:
        self.steps.append(("responseSpectrumAnalysis", (direction, *args)))


class _FakeBuilt:
    def emit(self, emitter: Any) -> None:
        pass


def _fake_run(ops: apeSees, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "build", lambda: _FakeBuilt())
    monkeypatch.setattr(live, "LiveOpsEmitter", _FakeLive)


def test_residual_modes_runs_eigen_then_residual_then_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine(monkeypatch, NEW)
    ops = _bridge()
    _fake_run(ops, monkeypatch)
    res = ops.residual_modes((1, 2), num_modes=2, period=0.05,
                             solver="-fullGenLapack")
    assert _FakeLive.steps == [
        ("eigen", (2, "-fullGenLapack")),
        ("residualModes", ("-dir", 1, 2, "-period", 0.05)),
        ("modalProperties", ()),
    ]
    assert list(res.eigenvalues) == [4.0, 9.0, 100.0]   # the completed set


def test_ritz_vectors_replaces_eigen(monkeypatch: pytest.MonkeyPatch) -> None:
    _engine(monkeypatch, NEW)
    ops = _bridge()
    with ops.pattern.Plain(series=ops.timeSeries.Linear(), name="g") as p:
        p.load(node=2, forces=(0.0, 0.0, -1.0, 0.0, 0.0, 0.0))
    _fake_run(ops, monkeypatch)
    ops.ritz_vectors(3, dirs=(1,), patterns=("g",), cycles=(0, 2))
    tag = ops.tag_for(p)
    assert _FakeLive.steps == [
        ("ritzVectors", (3, "-dir", 1, "-pattern", tag, "-cycles", 0, 2)),
        ("modalProperties", ()),
    ]


_MODAL_BASIS_VERBS = [
    pytest.param(lambda ops: ops.residual_modes((1,), num_modes=2),
                 "residualModes", id="residual_modes"),
    pytest.param(lambda ops: ops.ritz_vectors(2, dirs=(1,)),
                 "ritzVectors", id="ritz_vectors"),
]


@pytest.mark.parametrize(("call", "command"), _MODAL_BASIS_VERBS)
def test_modal_basis_verbs_refuse_a_partitioned_model(
    call: Any, command: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine(monkeypatch, NEW)
    ops = _bridge(make_two_column_frame_partitioned())
    _no_build(ops, monkeypatch)
    with pytest.raises(BridgeError, match=f"refuses {command} on a partitioned"):
        call(ops)


@pytest.mark.parametrize(("call", "command"), _MODAL_BASIS_VERBS)
def test_modal_basis_verbs_refuse_an_mpi_system(
    call: Any, command: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine(monkeypatch, NEW)
    ops = _bridge(analysis=False)
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.Mumps()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    _no_build(ops, monkeypatch)
    with pytest.raises(BridgeError, match="system Mumps runs under MPI"):
        call(ops)


def test_ritz_vectors_needs_an_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    _engine(monkeypatch, NEW)
    ops = _bridge(analysis=False)
    _no_build(ops, monkeypatch)
    with pytest.raises(BridgeError, match="declare one first"):
        ops.ritz_vectors(2, dirs=(1,))


@pytest.mark.parametrize("call", [
    lambda ops: ops.residual_modes((1,), num_modes=2),
    lambda ops: ops.ritz_vectors(2, dirs=(1,)),
    lambda ops: ops.response_spectrum_analysis(
        1, periods=TN, accels=SA, combine="SRSS", num_modes=2,
        require_mass=0.9),
], ids=["residual_modes", "ritz_vectors", "require_mass"])
def test_an_old_fork_refuses_through_the_probe(
    call: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine(monkeypatch, OLD)
    ops = _bridge()
    _no_build(ops, monkeypatch)
    with pytest.raises(ForkTooOldError, match="fork too old: needs chain M"):
        call(ops)


@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5])
def test_require_mass_is_a_ratio(ratio: float, monkeypatch: pytest.MonkeyPatch) -> None:
    _engine(monkeypatch, NEW)
    ops = _bridge()
    _no_build(ops, monkeypatch)
    with pytest.raises(ValueError, match=r"require_mass must be in \(0, 1\]"):
        ops.response_spectrum_analysis(1, periods=TN, accels=SA, combine="SRSS",
                                       num_modes=2, require_mass=ratio)


def test_require_mass_reaches_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    _engine(monkeypatch, NEW)
    ops = _bridge()
    _fake_run(ops, monkeypatch)
    ops.response_spectrum_analysis(1, periods=TN, accels=SA, combine="SRSS",
                                   num_modes=2, require_mass=0.9)
    rsa = _FakeLive.steps[-1]
    assert rsa == ("responseSpectrumAnalysis",
                   (1, "-Tn", *TN, "-Sa", *SA, "-combine", "SRSS",
                    "-requireMass", 0.9))


def test_without_require_mass_the_line_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engine(monkeypatch, OLD)                 # no chain M needed without it
    ops = _bridge()
    _fake_run(ops, monkeypatch)
    ops.response_spectrum_analysis(1, periods=TN, accels=SA, combine="SRSS",
                                   num_modes=2)
    assert _FakeLive.steps[-1] == (
        "responseSpectrumAnalysis", (1, "-Tn", *TN, "-Sa", *SA, "-combine", "SRSS"))
