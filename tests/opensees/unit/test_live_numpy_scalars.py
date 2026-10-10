"""Numpy scalars reach openseespy as plain Python numbers on the live route
(#1352, program slice R0-c).

openseespy parses only Python ``int`` / ``float`` / ``str`` / ``bool``
positional arguments. ``np.float64`` passes because it subclasses
``float``; ``np.float32``, ``np.int64``, ``np.bool_`` and 0-d arrays
subclass neither, so a ``Truss(A=np.float32(1.0))`` reached
``LiveOpsEmitter.element`` unchanged and died with ``OpenSeesError``
(#1336 fixed only the py and Tcl decks, #1351). The live route now binds
the module behind ``_CoercingOps``, which passes every positional
argument through ``plain_scalar`` before the real call.

Two oracles. The unit half drives the proxy and the emitter against a
recording fake module and checks the *exact* Python type of every
argument that arrives, which is what openseespy's parser keys on. The
``live`` half runs the #1336 model with numpy inputs on the bound
backend and checks the tip displacement against the closed form
``u = P L / (E A)``; with the proxy removed that test fails at
``LiveOpsEmitter.element`` with ``OpenSeesError``.
"""
from __future__ import annotations

import types
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.emitter.live import LiveOpsEmitter, _CoercingOps

from tests.opensees.fixtures.fem_stub import make_two_node_beam


# ---------------------------------------------------------------------------
# A recording openseespy stand-in
# ---------------------------------------------------------------------------

def _recording_module(*names: str) -> types.ModuleType:
    """A ``ModuleType`` whose ``names`` record ``(name, args, kwargs)``
    into ``mod.calls`` and return ``0``."""
    mod = types.ModuleType("openseespy.opensees")
    mod.calls = []  # type: ignore[attr-defined]

    def _make(name: str) -> Any:
        def _fn(*args: Any, **kwargs: Any) -> int:
            mod.calls.append((name, args, kwargs))  # type: ignore[attr-defined]
            return 0
        return _fn

    for name in names:
        setattr(mod, name, _make(name))
    return mod


_VERBS = (
    "wipe", "model", "node", "fix", "element", "uniaxialMaterial",
    "timeSeries", "pattern", "load", "test", "integrator", "analyze",
    "nodeDisp",
)


@pytest.fixture
def mod(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A recording fake bound as the live backend for the test."""
    m = _recording_module(*_VERBS)
    monkeypatch.setattr(live, "_get_ops", lambda: m)
    return m


def _only_call(m: types.ModuleType) -> tuple[Any, ...]:
    (call,) = m.calls  # type: ignore[attr-defined]
    return cast("tuple[Any, ...]", call[1])


# ---------------------------------------------------------------------------
# The proxy
# ---------------------------------------------------------------------------

#: (numpy value, the Python value openseespy must see). The type must
#: match exactly, not by ``isinstance``: that is what the parser keys on.
_CASES: list[tuple[Any, Any]] = [
    (np.float32(0.5), 0.5),
    (np.float32(0.1), float(np.float32(0.1))),   # widens exactly
    (np.float64(100.0), 100.0),
    (np.float16(0.25), 0.25),
    (np.int64(3), 3),
    (np.int32(-7), -7),
    (np.uint8(4), 4),
    (np.bool_(True), True),
    (np.array(2.5), 2.5),
    (np.array(9, dtype=np.int64), 9),
    (np.str_("Newton"), "Newton"),
]


@pytest.mark.parametrize(("value", "plain"), _CASES)
def test_proxy_passes_numpy_scalars_as_their_exact_python_type(
    value: Any, plain: Any,
) -> None:
    m = _recording_module("element")
    _CoercingOps(m).element("Truss", value, 1)
    _, got, _ = _only_call(m)
    assert got == plain
    assert type(got) is type(plain), (got, type(got))


def test_proxy_forwards_plain_arguments_as_the_same_tuple() -> None:
    """The fast path: nothing to coerce, so the tuple is not rebuilt."""
    seen: list[tuple[Any, ...]] = []
    m = types.ModuleType("openseespy.opensees")
    m.node = lambda *args: seen.append(args)  # type: ignore[attr-defined]
    args = (1, 0.0, 2.5, "-ndf", True)
    assert live._coerce_args(args) is args
    _CoercingOps(m).node(*args)
    assert seen == [args]


def test_proxy_passes_non_numpy_values_through_unchanged() -> None:
    """``str``, ``None``, a list, a tuple, a custom object: untouched. A
    container is not walked either (no live call site passes one)."""
    m = _recording_module("recorder")
    sentinel = object()
    inner = [np.float32(1.0), np.int64(2)]
    _CoercingOps(m).recorder("Node", None, inner, (1, 2), sentinel)
    kind, none, got_list, got_tuple, got_obj = _only_call(m)
    assert kind == "Node" and none is None and got_obj is sentinel
    assert got_list is inner and got_tuple == (1, 2)
    assert type(got_list[0]) is np.float32


def test_proxy_keeps_keyword_arguments() -> None:
    m = _recording_module("analyze")
    _CoercingOps(m).analyze(np.int64(1), dt=np.float32(0.5))
    (call,) = m.calls  # type: ignore[attr-defined]
    assert call == ("analyze", (1,), {"dt": np.float32(0.5)})


def test_an_array_with_elements_raises_before_the_call() -> None:
    """``plain_scalar`` refuses an ``ndim > 0`` array; the refusal must
    land before openseespy registers anything."""
    m = _recording_module("element")
    with pytest.raises(TypeError, match=r"shape \(2,\)"):
        _CoercingOps(m).element("Truss", 1, np.array([1.0, 2.0]))
    assert m.calls == []  # type: ignore[attr-defined]


def test_proxy_attribute_access_matches_the_module() -> None:
    m = _recording_module("wipe")
    m.version = "3.7.1"  # type: ignore[attr-defined]
    proxy = _CoercingOps(m)
    assert proxy.module is m
    # A verb the build lacks: the capability probes still read None/False.
    assert getattr(proxy, "contactSurface", None) is None
    assert not hasattr(proxy, "contactSurface")
    with pytest.raises(AttributeError):
        proxy.contactSurface
    # A non-callable attribute is returned as-is.
    assert proxy.version == "3.7.1"
    # The wrapper is cached per verb and keyed on the module function.
    assert proxy.wipe is proxy.wipe
    assert proxy.wipe.__wrapped__ is m.wipe  # type: ignore[attr-defined]


def test_proxy_honours_a_function_swapped_on_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = _recording_module("analyze")
    proxy = _CoercingOps(m)
    assert proxy.analyze(1) == 0
    monkeypatch.setattr(m, "analyze", lambda *a: 7)
    assert proxy.analyze(1) == 7


# ---------------------------------------------------------------------------
# The emitter binds the proxy, in and out of partition blocks
# ---------------------------------------------------------------------------

def test_emitter_binds_the_proxy_on_both_bindings(
    mod: types.ModuleType,
) -> None:
    e = LiveOpsEmitter(wipe=True)
    assert isinstance(e._ops, _CoercingOps)
    assert e._real_ops is e._ops
    assert e.ops is e._ops
    assert e._ops.module is mod
    assert mod.calls == [("wipe", (), {})]  # type: ignore[attr-defined]


def test_emitter_verbs_coerce_through_the_proxy(
    mod: types.ModuleType,
) -> None:
    """The #1336 argument kinds, through the emitter's own verbs: an
    ``np.int64`` tag and count, an ``np.float32`` area, and node
    coordinates spread from a ``float32`` array."""
    e = LiveOpsEmitter(wipe=False)
    e.model(ndm=3, ndf=3)
    e.node(cast(int, np.int64(1)), *np.array([0.0, 0.0, 1.0], np.float32))
    e.element("Truss", cast(int, np.int64(2)), 1, 2, np.float32(0.5), 1)
    e.test("NormDispIncr", 1e-10, cast(int, np.int64(10)))
    e.fix(1, *np.array([1, 1, 1], dtype=np.int64))
    for name, args, _ in mod.calls:  # type: ignore[attr-defined]
        for a in args:
            assert type(a) in (int, float, str), (name, a, type(a))
    assert ("node", (1, 0.0, 0.0, 1.0), {}) in mod.calls  # type: ignore[attr-defined]
    assert ("element", ("Truss", 2, 1, 2, 0.5, 1), {}) in mod.calls  # type: ignore[attr-defined]
    assert ("fix", (1, 1, 1, 1), {}) in mod.calls  # type: ignore[attr-defined]


def test_partition_close_restores_the_coercing_binding(
    mod: types.ModuleType,
) -> None:
    e = LiveOpsEmitter(wipe=False)
    with pytest.warns(UserWarning, match="single-process"):
        e.partition_open(1)
    assert isinstance(e._ops, live._NoOpOps)
    e.node(1, np.float32(0.0))          # swallowed, never reaches the fake
    e.partition_close()
    assert e._ops is e._real_ops
    assert isinstance(e._ops, _CoercingOps)
    e.node(2, np.float32(3.0))
    assert mod.calls == [("node", (2, 3.0), {})]  # type: ignore[attr-defined]
    assert type(mod.calls[0][1][1]) is float  # type: ignore[attr-defined]


def test_backend_verdict_is_keyed_on_the_unwrapped_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_backend_of`` unwraps the proxy, so the emitter reads the
    resolver's cached verdict for the bound module (one probe per
    process, not one per tet10 element)."""
    m = _recording_module("wipe")
    monkeypatch.setattr(live, "_OPS_CACHE", m)
    monkeypatch.setattr(live, "_BACKEND_INFO", None)
    verdict = live.get_backend_info()
    assert live._backend_of(_CoercingOps(m)) is verdict
    assert live._backend_of(m) is verdict
    e = LiveOpsEmitter(wipe=False)
    assert e._backend() is verdict


# ---------------------------------------------------------------------------
# Live: the #1336 model with numpy inputs runs on the bound backend
# ---------------------------------------------------------------------------

_P = 1.0      # axial tip load
_E = 100.0    # np.sqrt(4.0) * 50.0
_A = 0.5      # exactly representable in float32
_L = 1.0      # make_two_node_beam: node 2 at (0, 0, 1)


def _truss_with_numpy_inputs() -> Any:
    """The #1336 truss with every argument kind #1352 names: an
    ``np.float32`` area and load, an ``np.int64`` iteration cap, and
    ``np.float64`` modulus, factor and ``dlam``."""
    from apeGmsh.opensees import apeSees

    E: Any = np.sqrt(np.float64(4.0)) * 50.0
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=E)
    ops.element.Truss(pg="Cols", A=np.float32(_A), material=mat)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(pg="Top", dofs=(1, 1, 0))
    series = ops.timeSeries.Linear(factor=np.float64(1.0))
    with ops.pattern.Plain(series=series) as p:
        p.load(pg="Top", forces=(0.0, 0.0, np.float32(_P)))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=cast(int, np.int64(10)))
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=np.float64(1.0))
    ops.analysis.Static()
    return ops


@pytest.mark.live
def test_numpy_inputs_run_on_the_live_route_to_the_closed_form() -> None:
    """Fails with the proxy removed: ``OpenSeesError`` from
    ``LiveOpsEmitter.element`` on ``Truss(A=np.float32(0.5))``."""
    from apeGmsh.opensees.emitter.live import get_ops

    backend = get_ops()
    ops = _truss_with_numpy_inputs()
    try:
        assert ops.analyze(steps=1) == 0
        u = backend.nodeDisp(2, 3)
    finally:
        backend.wipe()
    assert u == pytest.approx(_P * _L / (_E * _A), rel=1e-12)


@pytest.mark.live
def test_numpy_node_coordinates_reach_the_live_domain() -> None:
    """A tag and coordinates taken straight from numpy arrays, through
    the emitter's ``node`` verb on the real backend."""
    e = LiveOpsEmitter(wipe=True)
    try:
        e.model(ndm=3, ndf=3)
        coords = np.array([0.25, 0.5, 1.0], dtype=np.float32)
        e.node(cast(int, np.int64(1)), *coords)
        assert list(e.ops.nodeCoord(1)) == [0.25, 0.5, 1.0]
        assert list(e.ops.nodeCoord(np.int64(1))) == [0.25, 0.5, 1.0]
    finally:
        e.ops.wipe()
