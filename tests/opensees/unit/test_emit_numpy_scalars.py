"""Numpy scalars reach the emitted decks as plain Python numbers (#1336).

Under numpy >= 2 ``repr(np.float64(1.5))`` is ``'np.float64(1.5)'``. The
deck formatters rendered a token with ``repr``, so a material constant
computed with ``np.sqrt`` reached the openseespy deck as
``np.float64(100.0)`` (the deck died with ``NameError: name 'np' is not
defined``) and the Tcl deck as the same unparsable token.

The oracle is the plain-Python spelling of the same value: a numpy token
must render byte-identically to ``float(v)`` / ``int(v)``. The end-to-end
case is a two-node truss with numpy inputs whose decks must equal the
decks of the same model built from plain floats and ints; the ``live``
case runs the py deck on openseespy and checks the tip displacement
against the closed form ``u = P L / (E A)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.base import StrategySpec, plain_scalar
from apeGmsh.opensees.emitter.py import PyEmitter, _fmt_value as _py_fmt
from apeGmsh.opensees.emitter.py import _ops_call
from apeGmsh.opensees.emitter.tcl import _fmt_value as _tcl_fmt
from apeGmsh.opensees.emitter.tcl import _join

from tests.opensees.fixtures.fem_stub import make_two_node_beam


#: (numpy value, the plain token both decks must carry).
_CASES: list[tuple[Any, str]] = [
    (np.float64(1.5), "1.5"),
    (np.float64(100.0), "100.0"),
    (np.float32(2.0), "2.0"),
    (np.float32(0.5), "0.5"),
    (np.float16(0.25), "0.25"),
    (np.int64(3), "3"),
    (np.int32(-7), "-7"),
    (np.uint8(4), "4"),
    (np.bool_(True), "1"),
    (np.bool_(False), "0"),
    (np.array(2.5), "2.5"),
    (np.array(9, dtype=np.int64), "9"),
]


@pytest.mark.parametrize(("value", "token"), _CASES)
def test_py_formatters_render_numpy_as_plain_numbers(
    value: Any, token: str,
) -> None:
    assert _py_fmt(value) == token
    assert _ops_call("x", value) == f"ops.x({token})"


@pytest.mark.parametrize(("value", "token"), _CASES)
def test_tcl_formatters_render_numpy_as_plain_numbers(
    value: Any, token: str,
) -> None:
    assert _tcl_fmt(value) == token
    assert _join("x", value) == f"x {token}"


def test_issue_formatter_check_gives_plain_tokens() -> None:
    """The exact check the issue reports."""
    args = (np.float64(1.5), np.int64(3), np.float32(2.0))
    assert _ops_call("x", *args) == "ops.x(1.5, 3, 2.0)"
    assert _join("x", *args) == "x 1.5 3 2.0"


def test_float32_widens_exactly_not_through_its_decimal_repr() -> None:
    """``np.float32(0.1)`` is not 0.1; the deck carries its exact value."""
    v = np.float32(0.1)
    assert _py_fmt(v) == repr(float(v)) == "0.10000000149011612"
    assert _tcl_fmt(v) == repr(float(v))


def test_int_and_float_subclasses_render_as_their_base_value() -> None:
    class Half(float):
        def __repr__(self) -> str:
            return "Half()"

    class Count(int):
        def __repr__(self) -> str:
            return "Count()"

        def __str__(self) -> str:
            return "Count()"

    assert _py_fmt(Half(0.5)) == "0.5"
    assert _tcl_fmt(Half(0.5)) == "0.5"
    assert _py_fmt(Count(4)) == "4"
    assert _tcl_fmt(Count(4)) == "4"


@pytest.mark.parametrize("fmt", [_py_fmt, _tcl_fmt])
def test_an_array_with_ndim_above_zero_is_refused(fmt: Any) -> None:
    with pytest.raises(TypeError, match=r"shape \(2,\)"):
        fmt(np.array([1.0, 2.0]))


def test_plain_scalar_passes_non_numpy_through_unchanged() -> None:
    for v in (1, 1.5, "Newton", True, None):
        assert plain_scalar(v) is v


def test_plain_scalar_refuses_a_value_with_no_python_equivalent() -> None:
    v = np.longdouble(1.5)
    if isinstance(v.item(), float):
        # longdouble is the C double here (Windows, MSVC): it IS plain.
        assert plain_scalar(v) == 1.5
    else:
        with pytest.raises(TypeError, match="no plain Python equivalent"):
            plain_scalar(v)


def test_py_strategy_rungs_render_numpy_as_plain_numbers() -> None:
    e = PyEmitter()
    spec = StrategySpec(
        name="lad",
        rungs=(
            ("Newton",),
            ("KrylovNewton", "-maxDim", cast(int, np.int64(3))),
        ),
    )
    e.analyze(steps=1, strategy=spec)
    (line,) = [ln for ln in e.lines() if ln.startswith("_apesees_rungs =")]
    assert line == "_apesees_rungs = [('Newton',), ('KrylovNewton', '-maxDim', 3,)]"


# ---------------------------------------------------------------------------
# End to end: the issue's model, numpy inputs vs plain inputs
# ---------------------------------------------------------------------------

_P = 1.0      # axial tip load
_E = 100.0    # np.sqrt(4.0) * 50.0
_A = 0.5      # exactly representable in float32
_L = 1.0      # make_two_node_beam: node 2 at (0, 0, 1)


def _truss(numpy_inputs: bool) -> apeSees:
    """A vertical two-node truss under an axial tip load.

    ``numpy_inputs`` builds every value the issue saw leaking from numpy
    (``np.sqrt`` for ``E``, ``np.float32`` area, ``np.int64`` iteration
    cap, ``np.float64`` load, factor and ``dlam``); otherwise the same
    values as plain Python literals.
    """
    if numpy_inputs:
        E: Any = np.sqrt(np.float64(4.0)) * 50.0
        A: Any = np.float32(_A)
        P: Any = np.float64(_P)
        one: Any = np.float64(1.0)
        max_iter: Any = np.int64(10)
    else:
        E, A, P, one, max_iter = _E, _A, _P, 1.0, 10
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=E)
    ops.element.Truss(pg="Cols", A=A, material=mat)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(pg="Top", dofs=(1, 1, 0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear(factor=one)) as p:
        p.load(pg="Top", forces=(0.0, 0.0, P))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=max_iter)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=one)
    ops.analysis.Static()
    return ops


def _decks(ops: apeSees, tmp_path: Path, stem: str) -> tuple[str, str]:
    py_path = tmp_path / f"{stem}.py"
    tcl_path = tmp_path / f"{stem}.tcl"
    ops.py(str(py_path))
    ops.tcl(str(tcl_path))
    return (
        py_path.read_text(encoding="utf-8"),
        tcl_path.read_text(encoding="utf-8"),
    )


def test_numpy_inputs_emit_the_same_decks_as_plain_inputs(
    tmp_path: Path,
) -> None:
    py_np, tcl_np = _decks(_truss(numpy_inputs=True), tmp_path, "np")
    py_plain, tcl_plain = _decks(_truss(numpy_inputs=False), tmp_path, "plain")

    assert "np." not in py_np
    assert "np." not in tcl_np
    assert py_np == py_plain
    assert tcl_np == tcl_plain
    # The lines the issue showed leaking, spelled out.
    assert "ops.uniaxialMaterial('Elastic', 1, 100.0)" in py_np.splitlines()
    assert "ops.integrator('LoadControl', 1.0)" in py_np.splitlines()
    assert "uniaxialMaterial Elastic 1 100.0" in tcl_np.splitlines()
    assert "integrator LoadControl 1.0" in tcl_np.splitlines()


class _PlainArgsOps:
    """An ``ops`` stand-in that refuses any non-plain positional value."""

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any) -> int:
            for a in args:
                assert type(a) in (int, float, str), (name, a, type(a))
            return 0
        return call


def test_numpy_py_deck_executes_with_plain_arguments_only(
    tmp_path: Path,
) -> None:
    """The issue's failure: executing the py deck raised ``NameError``."""
    src, _ = _decks(_truss(numpy_inputs=True), tmp_path, "np")
    body = "\n".join(
        ln for ln in src.splitlines()
        if not ln.startswith("import openseespy")
    )
    exec(compile(body, "deck.py", "exec"), {"ops": _PlainArgsOps()})


@pytest.mark.live
def test_numpy_py_deck_runs_on_openseespy_to_the_closed_form(
    tmp_path: Path,
) -> None:
    """Run the emitted py deck on the live backend: ``u = P L / (E A)``."""
    from apeGmsh.opensees.emitter.live import get_ops

    live = get_ops()
    src, _ = _decks(_truss(numpy_inputs=True), tmp_path, "np")
    body = "\n".join(
        ln for ln in src.splitlines()
        if not ln.startswith("import openseespy")
    )
    try:
        exec(compile(body, "deck.py", "exec"), {"ops": live})
        assert live.analyze(1) == 0
        u = live.nodeDisp(2, 3)
    finally:
        live.wipe()
    assert u == pytest.approx(_P * _L / (_E * _A), rel=1e-12)
