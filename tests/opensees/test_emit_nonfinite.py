"""A non-finite float (nan / +-inf) is refused at emit (#1356).

The deck formatters rendered ``float("nan")`` as a bare ``nan`` token: the
openseespy deck died with ``NameError: name 'nan' is not defined`` and
Tcl's ``Tcl_GetDouble`` rejects ``nan`` / ``inf``. Both emitters now raise
:class:`BridgeError` at emit, naming the command and the argument.

The oracle is the refusal itself: every non-finite spelling (plain float,
numpy scalar, a member of a list argument) raises for py and Tcl, the
message names the command and the argument position, every finite value
still renders exactly as before, and ``ops.py`` / ``ops.tcl`` leave no deck
on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.emitter.base import StrategySpec
from apeGmsh.opensees.emitter.py import PyEmitter, _ops_call
from apeGmsh.opensees.emitter.tcl import _join

from tests.opensees.fixtures.fem_stub import make_two_node_beam


_NONFINITE: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    np.float64("nan"),
    np.float32("inf"),
    np.array(float("-inf")),
]


def _py_line(*args: Any) -> str:
    return _ops_call("uniaxialMaterial", "Elastic", 1, *args)


def _tcl_line(*args: Any) -> str:
    return _join("uniaxialMaterial", "Elastic", 1, *args)


# The argument index counts positional arguments after the command word,
# so ``Elastic`` is 0, the tag is 1 and the offender is 2 in both decks.
_RENDERERS: list[tuple[str, Callable[..., str], str]] = [
    ("py", _py_line, r"ops\.uniaxialMaterial\('Elastic', \.\.\.\): argument 2 "),
    ("tcl", _tcl_line, r"uniaxialMaterial Elastic: argument 2 "),
]
_IDS = [r[0] for r in _RENDERERS]


@pytest.mark.parametrize("value", _NONFINITE, ids=repr)
@pytest.mark.parametrize(("backend", "render", "head"), _RENDERERS, ids=_IDS)
def test_a_nonfinite_argument_raises_naming_command_and_argument(
    backend: str, render: Callable[..., str], head: str, value: Any,
) -> None:
    with pytest.raises(BridgeError, match=head + r".*non-finite float"):
        render(value)


@pytest.mark.parametrize(("backend", "render", "head"), _RENDERERS, ids=_IDS)
def test_a_nan_inside_a_list_argument_raises(
    backend: str, render: Callable[..., str], head: str,
) -> None:
    with pytest.raises(BridgeError, match=head + r"is \[1\.0, nan\]"):
        render([1.0, float("nan")])
    with pytest.raises(BridgeError, match=head):
        render((2.0, np.float64("inf")))


@pytest.mark.parametrize(("backend", "render", "head"), _RENDERERS, ids=_IDS)
def test_finite_values_render_unchanged(
    backend: str, render: Callable[..., str], head: str,
) -> None:
    values = (0.0, -0.0, 1.5, 1e308, -1e-308, 5e-324, np.float64(2.5))
    line = render(*values)
    tokens = ["0.0", "-0.0", "1.5", "1e+308", "-1e-308", "5e-324", "2.5"]
    if backend == "py":
        assert line == (
            "ops.uniaxialMaterial('Elastic', 1, " + ", ".join(tokens) + ")"
        )
    else:
        assert line == "uniaxialMaterial Elastic 1 " + " ".join(tokens)
    # A list of finite floats keeps its fallback rendering.
    assert "[1.0, 2.0]" in render([1.0, 2.0])


def test_py_strategy_rung_with_nan_raises_naming_the_rung() -> None:
    e = PyEmitter()
    spec = StrategySpec(
        name="lad",
        rungs=(
            ("Newton",),
            ("NewtonLineSearch", "-tol", float("nan")),
        ),
    )
    with pytest.raises(
        BridgeError, match=r"analyze strategy 'lad' rung 1: argument 2 is nan",
    ):
        e.analyze(steps=1, strategy=spec)


def _truss(E: float) -> apeSees:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=E)
    ops.element.Truss(pg="Cols", A=0.5, material=mat)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear(factor=1.0)) as p:
        p.load(pg="Top", forces=(0.0, 0.0, 1.0))
    ops.analysis.Static()
    return ops


@pytest.mark.parametrize("backend", ["py", "tcl"])
def test_emitting_a_model_with_nan_raises_and_writes_no_deck(
    backend: str, tmp_path: Path,
) -> None:
    """The card's done-when: a NaN material constant refuses the deck."""
    ops = _truss(E=float("nan"))
    deck = tmp_path / f"model.{backend}"
    with pytest.raises(
        BridgeError, match=r"uniaxialMaterial.*argument 2 is nan",
    ):
        getattr(ops, backend)(str(deck))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("backend", ["py", "tcl"])
def test_the_same_model_with_finite_values_emits(
    backend: str, tmp_path: Path,
) -> None:
    ops = _truss(E=100.0)
    deck = tmp_path / f"model.{backend}"
    getattr(ops, backend)(str(deck))
    assert "uniaxialMaterial" in deck.read_text(encoding="utf-8")
