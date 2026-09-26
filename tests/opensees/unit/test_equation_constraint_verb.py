"""``apeSees.equation_constraint`` — user ``equationConstraint`` rows.

Unit layer: declaration validation, the emitted row, and the handler rules
it shares with an ``enforce="equation"`` tie (ADR 0068 INV-4): with no
declared handler the bridge auto-emits ``Lagrange`` (``Transformation``
would drop the row silently), a declared ``Transformation`` / ``Auto``
raises, a partitioned emit refuses, and ``ops.h5`` warns that the row is
not archived.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.emitter.h5 import H5FeatureDeferredWarning
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _two_bars() -> FEMStub:
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
        node_pgs={"Base": [1, 3]},
    )
    elements = _ElementsStub(elem_pgs={
        "Bars": _ElementGroupView(ids=(1, 2), connectivity=((1, 2), (3, 4))),
    })
    return FEMStub(nodes=nodes, elements=elements)


def _bridge() -> apeSees:
    ops = apeSees(cast("object", _two_bars()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    ops.element.Truss(pg="Bars", A=1e-3, material=steel)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    return ops


def _chain(ops: apeSees, handler: str | None = None) -> None:
    if handler is not None:
        getattr(ops.constraints, handler)()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormUnbalance(tol=1e-8, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()


def _calls(ops: apeSees, name: str) -> list[tuple]:
    em = RecordingEmitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.build().emit(em)
    return [args for (n, args, _kw) in em.calls if n == name]


# ---------------------------------------------------------------- emission

def test_row_is_emitted_in_opensees_order() -> None:
    ops = _bridge()
    ops.equation_constraint(
        constrained=(4, 2), retained=[(2, 2, -0.5), (2, 1, 0.25)], coef=2.0,
    )
    _chain(ops)
    (row,) = _calls(ops, "equationConstraint")
    assert row[:3] == (4, 2, 2.0)
    assert [tuple(r) for r in row[3]] == [(2, 2, -0.5), (2, 1, 0.25)]


def test_tcl_line(tmp_path: Path) -> None:
    ops = _bridge()
    ops.equation_constraint(constrained=(4, 1), retained=[(2, 1, -1.0)])
    _chain(ops, "Lagrange")
    path = tmp_path / "m.tcl"
    ops.tcl(str(path))
    lines = [ln.split() for ln in path.read_text().splitlines()
             if ln.startswith("equationConstraint")]
    assert len(lines) == 1
    tok = lines[0]
    assert (int(tok[1]), int(tok[2]), float(tok[3])) == (4, 1, 1.0)
    assert (int(tok[4]), int(tok[5]), float(tok[6])) == (2, 1, -1.0)


# ----------------------------------------------------------------- handler

def test_no_declared_handler_auto_emits_lagrange_not_transformation() -> None:
    ops = _bridge()
    ops.equation_constraint(constrained=(4, 1), retained=[(2, 1, -1.0)])
    _chain(ops)
    assert [c[0] for c in _calls(ops, "constraints")] == ["Lagrange"]


def test_without_rows_nothing_is_auto_emitted() -> None:
    ops = _bridge()
    _chain(ops)
    assert _calls(ops, "constraints") == []


@pytest.mark.parametrize("handler", ["Transformation", "Auto"])
def test_a_handler_that_drops_eq_rows_is_refused(handler: str) -> None:
    ops = _bridge()
    ops.equation_constraint(constrained=(4, 1), retained=[(2, 1, -1.0)])
    _chain(ops, handler)
    with pytest.raises(ValueError, match="cannot enforce"):
        ops.build().emit(RecordingEmitter())


def test_partitioned_emit_refuses_the_rows() -> None:
    ops = _bridge()
    ops.equation_constraint(constrained=(4, 1), retained=[(2, 1, -1.0)])
    bm = ops.build()
    with pytest.raises(BridgeError, match="partitioned"):
        bm._emit_partitioned(  # the guard runs before any argument is used
            emitter=RecordingEmitter(), tags=None, transforms=[],  # type: ignore[arg-type]
            elements=[], inferred_ndf={}, pre_element=[], post_element=[],
            base_resolver=None,
        )


def test_h5_warns_the_rows_are_not_archived(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    ops = _bridge()
    ops.equation_constraint(constrained=(4, 1), retained=[(2, 1, -1.0)])
    _chain(ops, "Lagrange")
    with pytest.warns(H5FeatureDeferredWarning, match="NOT archived"):
        ops.h5(str(tmp_path / "m.h5"))


# -------------------------------------------------------------- validation

@pytest.mark.parametrize("kwargs, match", [
    ({"constrained": (4, 1), "retained": []}, "at least one"),
    ({"constrained": (4, 1), "retained": [(2, 1, 0.0)]}, "non-zero"),
    ({"constrained": (4, 1), "retained": [(2, 1, float("nan"))]}, "finite"),
    ({"constrained": (4, 1), "retained": [(2, 1, -1.0)], "coef": 0.0},
     "non-zero"),
    ({"constrained": (4, 0), "retained": [(2, 1, -1.0)]}, ">= 1"),
    ({"constrained": (4, 1), "retained": [(4, 1, -1.0)]}, "both sides"),
    ({"constrained": 4, "retained": [(2, 1, -1.0)]}, r"\(node, dof\) pair"),
    ({"constrained": (4, 1), "retained": [(2, 1)]}, "triple"),
])
def test_declaration_is_validated(kwargs: dict, match: str) -> None:
    ops = _bridge()
    with pytest.raises((ValueError, TypeError), match=match):
        ops.equation_constraint(**kwargs)


def test_unknown_node_and_oversized_dof_fail_at_emit() -> None:
    ops = _bridge()
    ops.equation_constraint(constrained=(99, 1), retained=[(2, 1, -1.0)])
    _chain(ops, "Lagrange")
    with pytest.raises(BridgeError, match="node 99"):
        ops.build().emit(RecordingEmitter())

    ops = _bridge()
    ops.equation_constraint(constrained=(4, 4), retained=[(2, 1, -1.0)])
    _chain(ops, "Lagrange")
    with pytest.raises(BridgeError, match="exceeds that node's ndf"):
        ops.build().emit(RecordingEmitter())
