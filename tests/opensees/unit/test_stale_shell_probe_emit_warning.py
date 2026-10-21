"""#2001: the emit warns when a recorder reads stale upstream-shell state.

After a step solved with ``algorithm Linear`` the shells in
``ZERO_GAUSS_PROBE_CLASSES`` answer every section probe from state that
never saw the converged displacement; ``.out``, MPCO and ``.ladruno``
record it with no error (measured in
``tests/test_results_shell_resultant_vocabulary.py``). The bridge emit is
the one place that sees both the algorithm and the recorders, so it
warns there, once per (recorder record, class), and names Newton.

The bridge emits two of the five listed classes (ShellMITC4, ShellDKGQ);
the validator is also driven directly for the NL classes, whose nodal
forces are stale too.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.pg_expand import ElementPlanRows
from apeGmsh.opensees._internal.build.validators.shell_probe import (
    StaleShellProbeWarning,
    warn_stale_shell_probe_under_linear,
)
from apeGmsh.opensees.analysis import algorithm as algo_mod
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.recorder import Element as ElementRecorder

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

_LISTED_BRIDGE_CLASSES = ("ShellMITC4", "ShellDKGQ")


def _two_quads() -> FEMStub:
    """``Wall`` (element 1) and ``Slab`` (element 2), sharing no node."""
    nodes = _NodesStub(
        ids=list(range(1, 9)),
        coords=[
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
            (5.0, 0.0, 0.0), (6.0, 0.0, 0.0), (6.0, 1.0, 0.0), (5.0, 1.0, 0.0),
        ],
        node_pgs={"All": list(range(1, 9))},
    )
    elements = _ElementsStub(elem_pgs={
        "Wall": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),)),
        "Slab": _ElementGroupView(ids=(2,), connectivity=((5, 6, 7, 8),)),
    })
    return FEMStub(nodes=nodes, elements=elements)


def _bridge(wall_class: str) -> apeSees:
    ops = apeSees(cast("object", _two_quads()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=2.0e10, nu=0.2, h=0.2)
    getattr(ops.element, wall_class)(pg="Wall", section=sec)
    ops.element.ASDShellQ4(pg="Slab", section=sec)
    return ops


def _chain(ops: apeSees, algorithm: str) -> None:
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    getattr(ops.algorithm, algorithm)()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()


#: One recorder per route, each reading the sections of ``Wall``.
_ROUTES = {
    ".out": lambda ops: ops.recorder.Element(
        file="w.out", response=("stresses",), pg="Wall"),
    ".out section": lambda ops: ops.recorder.Element(
        file="w.out", response=("section", "1", "force"), pg="Wall"),
    "declare": lambda ops: ops.recorder.declare(
        gauss="shell_resultant", pg="Wall", name="wall"),
    "MPCO": lambda ops: ops.recorder.MPCO(
        file="w.mpco", elem_responses=("section.force",),
        elements_pg="Wall", nodes_pg="All"),
    ".ladruno": lambda ops: ops.recorder.Ladruno(
        file="w.ladruno", elem_responses=("section.force",),
        elements_pg="Wall", nodes_pg="All"),
}


def _emit(ops: apeSees) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ops.build().emit(TclEmitter())
    return [w for w in caught if issubclass(w.category, StaleShellProbeWarning)]


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("wall_class", _LISTED_BRIDGE_CLASSES)
def test_linear_warns_once_naming_class_and_newton(
    route: str, wall_class: str,
) -> None:
    ops = _bridge(wall_class)
    _ROUTES[route](ops)
    _chain(ops, "Linear")
    got = _emit(ops)
    assert len(got) == 1, [str(w.message) for w in got]
    msg = str(got[0].message)
    assert f"1 {wall_class} element(s)" in msg
    assert "algorithm Linear" in msg and "algorithm Newton" in msg
    assert "stale" in msg


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("wall_class", _LISTED_BRIDGE_CLASSES)
def test_newton_is_silent(route: str, wall_class: str) -> None:
    ops = _bridge(wall_class)
    _ROUTES[route](ops)
    _chain(ops, "Newton")
    with warnings.catch_warnings():
        warnings.simplefilter("error", StaleShellProbeWarning)
        ops.build().emit(TclEmitter())


def test_recorder_on_an_unlisted_shell_is_silent() -> None:
    ops = _bridge("ShellMITC4")
    ops.recorder.Element(file="s.out", response=("stresses",), pg="Slab")
    ops.recorder.MPCO(file="s.mpco", elem_responses=("section.force",),
                      elements_pg="Slab", nodes_pg="All")
    _chain(ops, "Linear")
    assert _emit(ops) == []


def test_whole_model_mpco_warns_for_the_listed_class_only() -> None:
    ops = _bridge("ShellMITC4")
    ops.recorder.MPCO(file="all.mpco", elem_responses=("section.force",))
    _chain(ops, "Linear")
    got = _emit(ops)
    assert len(got) == 1
    assert "1 ShellMITC4 element(s)" in str(got[0].message)


def test_explicit_ops_tags_are_matched() -> None:
    """``elements=`` holds OpenSees tags: matched against the planned
    tags, so the Slab's tag stays silent and the Wall's warns."""
    ops = _bridge("ShellDKGQ")
    _chain(ops, "Linear")
    probe = TclEmitter()
    ops.build().emit(probe)
    tags = {
        line.split()[1]: int(line.split()[2]) for line in probe.lines()
        if line.startswith("element ")
    }
    ops.recorder.Element(
        file="s.out", response=("stresses",), elements=(tags["ASDShellQ4"],))
    assert _emit(ops) == []
    ops.recorder.Element(
        file="w.out", response=("stresses",), elements=(tags["ShellDKGQ"],))
    assert len(_emit(ops)) == 1


def test_measured_correct_forces_do_not_warn_on_linear_kinematics_shells(
) -> None:
    """ShellMITC4 / ShellDKGQ form nodal forces from the current
    displacement: measured correct under Linear, so not a warning."""
    ops = _bridge("ShellMITC4")
    ops.recorder.Element(file="f.out", response=("globalForce",), pg="Wall")
    ops.recorder.declare(
        elements=("nodal_resisting_force_x", "nodal_resisting_force_y",
                  "nodal_resisting_force_z"),
        pg="Wall", name="nf")
    _chain(ops, "Linear")
    assert _emit(ops) == []


def test_no_analysis_chain_is_silent() -> None:
    """A model-only export never solves."""
    ops = _bridge("ShellMITC4")
    ops.recorder.Element(file="w.out", response=("stresses",), pg="Wall")
    ops.algorithm.Linear()
    assert _emit(ops) == []


def _staged(
    first: str, second: str, claim_in: int, *, rung: str | None = None,
) -> apeSees:
    """Two stages; with ``rung``, the second stage runs under a ladder
    whose one rescue rung is ``ops.algorithm.<rung>()``."""
    ops = _bridge("ShellMITC4")
    rec = ops.recorder.Element(file="w.out", response=("stresses",), pg="Wall")
    for i, algo in enumerate((first, second)):
        with ops.stage(name=f"s{i}") as s:
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
                algorithm=getattr(ops.algorithm, algo)(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.Plain(),
                system=ops.system.FullGeneral(),
                analysis=ops.analysis.Static(),
            )
            if i == claim_in:
                s.recorder(rec)
            ladder = (
                ops.strategy.Ladder(rungs=[getattr(algo_mod, rung)()])
                if rung is not None and i == 1 else None
            )
            s.run(n_increments=1, strategy=ladder)
    return ops


def test_staged_recorder_sees_a_later_linear_stage() -> None:
    got = _emit(_staged("Newton", "Linear", claim_in=0))
    assert len(got) == 1
    assert "stage(s) 's1'" in str(got[0].message)


def test_staged_linear_ladder_rung_warns() -> None:
    """A Linear rescue rung solves the steps it rescues: it counts."""
    got = _emit(_staged("Newton", "Newton", claim_in=0, rung="Linear"))
    assert len(got) == 1
    assert "stage(s) 's1'" in str(got[0].message)


def test_staged_iterating_ladder_rung_is_silent() -> None:
    assert _emit(_staged("Newton", "Newton", claim_in=0, rung="KrylovNewton")) == []


def test_staged_recorder_claimed_after_the_linear_stage_is_silent() -> None:
    assert _emit(_staged("Linear", "Newton", claim_in=1)) == []


# ---------------------------------------------------------------------------
# The validator directly: the classes the bridge cannot emit
# ---------------------------------------------------------------------------

def _plan(class_name: str) -> tuple:
    spec = type(class_name, (), {})()
    rows = ElementPlanRows(
        np.array([7], dtype=np.int64), np.array([[1, 2, 3, 4]]), 70)
    return ((spec, rows),)


def _linear() -> object:
    from apeGmsh.opensees.analysis.algorithm import Linear
    return Linear()


@pytest.mark.parametrize(
    ("class_name", "response", "warns"),
    [
        ("ShellNLDKGQ", ("globalForce",), True),
        ("ShellNLDKGT", ("forces",), True),
        ("ShellDKGT", ("globalForce",), False),
        ("ShellDKGT", ("strains",), True),
        ("ShellNLDKGT", ("someUnmeasuredToken",), True),
    ],
)
def test_validator_token_rule_per_class(
    class_name: str, response: tuple[str, ...], warns: bool,
) -> None:
    rec = ElementRecorder(file="x.out", response=response, elements=(70,))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_stale_shell_probe_under_linear(
            cast("object", None), _plan(class_name), [rec],  # type: ignore[arg-type]
            staged=False, flat_algorithms=[_linear()], stage_chain=[],
        )
    hits = [w for w in caught if issubclass(w.category, StaleShellProbeWarning)]
    assert len(hits) == (1 if warns else 0)


def test_validator_refuses_an_unclassified_recorder() -> None:
    from apeGmsh.opensees._internal.types import Recorder

    class _NewRecorder(Recorder):
        def _emit(self, emitter, tag):  # pragma: no cover - never emitted
            raise AssertionError

        def dependencies(self):  # pragma: no cover
            return ()

    with pytest.raises(BridgeError, match="not classified"):
        warn_stale_shell_probe_under_linear(
            cast("object", SimpleNamespace()), _plan("ShellMITC4"),  # type: ignore[arg-type]
            [_NewRecorder()], staged=False,
            flat_algorithms=[_linear()], stage_chain=[],
        )
