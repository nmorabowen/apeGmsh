"""R3-S1b (#1822): ``InertStaticDtWarning`` covers every ``Static`` integrator.

Oracle (warn-iff, the maintainer's D1 re-ruling): under ``Static``
OpenSees ignores ``analyze``'s ``dt``, so a given ``dt`` warns at stage
close, except when it equals the integrator's declared step field
(``static_step_field``: ``dlam`` for ``LoadControl`` /
``LadrunoLoadControl``).  An integrator with no step field warns on any
``dt``; one that does not declare the ClassVar warns too (fail loud).
Omitting ``dt`` and every ``Transient`` stage are silent.

Silent cases turn the warning into an error.  Run the file with
``-W error::apeGmsh.opensees.InertStaticDtWarning``.
"""
from __future__ import annotations

import dataclasses
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import pytest

from apeGmsh.opensees import InertStaticDtWarning
from apeGmsh.opensees._internal.types import Integrator, Primitive
from apeGmsh.opensees.analysis import integrator as integrator_mod
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.unit.test_stages import _make_single_quad_fem


@contextmanager
def _silent() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", InertStaticDtWarning)
        yield


def _ops() -> apeSees:
    ops = apeSees(_make_single_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 1))
    return ops


def _stage(
    ops: apeSees,
    integ: Callable[[apeSees], Any],
    *,
    transient: bool = False,
    **run: Any,
) -> None:
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(pg="Bottom", forces=(1.0, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
            algorithm=ops.algorithm.Newton(),
            integrator=integ(ops),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.Plain(),
            system=ops.system.BandGeneral(),
            analysis=(ops.analysis.Transient() if transient
                      else ops.analysis.Static()),
        )
        s.run(n_increments=4, **run)


#: Static integrators with no step field: any given dt warns.
NO_STEP_FIELD: dict[str, Callable[[apeSees], Any]] = {
    "DisplacementControl": lambda o: o.integrator.DisplacementControl(
        node=2, dof=1, dU=0.01),
    "ArcLength": lambda o: o.integrator.ArcLength(s=0.1, alpha=1.0),
    "LadrunoArcLength": lambda o: o.integrator.LadrunoArcLength(
        s=0.1, alpha=1.0),
    "LadrunoDynamicRelaxation": lambda o: (
        o.integrator.LadrunoDynamicRelaxation()),
    "LadrunoIndirectControl": lambda o: o.integrator.LadrunoIndirectControl(
        incr=0.01, controls=[(2, 1, 1.0)]),
}

#: Static integrators whose step is ``dlam``: dt warns iff dt != dlam.
DLAM_STEP: dict[str, Callable[[apeSees], Any]] = {
    "LoadControl": lambda o: o.integrator.LoadControl(dlam=0.25),
    "LadrunoLoadControl": lambda o: o.integrator.LadrunoLoadControl(
        dlam=0.25),
}


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def _family() -> list[type]:
    return [getattr(integrator_mod, n) for n in integrator_mod.__all__]


def test_every_declared_step_field_is_a_real_field() -> None:
    for cls in _family():
        field = getattr(cls, "static_step_field", None)
        if field is not None:
            names = {f.name for f in dataclasses.fields(cls)}
            assert field in names, (cls.__name__, field)


def test_static_family_declares_its_step_field() -> None:
    declared = {
        cls.__name__: cls.static_step_field
        for cls in _family() if "static_step_field" in vars(cls)
    }
    assert declared == {
        **{name: None for name in NO_STEP_FIELD},
        **{name: "dlam" for name in DLAM_STEP},
    }


# ---------------------------------------------------------------------------
# Warn: every static integrator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NO_STEP_FIELD))
def test_no_step_field_any_dt_warns(name: str) -> None:
    ops = _ops()
    with pytest.warns(InertStaticDtWarning) as rec:
        _stage(ops, NO_STEP_FIELD[name], dt=0.25)
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert msg.startswith(
        f"dt has no effect under Static; {name} sets the step")
    assert "no dlam for dt to match" in msg
    assert "'push'" in msg and "dt=0.25" in msg
    # A warning, not a refusal: the stage is recorded with its dt.
    assert ops._stage_records[0].dt == 0.25


@pytest.mark.parametrize("name", sorted(DLAM_STEP))
def test_dlam_step_differing_dt_warns(name: str) -> None:
    with pytest.warns(InertStaticDtWarning) as rec:
        _stage(_ops(), DLAM_STEP[name], dt=1.0)
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert msg.startswith("dt has no effect under Static; dlam sets the step")
    assert "dt=1.0" in msg and "dlam=0.25" in msg


def test_transient_integrator_under_static_warns() -> None:
    """Newmark declares no step field; under Static its dt is inert too."""
    with pytest.warns(InertStaticDtWarning, match="Newmark sets the step"):
        _stage(_ops(), lambda o: o.integrator.Newmark(gamma=0.5, beta=0.25),
               dt=0.01)


@dataclass(frozen=True, kw_only=True, slots=True)
class _UndeclaredStatic(Integrator):
    """A static integrator that forgot ``static_step_field``."""

    dlam: float

    def _emit(self, emitter: Any, tag: int) -> None:
        _ = tag
        emitter.integrator("LoadControl", self.dlam)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


def test_undeclared_integrator_warns_even_when_dt_matches_its_dlam() -> None:
    """Fail loud: no declaration means no step field, so any dt warns."""
    assert not hasattr(_UndeclaredStatic, "static_step_field")
    with pytest.warns(InertStaticDtWarning, match="_UndeclaredStatic"):
        _stage(_ops(), lambda o: _UndeclaredStatic(dlam=0.25), dt=0.25)


# ---------------------------------------------------------------------------
# Silent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted({**NO_STEP_FIELD, **DLAM_STEP}))
def test_static_dt_omitted_is_silent(name: str) -> None:
    with _silent():
        _stage(_ops(), {**NO_STEP_FIELD, **DLAM_STEP}[name])


@pytest.mark.parametrize("name", sorted(DLAM_STEP))
def test_dlam_step_dt_equal_to_dlam_is_silent(name: str) -> None:
    with _silent():
        _stage(_ops(), DLAM_STEP[name], dt=0.25)


@pytest.mark.parametrize("name", sorted({**NO_STEP_FIELD, **DLAM_STEP}))
def test_transient_is_silent_for_any_integrator(name: str) -> None:
    with _silent():
        _stage(_ops(), {**NO_STEP_FIELD, **DLAM_STEP}[name],
               transient=True, dt=0.7)


def test_transient_newmark_is_silent() -> None:
    with _silent():
        _stage(_ops(), lambda o: o.integrator.Newmark(gamma=0.5, beta=0.25),
               transient=True, dt=0.01)
