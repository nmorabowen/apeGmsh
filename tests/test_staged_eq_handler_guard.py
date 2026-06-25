"""ADR 0068 Open item 5 — staged-path EQ handler guard.

The global EQ-aware handler auto-emit does not run for staged models (each
stage declares its own analysis chain). Without a guard, an
``enforce="equation"`` tie in a staged model whose stage uses
Transformation / Auto / Plain (or no handler) is silently dropped. The
guard (`apeSees._validate_staged_eq_handlers`) fails loud per stage.

Tested at the method level on a lightweight bridge stub — building a full
staged ``apeSees`` is unnecessary; the method only reads ``self.fem`` and
``self.stage_records``.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh._kernel.records._kinds import ConstraintKind as K
from apeGmsh.opensees.apesees import BuiltModel
from apeGmsh.opensees.analysis.constraint_handler import (
    Auto, Lagrange, LadrunoProjection, Penalty, Plain, Transformation,
)


def _rec(enforce):
    return InterpolationRecord(
        kind=K.TIE, slave_node=1, master_nodes=[2, 3, 4],
        weights=np.array([0.3, 0.3, 0.4]), enforce=enforce,
    )


class _SC:
    def __init__(self, recs):
        self._r = list(recs)

    def interpolations(self):
        return iter(self._r)


class _Fem:
    def __init__(self, interp_recs):
        self.elements = type("E", (), {"constraints": _SC(interp_recs)})()


class _Stage:
    def __init__(self, name, constraints, stage_constraint_records=()):
        self.name = name
        self.constraints = constraints
        self.stage_constraint_records = tuple(stage_constraint_records)


class _Bridge:
    def __init__(self, fem, stages):
        self.fem = fem
        self.stage_records = stages


def _validate(fem, stages):
    BuiltModel._validate_staged_eq_handlers(_Bridge(fem, stages))


# --------------------------------------------------------------------------
# global equation tie present
# --------------------------------------------------------------------------

@pytest.mark.parametrize("handler", [
    Transformation(), Auto(), Plain(), None,
])
def test_global_eq_with_incapable_stage_handler_raises(handler):
    fem = _Fem([_rec("equation")])
    with pytest.raises(ValueError, match="enforce='equation'"):
        _validate(fem, [_Stage("excavate", handler)])


@pytest.mark.parametrize("handler", [
    Lagrange(), Penalty(alpha_sp=1e12, alpha_mp=1e12), LadrunoProjection(),
])
def test_global_eq_with_capable_stage_handler_ok(handler):
    fem = _Fem([_rec("equation")])
    _validate(fem, [_Stage("excavate", handler)])   # no raise


def test_one_bad_stage_among_good_raises_naming_it():
    fem = _Fem([_rec("equation")])
    stages = [
        _Stage("s0", Lagrange()),
        _Stage("s1", Transformation()),   # the offender
    ]
    with pytest.raises(ValueError, match="'s1'"):
        _validate(fem, stages)


# --------------------------------------------------------------------------
# no equation tie → guard is a no-op (MP-only staged models unaffected)
# --------------------------------------------------------------------------

def test_no_eq_tie_allows_transformation():
    fem = _Fem([_rec("penalty")])
    _validate(fem, [_Stage("s0", Transformation()), _Stage("s1", None)])


# --------------------------------------------------------------------------
# stage-bound equation tie (no global tie) still guarded for that stage
# --------------------------------------------------------------------------

def test_stage_bound_eq_tie_guards_its_own_stage():
    fem = _Fem([])  # no global ties
    stages = [
        _Stage("s0", Transformation()),  # no eq here → fine
        _Stage("s1", Transformation(),
               stage_constraint_records=[_rec("equation")]),  # eq here → bad
    ]
    with pytest.raises(ValueError, match="'s1'"):
        _validate(fem, stages)
