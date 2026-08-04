"""Silent-failures slice 3 — loud defaults for the penalty tie family.

Two warnings ship here:

* the untouched ``ASDEmbeddedNodeElement`` C++-parity default
  ``stiffness=1e18`` warns at emit (unit-blind; in N/mm/MPa it destroys
  the conditioning of the stiffness matrix and Newton stalls, while
  1e10–1e12 converge — measured on a two-block series column with an
  exact closed form);
* the ``Lagrange`` handler combined with an absolute ``NormDispIncr``
  test warns (multiplier DOFs enter the displacement-increment norm
  with force-like scaling, so a tight absolute tolerance can be
  unreachable on an exactly converged solve).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh.opensees._internal.build import _emit_one_interpolation
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter


def _emit(rec: InterpolationRecord) -> list:
    e = RecordingEmitter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _emit_one_interpolation(e, rec, TagAllocator())
    return [w for w in caught if issubclass(w.category, UserWarning)]


def test_default_1e18_penalty_emit_warns() -> None:
    rec = InterpolationRecord(
        kind="tie", slave_node=7, master_nodes=[8, 9, 10],
        weights=np.array([0.3, 0.3, 0.4]),
    )   # stiffness left at the 1e18 dataclass default
    caught = _emit(rec)
    assert any("K=1e18" in str(w.message) for w in caught)


def test_calibrated_stiffness_does_not_warn() -> None:
    rec = InterpolationRecord(
        kind="tie", slave_node=7, master_nodes=[8, 9, 10],
        weights=np.array([0.3, 0.3, 0.4]),
        stiffness=1.0e12,
    )
    caught = _emit(rec)
    assert not any("K=1e18" in str(w.message) for w in caught)


def test_equation_route_does_not_warn() -> None:
    rec = InterpolationRecord(
        kind="tie", slave_node=7, master_nodes=[8, 9, 10],
        weights=np.array([0.3, 0.3, 0.4]),
        enforce="equation", dofs=[1, 2, 3],
    )   # stiffness field still 1e18 but never emitted on this route
    caught = _emit(rec)
    assert not any("K=1e18" in str(w.message) for w in caught)


def test_distributing_route_does_not_warn() -> None:
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3, 4],
        weights=None,
    )   # RBE3 emits the fork element; the 1e18 field is inert
    caught = _emit(rec)
    assert not any("K=1e18" in str(w.message) for w in caught)


# ---------------------------------------------------------------------
# Lagrange + NormDispIncr
# ---------------------------------------------------------------------


def test_lagrange_with_norm_disp_incr_warns() -> None:
    from apeGmsh.opensees.analysis.test import NormDispIncr
    from apeGmsh.opensees.apesees import (
        BuiltModel,
        OpenSeesAutoEmitWarning,
    )

    with pytest.warns(OpenSeesAutoEmitWarning, match="NormDispIncr"):
        BuiltModel._warn_lagrange_norm_disp_incr(
            [NormDispIncr(tol=1e-8, max_iter=30)])


def test_lagrange_with_norm_unbalance_stays_silent() -> None:
    from apeGmsh.opensees.analysis.test import NormUnbalance
    from apeGmsh.opensees.apesees import BuiltModel

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BuiltModel._warn_lagrange_norm_disp_incr(
            [NormUnbalance(tol=1e-3, max_iter=30)])
