"""Tier-0 serial-gather stopgap (ADR 0077).

A partition-authored model has no *distributed* modal path yet (Tier 1 /
FEAST is gated on the classic-Tcl ``-feast`` unlock). But the existing
single-process drivers already do the correct thing on such a model: the
live emitter has ``supports_partitions = False``
(``emitter/live.py:313``), so ``apeSees.eigen`` / ``apeSees.modal_properties``
build the **full, gathered** model in one process and run stock serial
ARPACK on it. That is exact — it does not scale the eigensolve (the whole
model is assembled on one rank), but the modes it returns are the true
global modes, unlike a bare ``eigen`` under ``OpenSeesMP`` (which solves
each rank's LOCAL subdomain — the refuted ADR 0077 v1, and the reason
``ops.damping.modal`` fails loud under partitioned emit).

This pins that contract: partitioned-serial eigen == unpartitioned eigen,
bit-for-bit (same flat build), and ``modal_properties`` likewise works
(with correct participation, unavailable in any distributed path today).

Gated by the ``live`` marker — requires openseespy.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees

openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    make_two_column_frame,
    make_two_column_frame_partitioned,
)

# The flat live emit ignores partitions, so the ADR 0027 auto-emit
# warnings (numberer / system / MP constraints) do not fire here; but keep
# the model construction quiet if a future change routes differently.
pytestmark = pytest.mark.filterwarnings(
    "ignore:len.fem.partitions.:UserWarning"
)


def _build_frame(ops: apeSees) -> None:
    """Two fixed-base columns with equal tip masses (ndf 6)."""
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(100.0, 100.0, 1e-6, 1e-6, 1e-6, 1e-6))


def _eigenvalues(fem: FEMStub, n: int) -> np.ndarray:
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _build_frame(ops)
    return ops.eigen(num_modes=n).eigenvalues


@pytest.mark.live
def test_partitioned_serial_eigen_equals_unpartitioned() -> None:
    """Serial eigen on a partition-authored model == the flat model's."""
    flat = _eigenvalues(make_two_column_frame(), 4)
    part = _eigenvalues(make_two_column_frame_partitioned(), 4)

    assert flat.shape == (4,)
    np.testing.assert_allclose(part, flat, rtol=1e-12, atol=0.0)


@pytest.mark.live
def test_partitioned_modal_properties_available_and_consistent() -> None:
    """modal_properties runs on a partitioned model and its eigenvalues
    match the plain eigen solve (participation factors are available —
    the Tier-0 advantage over any distributed path today)."""
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _build_frame(ops)

    mp = ops.modal_properties(num_modes=4)
    assert mp.eigenvalues.shape == (4,)
    # Participation-factor accessors resolve (serial DomainModalProperties).
    ratios = mp.mass_ratios
    assert ratios is not None

    np.testing.assert_allclose(
        mp.eigenvalues, _eigenvalues(make_two_column_frame(), 4),
        rtol=1e-12, atol=0.0,
    )
