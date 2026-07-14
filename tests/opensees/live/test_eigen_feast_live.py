"""Live tests for :meth:`apeSees.eigen_feast` (ADR 0075 slice 4).

FEAST rides the stock ``ops.eigen`` symbol, so there is no attribute
to probe; the ``modalResponseHistory`` attribute is used as the build
proxy — fork ADR-43 (FEAST) shipped before ADR-44 on the fork
timeline, so any ADR-44 build carries FEAST.

Oracle: the band-targeted solve must return exactly the
``-fullGenLapack`` eigenvalues filtered to the requested band.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees

# Module-level gate: skip every test if openseespy is not installed.
openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_node_beam,
)


requires_feast = pytest.mark.skipif(
    getattr(openseespy, "modalResponseHistory", None) is None,
    reason=(
        "bound openseespy build predates the Ladruno modal family "
        "(ADR-43 FEAST ships before ADR-44) — rebuild the fork"
    ),
)


def _cantilever() -> "apeSees":
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    # Comparable masses on every DOF so the model has a spread spectrum.
    ops.mass(pg="Top", values=(100.0, 100.0, 100.0, 1.0, 1.0, 1.0))
    return ops


@requires_feast
@pytest.mark.live
def test_eigen_feast_band_matches_filtered_full_solve() -> None:
    reference = _cantilever().eigen(num_modes=6, solver="-fullGenLapack")
    freqs = np.sort(reference.freq)
    # Band covering the lowest two modes only, with margin.
    f_lo = 0.5 * float(freqs[0])
    f_hi = float(np.sqrt(freqs[1] * freqs[2]))  # geometric midpoint

    result = _cantilever().eigen_feast(f_lo, f_hi, certify=True)
    in_band = freqs[(freqs >= f_lo) & (freqs <= f_hi)]

    assert len(result.eigenvalues) == len(in_band)
    np.testing.assert_allclose(
        np.sort(result.freq), in_band, rtol=1e-8,
    )


@requires_feast
@pytest.mark.live
def test_eigen_feast_mode_shape_accessible() -> None:
    reference = _cantilever().eigen(num_modes=2, solver="-fullGenLapack")
    f1 = float(reference.freq[0])
    result = _cantilever().eigen_feast(0.5 * f1, 1.5 * f1)
    assert len(result.eigenvalues) >= 1
    shape = result.mode_shape(node=2, mode=1)
    assert shape.shape == (6,)
    assert float(np.abs(shape).max()) > 0.0
