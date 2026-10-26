"""Pure-numpy tests of the spectrum module (ADR 0124 D5–D7): the design
spectrum, the extractor contract helpers, and the combination kernels.
The driver itself is exercised in
``tests/opensees/live/test_spectrum_after_gravity_live.py``."""
from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.analysis.spectrum import (
    DesignSpectrum,
    ElementResponse,
    NodeDisp,
    NodeReaction,
    combine_directions,
    combine_modes,
    cqc_correlation,
    seismic_envelope,
)

from tests.opensees.fixtures.fem_stub import make_two_node_beam


def test_spectrum_interpolates_clamps_and_scales_like_the_fork() -> None:
    s = DesignSpectrum([0.0, 0.5, 1.0], [2.0, 4.0, 1.0], direction=1, scale=3.0)
    np.testing.assert_allclose(
        s.sa([0.0, 0.25, 0.75, 1.0, 5.0]),
        3.0 * np.array([2.0, 3.0, 2.5, 1.0, 1.0]))
    assert float(DesignSpectrum([0.4], [7.0], direction=2).sa(9.0)) == 7.0


@pytest.mark.parametrize(("kw", "match"), [
    (dict(periods=[], accels=[]), "non-empty"),
    (dict(periods=[0.0, 1.0], accels=[1.0]), "equal-length"),
    (dict(periods=[0.5, 0.5], accels=[1.0, 1.0]), "strictly"),
    (dict(periods=[-0.1, 1.0], accels=[1.0, 1.0]), "non-negative"),
    (dict(periods=[0.0, 1.0], accels=[1.0, -1.0]), "accels"),
    (dict(periods=[0.0], accels=[1.0], direction=0), "1-based"),
    (dict(periods=[0.0], accels=[1.0], scale=0.0), "scale"),
])
def test_spectrum_refuses_bad_input(kw: dict[str, Any], match: str) -> None:
    kw.setdefault("direction", 1)
    with pytest.raises(ValueError, match=match):
        DesignSpectrum(**kw)


def test_built_in_extractors_satisfy_the_protocol_and_label() -> None:
    nd = NodeDisp(nodes=[3, 4], dofs=(1, 2))
    nr = NodeReaction(nodes=[1], dofs=(1,))
    er = ElementResponse(elements=[7], response="localForce")
    for ex in (nd, nr, er):
        assert callable(ex)
    assert nd.labels == ("node3.u1", "node3.u2", "node4.u1", "node4.u2")
    assert nr.labels == ("node1.R1",)
    assert er.response == ("localForce",)
    with pytest.raises(ValueError, match="at least one"):
        NodeDisp(nodes=[], dofs=(1,))


class _FakeOps:
    def __init__(self) -> None:
        self.reactions_called = 0

    def nodeDisp(self, n: int, d: int) -> float:
        return 10.0 * n + d

    def nodeResponse(self, n: int, d: int, kind: int) -> float:
        assert kind == 4          # IncrDisp: trial - committed
        return 0.5

    def reactions(self) -> None:
        self.reactions_called += 1

    def nodeReaction(self, n: int, d: int) -> float:
        return -(10.0 * n + d)

    def eleResponse(self, e: int, *args: str) -> list[float]:
        return [] if e == 99 else [float(e), float(e) + 0.5]


def test_built_in_extractors_read_the_module() -> None:
    ops = _FakeOps()
    np.testing.assert_array_equal(
        NodeDisp(nodes=[1, 2], dofs=(1, 3))(ops), [11.5, 13.5, 21.5, 23.5])
    np.testing.assert_array_equal(
        NodeReaction(nodes=[2], dofs=(2,))(ops), [-22.0])
    assert ops.reactions_called == 1
    np.testing.assert_array_equal(
        ElementResponse(elements=[1, 2], response="force")(ops),
        [1.0, 1.5, 2.0, 2.5])
    with pytest.raises(ValueError, match="returned nothing"):
        ElementResponse(elements=[99], response="force")(ops)


def test_cqc_correlation_matches_der_kiureghian() -> None:
    w = np.array([2.0, 2.2, 9.0])
    xi = np.array([0.05, 0.02, 0.05])
    rho = cqc_correlation(w, xi)
    np.testing.assert_allclose(np.diag(rho), 1.0)
    np.testing.assert_allclose(rho, rho.T, rtol=1e-12)
    r = w[1] / w[0]
    num = 8 * math.sqrt(xi[0] * xi[1]) * (xi[0] + r * xi[1]) * r ** 1.5
    den = ((1 - r * r) ** 2 + 4 * xi[0] * xi[1] * r * (1 + r * r)
           + 4 * (xi[0] ** 2 + xi[1] ** 2) * r * r)
    assert rho[0, 1] == pytest.approx(num / den, rel=1e-14)
    # Equal damping, r -> 1: rho -> 1; far apart: small.
    assert cqc_correlation([1.0, 1.0 + 1e-9], 0.05)[0, 1] == pytest.approx(1.0)
    assert rho[0, 2] < 0.01
    with pytest.raises(ValueError, match="damping"):
        cqc_correlation(w, -0.01)
    with pytest.raises(ValueError, match="omega"):
        cqc_correlation([0.0, 1.0], 0.05)


def test_combine_modes_rules() -> None:
    r = np.array([[3.0, -1.0], [4.0, 1.0]])
    np.testing.assert_allclose(combine_modes(r, rule="SRSS"), [5.0, math.sqrt(2)])
    # Zero damping: CQC collapses to SRSS.
    np.testing.assert_allclose(
        combine_modes(r, rule="CQC", omega=[1.0, 1.3], damping=0.0),
        combine_modes(r, rule="SRSS"))
    # Coincident frequencies: CQC is the algebraic sum.
    np.testing.assert_allclose(
        combine_modes(r, rule="CQC", omega=[2.0, 2.0], damping=0.05),
        [7.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError, match="needs omega"):
        combine_modes(r, rule="CQC")
    with pytest.raises(ValueError, match="rule"):
        combine_modes(r, rule="ABS")


def test_combine_directions_and_envelope() -> None:
    e = np.array([[10.0, 1.0], [2.0, 8.0]])
    np.testing.assert_allclose(
        combine_directions(e, method="100-30"), [10.6, 8.3])
    np.testing.assert_allclose(
        combine_directions(e, method="SRSS"), np.hypot(e[0], e[1]))
    three = np.array([[1.0], [2.0], [3.0]])
    assert float(combine_directions(three)[0]) == pytest.approx(3.0 + 0.9)
    up, lo = seismic_envelope([5.0, -5.0], [2.0, 2.0], factor=1.5)
    np.testing.assert_allclose(up, [8.0, -2.0])
    np.testing.assert_allclose(lo, [2.0, -8.0])
    with pytest.raises(ValueError, match="shape"):
        seismic_envelope([1.0], [1.0, 2.0])


def _bridge() -> apeSees:
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()
    return ops


def test_driver_refuses_before_the_build() -> None:
    spec = DesignSpectrum([0.0, 1.0], [1.0, 1.0], direction=1)
    ex = {"d": NodeDisp(nodes=[2], dofs=(1,))}
    with pytest.raises(ValueError, match="DesignSpectrum"):
        _bridge().spectrum_after_gravity(1, spectra=[], extractors=ex)
    with pytest.raises(ValueError, match="extractor"):
        _bridge().spectrum_after_gravity(1, spectra=[spec], extractors={})
    rot = DesignSpectrum([0.0, 1.0], [1.0, 1.0], direction=4)
    with pytest.raises(BridgeError, match="translational"):
        _bridge().spectrum_after_gravity(1, spectra=[rot], extractors=ex)
    with pytest.raises(ValueError, match="num_modes"):
        _bridge().spectrum_after_gravity(0, spectra=[spec], extractors=ex)


class _StubCodeSpectrum:
    """A codeSpectra-like object: vectorised ``at(T)`` in g, nothing else."""

    def at(self, periods: Any) -> np.ndarray:
        t = np.asarray(periods, dtype=np.float64)
        return np.where(t < 0.5, 1.0, 0.5 / np.maximum(t, 1e-12))


def test_spectrum_takes_an_object_with_at() -> None:
    s = DesignSpectrum(_StubCodeSpectrum(), direction=1, scale=9.81)
    np.testing.assert_allclose(
        s.sa([0.1, 0.5, 2.0]), 9.81 * np.array([1.0, 1.0, 0.25]))
    assert s.periods == ()


def test_spectrum_takes_a_callable_sa() -> None:
    def code_sa(t: np.ndarray) -> np.ndarray:   # e.g. a codeSpectra .at
        return np.where(t < 0.5, 1.0, 0.5 / np.maximum(t, 1e-12))

    s = DesignSpectrum(code_sa, direction=2, scale=9.81)
    np.testing.assert_allclose(
        s.sa([0.1, 0.5, 2.0]), 9.81 * np.array([1.0, 1.0, 0.25]))
    assert s.function is code_sa and s.periods == ()
    with pytest.raises(ValueError, match="not both"):
        DesignSpectrum(code_sa, [1.0], direction=1)
    with pytest.raises(ValueError, match="vectorised"):
        DesignSpectrum(lambda t: 1.0, direction=1).sa([0.1, 0.2])


def test_driver_refuses_declared_recorders() -> None:
    """Every probe leg and return solve commits; a recorder would log
    rows per mode that are not states of the structure."""
    ops = _bridge()
    ops.recorder.Node(file="roof.out", response="disp", nodes=(2,), dofs=(1,))
    spec = DesignSpectrum([0.0, 1.0], [1.0, 1.0], direction=1)
    with pytest.raises(BridgeError, match="declares recorders"):
        ops.spectrum_after_gravity(
            1, spectra=[spec], extractors={"d": NodeDisp(nodes=[2], dofs=(1,))})
