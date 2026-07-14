"""Live tests for :meth:`apeSees.modal_response_history` and
:meth:`apeSees.response_spectrum_analysis` (ADR 0075 slice 2).

Oracles follow the fork guide (``LadrunoModalResponse_guide.md``):

* the exact modal transient must match a direct Newmark run of the
  SAME linear model under **mass-proportional (alphaM) Rayleigh only**
  — element-level ``betaK`` does not reproduce the assembled ``a1·K``
  on some elements and differs by a few percent;
* the SRSS-combined design displacement must match a numpy hand-oracle
  built from ``modalProperties`` participation factors and mode
  shapes (``u_a = Γ_a · Sa(T_a)/ω_a² · φ_a``, scale-invariant via the
  ``Γ_a·φ_a`` product).

The commands need a Ladruno build with the ADR-44 modal family (fork
PRs #537+); tests skip on older builds via the ``modalResponseHistory``
attribute probe. The inverse test pins the friendly fork error on
builds WITHOUT the family.
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


def _has_modal_response() -> bool:
    return getattr(openseespy, "modalResponseHistory", None) is not None


requires_modal_family = pytest.mark.skipif(
    not _has_modal_response(),
    reason=(
        "bound openseespy build lacks the Ladruno ADR-44 modal family "
        "(modalResponseHistory) — rebuild the fork from ladruno HEAD"
    ),
)

# Tip-mass cantilever properties (mirrors test_eigen_cantilever.py):
# omega_1 = sqrt(3*E*Iz / (m*L^3)) ~ 775 rad/s -> T1 ~ 8.1 ms.
_E, _IZ, _L, _M_TIP = 200e9, 1e-4, 1.0, 100.0
_DT, _N_STEPS = 2.0e-4, 400
_A0 = 2.0  # alphaM-only Rayleigh


def _cantilever() -> "apeSees":
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=_E, Iz=_IZ, Iy=_IZ, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(_M_TIP, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6))
    return ops


def _pulse_series(ops: "apeSees"):
    """Half-sine-ish pulse, PWL-sampled, extending past the run end."""
    t_total = (_N_STEPS + 5) * _DT
    n = int(round(t_total / _DT))
    t = np.arange(n + 1) * _DT
    t_pulse = 0.02
    values = np.where(
        t <= t_pulse, np.sin(np.pi * t / t_pulse), 0.0,
    )
    return ops.timeSeries.Path(values=tuple(values), dt=_DT)


def _direct_chain(ops: "apeSees") -> None:
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=20)
    ops.algorithm.Linear()
    ops.integrator.Newmark(gamma=0.5, beta=0.25)
    ops.analysis.Transient()


@requires_modal_family
@pytest.mark.live
def test_modal_history_load_channel_matches_direct_newmark() -> None:
    """Nodal-force channel: exact modal transient == Newmark (alphaM)."""
    # Direct Newmark reference — pattern's own series drives P(t).
    ops_a = _cantilever()
    ts_a = _pulse_series(ops_a)
    pat_a = ops_a.pattern.Plain(series=ts_a)
    pat_a.load(node=2, forces=(1.0e5, 0, 0, 0, 0, 0))
    ops_a.damping.rayleigh(alpha_m=_A0, beta_k=0.0)
    _direct_chain(ops_a)
    rc = ops_a.analyze(steps=_N_STEPS, dt=_DT)
    assert rc == 0
    from apeGmsh.opensees.emitter.live import _get_ops
    u_direct = float(_get_ops().nodeDisp(2, 1))

    # Modal superposition — same P via -load/-series; damping via flag.
    ops_b = _cantilever()
    ts_b = _pulse_series(ops_b)
    pat_b = ops_b.pattern.Plain(series=ts_b)
    pat_b.load(node=2, forces=(1.0e5, 0, 0, 0, 0, 0))
    result = ops_b.modal_response_history(
        dt=_DT, n_steps=_N_STEPS, num_modes=6,
        load=pat_b, series=ts_b,
        rayleigh=(_A0, 0.0),
        solver="-fullGenLapack",
    )
    u_modal = result.node_disp(2, 1)

    assert u_direct != 0.0
    assert u_modal == pytest.approx(u_direct, rel=1e-2)


@requires_modal_family
@pytest.mark.live
def test_modal_history_base_accel_matches_uniform_excitation() -> None:
    """Base-accel channel: relative response == UniformExcitation run."""
    ops_a = _cantilever()
    ts_a = _pulse_series(ops_a)
    ops_a.pattern.UniformExcitation(direction=1, series=ts_a)
    ops_a.damping.rayleigh(alpha_m=_A0, beta_k=0.0)
    _direct_chain(ops_a)
    rc = ops_a.analyze(steps=_N_STEPS, dt=_DT)
    assert rc == 0
    from apeGmsh.opensees.emitter.live import _get_ops
    u_direct = float(_get_ops().nodeDisp(2, 1))

    ops_b = _cantilever()
    ts_b = _pulse_series(ops_b)
    result = ops_b.modal_response_history(
        dt=_DT, n_steps=_N_STEPS, num_modes=6,
        base_accel=ts_b, direction=1,
        rayleigh=(_A0, 0.0),
        solver="-fullGenLapack",
    )
    u_modal = result.node_disp(2, 1)

    assert u_direct != 0.0
    assert u_modal == pytest.approx(u_direct, rel=1e-2)


@requires_modal_family
@pytest.mark.live
def test_rsa_srss_matches_numpy_hand_oracle() -> None:
    """SRSS-combined displacement == numpy SRSS over Γ·Sa/ω²·φ."""
    sa0 = 5.0  # flat design spectrum — interpolation-free
    periods = [1e-3, 10.0]
    accels = [sa0, sa0]

    # Hand oracle from modalProperties on an identical model.
    props = _cantilever().modal_properties(2, solver="-fullGenLapack")
    gammas = props.participation_factors("MX")
    omegas = props.omega
    contributions = []
    for mode in (1, 2):
        phi = float(props.mode_shape(node=2, mode=mode)[0])
        gamma = float(gammas[mode - 1])
        w2 = float(omegas[mode - 1]) ** 2
        contributions.append(gamma * sa0 / w2 * phi)
    u_expected = float(np.sqrt(np.sum(np.square(contributions))))

    result = _cantilever().response_spectrum_analysis(
        1, periods=periods, accels=accels,
        combine="SRSS", num_modes=2,
        solver="-fullGenLapack",
    )
    u_combined = result.node_disp(2, 1)

    assert u_expected != 0.0
    assert u_combined == pytest.approx(u_expected, rel=1e-3)


@pytest.mark.skipif(
    _has_modal_response(),
    reason="build HAS the ADR-44 family — the fork-required error "
           "path is unreachable",
)
@pytest.mark.live
def test_modal_history_raises_friendly_error_on_pre_adr44_build() -> None:
    """Pre-ADR-44 builds get the loud fork-required error, not an
    AttributeError (and RSA -combine is gated the same way — the
    upstream parser would silently skip the combination)."""
    ops = _cantilever()
    ts = _pulse_series(ops)
    with pytest.raises(RuntimeError, match="Ladruno fork build"):
        ops.modal_response_history(
            dt=_DT, n_steps=10, num_modes=2,
            base_accel=ts, direction=1, damp=0.05,
        )
    with pytest.raises(RuntimeError, match="Ladruno fork build"):
        _cantilever().response_spectrum_analysis(
            1, periods=[0.1, 1.0], accels=[1.0, 1.0],
            combine="SRSS", num_modes=2,
        )
