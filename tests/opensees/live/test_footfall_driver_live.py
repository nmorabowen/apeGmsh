"""Live tests for :meth:`apeSees.footfall_walking` (ADR 0109 S2).

The model is two tip-mass cantilevers standing side by side on one
deck (``make_two_column_frame``): each is an SDOF in global X, they
share no node, and their tip masses differ, so the basis is two
well-separated modes whose shapes are ``[φ, 0]`` and ``[0, φ]``. That
makes the off-diagonal FRF identically zero — which is exactly what
pins ``exc_node`` (the walker who governs is the one standing on the
occupant's own cantilever) and what makes ``excitation="full"`` agree
with ``excitation="self"`` on the diagonal.

Tuning the tip mass moves ``f_n`` at will, so the same fixture covers
both Design Guide branches: masses tuned to 12/14 Hz put the whole
sweep band above 9 Hz (``regime == "high"``, Eq 7-4…7-6 only), masses
tuned to 6/7 Hz open the low-frequency branch (Eq 7-1) as well.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees

# Module-level gate: skip every test if openseespy is not installed.
openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    make_two_column_frame,
)

# Two cantilevers, k = 3EI/L^3 = 6e7 each; the tip mass sets f_n.
_E, _IZ, _L = 200e9, 1e-4, 1.0
_K = 3.0 * _E * _IZ / _L**3
_SOLVER = "-fullGenLapack"

# SI: Q = 747 N (the Guide's 168 lb), g = 9.81 m/s^2.
_Q = 747.0
_G = 9.81


def _mass_for(f_n: float) -> float:
    """Tip mass giving a cantilever a natural frequency of ``f_n`` Hz."""
    return _K / (2.0 * np.pi * f_n) ** 2


def _two_cantilevers(f_a: float, f_b: float) -> "apeSees":
    """Node 2 tuned to ``f_a`` Hz, node 4 to ``f_b`` Hz."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=_E, Iz=_IZ, Iy=_IZ, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    tiny = (1e-6,) * 5
    ops.mass(nodes=[2], values=(_mass_for(f_a), *tiny))
    ops.mass(nodes=[4], values=(_mass_for(f_b), *tiny))
    return ops


def _high_frequency_floor() -> "apeSees":
    """Both cantilevers above 10 Hz — the whole band sits above 9 Hz."""
    return _two_cantilevers(12.0, 14.0)


def _low_frequency_floor() -> "apeSees":
    """Both cantilevers below 9 Hz — the Eq 7-1 branch opens."""
    return _two_cantilevers(6.0, 7.0)


@pytest.mark.live
def test_self_and_full_agree_on_the_diagonal() -> None:
    """``full`` reduces to ``self`` when the off-diagonal FRF is zero."""
    diagonal = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        dof=1, damp=0.03, solver=_SOLVER,
    )
    everything = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        excitation="full", dof=1, damp=0.03, solver=_SOLVER,
    )
    assert diagonal.nodes == (2, 4) == everything.nodes
    np.testing.assert_allclose(
        everything.a_p, diagonal.a_p, rtol=1e-12,
    )
    np.testing.assert_allclose(
        everything.f_dom, diagonal.f_dom, rtol=1e-12,
    )
    np.testing.assert_allclose(
        everything.ratio, diagonal.ratio, rtol=1e-12,
    )
    # ADR 0109 D2: the scale was asserted, not assumed.
    assert diagonal.normalization == "asserted(2 of 2)"


@pytest.mark.live
def test_exc_node_is_the_loaded_node() -> None:
    """Under ``self`` the walker is the occupant; under ``full`` on this
    fixture the governing walker is still the occupant's own cantilever."""
    result = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        dof=1, damp=0.03, solver=_SOLVER,
    )
    np.testing.assert_array_equal(result.exc_node, np.asarray([2, 4]))

    everything = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        excitation="full", excitation_nodes=[2, 4], dof=1, damp=0.03,
        solver=_SOLVER,
    )
    np.testing.assert_array_equal(everything.exc_node, np.asarray([2, 4]))


@pytest.mark.live
def test_regime_flips_when_the_tip_mass_crosses_9_hz() -> None:
    """Scaling the tip mass moves ``f_dom`` across 9 Hz and with it the
    branch that governs."""
    high = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        dof=1, damp=0.03, solver=_SOLVER,
    )
    assert list(high.regime) == ["high", "high"]
    assert np.all(high.f_dom >= 9.0)
    assert np.all(np.isnan(high.a_p_lf))
    assert np.all(np.isfinite(high.a_espa_hf))
    np.testing.assert_allclose(high.a_p, high.a_espa_hf, rtol=1e-12)

    low = _low_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        dof=1, damp=0.03, solver=_SOLVER,
    )
    assert list(low.regime) == ["both", "both"]
    assert np.all(low.f_dom < 9.0)
    assert np.all(np.isfinite(low.a_p_lf))
    np.testing.assert_allclose(low.a_p, low.a_p_lf, rtol=1e-12)
    # The limit is read at the governing dominant frequency, and the
    # ratio is the demand over it.
    np.testing.assert_allclose(
        low.ratio, low.a_p / low.limit, rtol=1e-12,
    )


@pytest.mark.live
def test_uniform_damp_equals_an_equal_modal_damp_list() -> None:
    """``damp=0.03`` and ``modal_damp=[0.03] * p`` are the same basis."""
    uniform = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        dof=1, damp=0.03, solver=_SOLVER,
    )
    per_mode = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        dof=1, modal_damp=[0.03, 0.03], solver=_SOLVER,
    )
    np.testing.assert_allclose(per_mode.a_p, uniform.a_p, rtol=1e-12)
    np.testing.assert_allclose(
        per_mode.a_espa_hf, uniform.a_espa_hf, rtol=1e-12,
    )
    np.testing.assert_allclose(
        per_mode.modes.beta, uniform.modes.beta, rtol=1e-12,
    )


@pytest.mark.live
def test_a_basis_short_of_f_max_warns_naming_eigen_feast() -> None:
    """The Eq 7-5 sum wants every mode up to ``f_max``; a truncated
    basis warns and points at the fork's band solve."""
    ops = _high_frequency_floor()
    with pytest.warns(UserWarning, match="eigen_feast"):
        result = ops.footfall_walking(
            num_modes=1, body_weight=_Q, g=_G, response_nodes=2,
            dof=1, damp=0.03, solver=_SOLVER,
        )
    assert result.nodes == (2,)
    assert result.modes.f_n.size == 1


@pytest.mark.live
def test_result_inspection_surfaces() -> None:
    """``frf``, ``mode_table`` and ``to_dataframe`` read back the
    tables the evaluation was built from."""
    result = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        dof=1, damp=0.03, solver=_SOLVER,
    )

    freq, mag = result.frf(2)
    assert freq.shape == mag.shape == result.freq.shape
    # The default excitation node is the one that governed.
    np.testing.assert_allclose(mag, result.frf(2, 2)[1], rtol=0.0)
    # frf_max is |A| at the governing dominant frequency.
    row = int(np.argmin(np.abs(freq - result.f_dom[0])))
    assert float(mag[row]) == pytest.approx(float(result.frf_max[0]))

    f_n, phi_i, phi_j, a_p_m = result.mode_table(2)
    assert f_n.shape == phi_i.shape == phi_j.shape == a_p_m.shape
    # Mode 2 lives on the other cantilever: no coupling at node 2.
    assert abs(float(phi_j[1])) < 1e-9
    # Eq 7-5 term for the local mode dominates the ESPA.
    assert abs(float(a_p_m[0])) > abs(float(a_p_m[1]))

    frame = result.to_dataframe()
    assert list(frame.index) == [2, 4]
    assert float(frame.loc[2, "a_p"]) == pytest.approx(float(result.a_p[0]))

    with pytest.raises(KeyError, match="not a response node"):
        result.frf(3)


@pytest.mark.live
def test_to_results_writes_a_native_map_with_nan_off_the_response_set(
    tmp_path,
) -> None:
    """ADR 0109 S3: ``to_results`` writes one static-kind stage, one
    time station, with the three components ``NaN`` outside the
    response nodes.

    ``Results.from_fem(fem, path, kind="native")`` is the documented
    read path (ADR 0109 D5), but it needs ``fem.to_native_h5`` /
    ``fem.snapshot_id`` to materialise a bound model — the lightweight
    ``FEMStub`` this file's fixture returns implements neither, so it
    cannot bind through ``from_fem``. Read back through
    ``NativeReader`` instead, the same low-level path
    ``tests/test_results_native_modes.py`` uses to verify a
    ``NativeWriter`` round-trip.
    """
    from apeGmsh.results.readers import NativeReader

    result = _high_frequency_floor().footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=[2, 4],
        dof=1, damp=0.03, solver=_SOLVER,
    )
    fem = make_two_column_frame()
    path = result.to_results(fem, tmp_path / "footfall.h5")
    assert path == tmp_path / "footfall.h5"

    with NativeReader(path) as r:
        stages = r.stages()
        assert len(stages) == 1
        stage = stages[0]
        assert stage.kind == "static"
        assert stage.n_steps == 1

        ap = r.read_nodes(stage.id, "footfall_ap")
        ratio = r.read_nodes(stage.id, "footfall_ratio")
        fdom = r.read_nodes(stage.id, "footfall_fdom")

    np.testing.assert_array_equal(ap.node_ids, np.asarray(fem.nodes.ids))
    for slab in (ap, ratio, fdom):
        assert slab.values.shape == (1, 4)

    # Response nodes are 2 and 4 (rows 1, 3 of fem.nodes.ids == [1,2,3,4]);
    # the base nodes 1 and 3 were never evaluated -> NaN.
    for slab in (ap, ratio, fdom):
        assert np.isnan(slab.values[0, 0])
        assert np.isnan(slab.values[0, 2])
        assert np.all(np.isfinite(slab.values[0, [1, 3]]))

    np.testing.assert_allclose(ap.values[0, [1, 3]], result.a_p, rtol=1e-12)
    np.testing.assert_allclose(
        ratio.values[0, [1, 3]], result.ratio, rtol=1e-12,
    )
    np.testing.assert_allclose(
        fdom.values[0, [1, 3]], result.f_dom, rtol=1e-12,
    )


@pytest.mark.live
def test_to_results_binds_through_from_fem_on_a_real_fem(
    g, tmp_path,
) -> None:
    """The how-to's documented read path, at a real (not stub) fem.

    ``test_to_results_writes_a_native_map_with_nan_off_the_response_set``
    above reads back through ``NativeReader`` because the lightweight
    ``FEMStub`` fixture cannot materialise a bound model
    (``fem.to_native_h5`` / ``fem.snapshot_id`` are missing). A fem from
    ``g.mesh.queries.get_fem_data`` (the how-to's own recipe) has both, so
    ``Results.from_fem`` should bind for real here and the ratio map
    should render, NaN nodes and all.
    """
    # Same 6 m distributed-mass floor bay as the how-to
    # (docs/how-to/footfall-vibration.md): mass=500 kg/m puts the first
    # two modes at ~16 Hz and ~65 Hz, so num_modes=2 clears f_max=20 Hz.
    L, area, e_mod, iz, mass_per_len = 6.0, 0.0083, 200e9, 3.5e-4, 500.0
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
    pm = g.model.geometry.add_point(L / 2, 0.0, 0.0)
    p1 = g.model.geometry.add_point(L, 0.0, 0.0)
    l0 = g.model.geometry.add_line(p0, pm)
    l1 = g.model.geometry.add_line(pm, p1)
    g.model.sync()
    g.physical.add(1, [l0, l1], name="Beam")
    g.physical.add(0, [p0], name="Left")
    g.physical.add(0, [p1], name="Right")
    g.physical.add(0, [pm], name="Midspan")
    g.mesh.sizing.set_global_size(L / 12.0)
    g.mesh.generation.generate(1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Beam", transf=transf, A=area, E=e_mod, Iz=iz,
        mass=mass_per_len,
    )
    ops.fix(pg="Left", dofs=(1, 1, 0))
    ops.fix(pg="Right", dofs=(0, 1, 0))
    midspan = ops.nodes.get(pg="Midspan")

    result = ops.footfall_walking(
        num_modes=2, body_weight=_Q, g=_G, response_nodes=midspan,
        dof=2, damp=0.03,
    )

    from apeGmsh.results import Results

    path = result.to_results(fem, tmp_path / "footfall_from_fem.h5")
    r = Results.from_fem(fem, path, kind="native", cache_root=tmp_path)
    assert r.fem is not None
    assert r.model is not None and r.model.fem is not None

    slab = r.nodes.get(component="footfall_ratio")
    assert slab.values.shape == (1, fem.nodes.ids.size)
    finite = np.isfinite(slab.values[0])
    assert int(finite.sum()) == 1  # exactly the one response node
    np.testing.assert_allclose(
        slab.values[0][finite], result.ratio, rtol=1e-12,
    )

    out = tmp_path / "footfall_ratio.png"
    picture = r.render(out, view="contour", component="footfall_ratio")
    if picture is None:
        assert not out.exists()
    else:
        assert picture == out
        assert out.exists()
        assert out.stat().st_size > 0
