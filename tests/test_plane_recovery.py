"""Per-element out-of-plane recovery: model-record parsing + σ_zz/ε_zz fill."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results import _plane_recovery as pr


def _model(elements, materials):
    return SimpleNamespace(
        elements=lambda: elements,
        materials=lambda family=None: materials,
    )


def _elem(token, args, fem_eid):
    return SimpleNamespace(type_token=token, args=tuple(args), fem_eid=fem_eid)


def _mat(token, tag, params):
    return SimpleNamespace(type_token=token, tag=tag, params=tuple(params))


@pytest.fixture(autouse=True)
def _clear_cache():
    pr._CACHE.clear()
    yield
    pr._CACHE.clear()


# ---------------------------------------------------------------------
# Record parsing → {fem_eid: (plane_type, nu)}
# ---------------------------------------------------------------------

def test_positional_quad_plane_strain_elastic():
    model = _model(
        [_elem("quad", (0.5, "PlaneStrain", 7), fem_eid=1)],
        [_mat("ElasticIsotropic", 7, (30e9, 0.2, 2400.0))],
    )
    assert pr.plane_recovery_map(model) == {1: ("PlaneStrain", 0.2)}


def test_tri31_plane_stress():
    model = _model(
        [_elem("tri31", (0.5, "PlaneStress", 3), fem_eid=9)],
        [_mat("ElasticIsotropic", 3, (1.0, 0.25, 0.0))],
    )
    assert pr.plane_recovery_map(model) == {9: ("PlaneStress", 0.25)}


def test_ladruno_quad_type_flag_and_default():
    # explicit -type flag
    m1 = _model(
        [_elem("LadrunoQuad", (5, "-type", "PlaneStress", "-thick", 0.2), 1)],
        [_mat("ElasticIsotropic", 5, (1.0, 0.3, 0.0))],
    )
    assert pr.plane_recovery_map(m1) == {1: ("PlaneStress", 0.3)}
    pr._CACHE.clear()
    # no -type flag → PlaneStrain default
    m2 = _model(
        [_elem("LadrunoCST", (5,), 2)],
        [_mat("ElasticIsotropic", 5, (1.0, 0.3, 0.0))],
    )
    assert pr.plane_recovery_map(m2) == {2: ("PlaneStrain", 0.3)}


def test_kg_material_nu_derivation():
    K, G = 5.0, 3.0
    model = _model(
        [_elem("quad", (0.5, "PlaneStrain", 1), fem_eid=1)],
        [_mat("J2Plasticity", 1, (K, G, 250.0, 300.0, 0.1, 0.0, 0.0))],
    )
    nu = pr.plane_recovery_map(model)[1][1]
    assert nu == pytest.approx((3 * K - 2 * G) / (2 * (3 * K + G)))


def test_asdplastic_poissons_ratio_token_scan():
    model = _model(
        [_elem("quad", (0.5, "PlaneStrain", 1), fem_eid=1)],
        [_mat("ASDPlasticMaterial3D", 1,
              ("YoungsModulus", 2.0e7, "PoissonsRatio", 0.33, "Cohesion", 1e4))],
    )
    assert pr.plane_recovery_map(model)[1] == ("PlaneStrain", pytest.approx(0.33))


def test_unparseable_element_absent():
    model = _model(
        [_elem("stdBrick", (999,), fem_eid=1)],   # 3-D element, no plane_type
        [_mat("ElasticIsotropic", 999, (1.0, 0.2, 0.0))],
    )
    assert pr.plane_recovery_map(model) == {}


# ---------------------------------------------------------------------
# Column injection
# ---------------------------------------------------------------------

def _stress_cols(xx, yy):
    return {
        "stress_xx": np.array([xx], dtype=float),
        "stress_yy": np.array([yy], dtype=float),
    }


def test_inject_plane_strain_stress():
    model = _model(
        [_elem("quad", (0.5, "PlaneStrain", 1), fem_eid=1)],
        [_mat("ElasticIsotropic", 1, (1.0, 0.3, 0.0))],
    )
    cols = _stress_cols([10.0], [0.0])
    ok = pr.inject_out_of_plane(cols, np.array([1]), prefix="stress", model=model)
    assert ok
    np.testing.assert_allclose(cols["stress_zz"], [[0.3 * 10.0]])


def test_inject_plane_stress_strain():
    model = _model(
        [_elem("quad", (0.5, "PlaneStress", 1), fem_eid=1)],
        [_mat("ElasticIsotropic", 1, (1.0, 0.25, 0.0))],
    )
    cols = {
        "strain_xx": np.array([[0.01]], dtype=float),
        "strain_yy": np.array([[0.0]], dtype=float),
    }
    pr.inject_out_of_plane(cols, np.array([1]), prefix="strain", model=model)
    np.testing.assert_allclose(cols["strain_zz"], [[-0.25 / 0.75 * 0.01]])


def test_inject_mixed_model_per_element():
    model = _model(
        [_elem("quad", (0.5, "PlaneStrain", 1), fem_eid=1),
         _elem("quad", (0.5, "PlaneStress", 1), fem_eid=2)],
        [_mat("ElasticIsotropic", 1, (1.0, 0.3, 0.0))],
    )
    cols = _stress_cols([10.0, 10.0], [0.0, 0.0])
    pr.inject_out_of_plane(cols, np.array([1, 2]), prefix="stress", model=model)
    # elem 1 (plane strain) → 0.3·10 = 3; elem 2 (plane stress) → 0
    np.testing.assert_allclose(cols["stress_zz"], [[3.0, 0.0]])


def test_inject_noop_when_zz_present():
    cols = {"stress_xx": np.array([[1.0]]), "stress_yy": np.array([[1.0]]),
            "stress_zz": np.array([[9.0]])}
    model = _model([_elem("quad", (0.5, "PlaneStrain", 1), 1)],
                   [_mat("ElasticIsotropic", 1, (1.0, 0.3, 0.0))])
    assert pr.inject_out_of_plane(cols, np.array([1]), prefix="stress", model=model) is False
    np.testing.assert_allclose(cols["stress_zz"], [[9.0]])   # untouched


def test_inject_noop_when_model_empty():
    model = _model([], [])
    cols = _stress_cols([10.0], [0.0])
    assert pr.inject_out_of_plane(cols, np.array([1]), prefix="stress", model=model) is False
    assert "stress_zz" not in cols


def test_recorded_zz_finite_used_verbatim():
    # A fully-recorded (finite) σ_zz is left untouched (no reconstruction).
    cols = {"stress_xx": np.array([[10.0]]), "stress_yy": np.array([[0.0]]),
            "stress_zz": np.array([[7.0]])}
    model = _model([_elem("quad", (0.5, "PlaneStrain", 1), 1)],
                   [_mat("ElasticIsotropic", 1, (1.0, 0.3, 0.0))])
    assert pr.inject_out_of_plane(cols, np.array([1]), prefix="stress", model=model) is False
    np.testing.assert_allclose(cols["stress_zz"], [[7.0]])


def test_recorded_zz_nan_sentinel_reconstructed_per_gp():
    # NaN sentinel (material couldn't supply σ_zz) is filled; finite kept.
    model = _model(
        [_elem("quad", (0.5, "PlaneStrain", 1), fem_eid=1),
         _elem("quad", (0.5, "PlaneStrain", 1), fem_eid=2)],
        [_mat("ElasticIsotropic", 1, (1.0, 0.3, 0.0))],
    )
    cols = {
        "stress_xx": np.array([[10.0, 10.0]]),
        "stress_yy": np.array([[0.0, 0.0]]),
        "stress_zz": np.array([[np.nan, 5.0]]),   # gp0 unavailable, gp1 recorded
    }
    ok = pr.inject_out_of_plane(cols, np.array([1, 2]), prefix="stress", model=model)
    assert ok
    # gp0 reconstructed = 0.3·10 = 3; gp1 keeps the recorded 5.
    np.testing.assert_allclose(cols["stress_zz"], [[3.0, 5.0]])
