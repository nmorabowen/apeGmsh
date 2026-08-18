"""Per-element out-of-plane recovery: model-record parsing + σ_zz/ε_zz fill."""
from __future__ import annotations

import warnings
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
    pr._WARNED.clear()
    yield
    pr._CACHE.clear()
    pr._WARNED.clear()


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


def test_ladruno_lst_classified():
    """The 6-node LST is a flag-style 2-D element like its CST sibling."""
    # no -type flag → the fork's PlaneStrain default
    m1 = _model(
        [_elem("LadrunoLST", (4, "-geom", "finite", "-thick", 1.0), 1)],
        [_mat("ElasticIsotropic", 4, (1.0, 0.3, 0.0))],
    )
    assert pr.plane_recovery_map(m1) == {1: ("PlaneStrain", 0.3)}
    pr._CACHE.clear()
    m2 = _model(
        [_elem("LadrunoLST", (4, "-type", "PlaneStress", "-thick", 1.0), 2)],
        [_mat("ElasticIsotropic", 4, (1.0, 0.3, 0.0))],
    )
    assert pr.plane_recovery_map(m2) == {2: ("PlaneStress", 0.3)}


def test_ladruno_up_2d_is_plane_strain_3d_is_not():
    """LadrunoUP spans both dimensions; the perm component count decides."""
    m2d = _model(
        [_elem("LadrunoUP",
               (6, "-thick", 1.0, "-Kf", 2.2e9, "-poro", 0.4, "-rhoF", 1000.0,
                "-perm", 1e-9, 1e-9), 1)],
        [_mat("ElasticIsotropic", 6, (1.0, 0.3, 0.0))],
    )
    assert pr.plane_recovery_map(m2d) == {1: ("PlaneStrain", 0.3)}
    pr._CACHE.clear()
    m3d = _model(
        [_elem("LadrunoUP",
               (6, "-Kf", 2.2e9, "-poro", 0.4, "-rhoF", 1000.0,
                "-permH", 1e-5, 1e-5, 1e-5, "-gammaW", 9810.0), 1)],
        [_mat("ElasticIsotropic", 6, (1.0, 0.3, 0.0))],
    )
    assert pr.plane_recovery_map(m3d) == {}


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


# ---------------------------------------------------------------------
# Unrecoverable elements are reported, never silent
# ---------------------------------------------------------------------

def test_warns_and_names_unclassifiable_token():
    """The whole point: an unknown 2-D token must not fail silently."""
    model = _model(
        [_elem("SomeFutureTri9", (7, "-thick", 1.0), fem_eid=1)],
        [_mat("ElasticIsotropic", 7, (1.0, 0.3, 0.0))],
    )
    cols = _stress_cols([10.0], [0.0])
    with pytest.warns(pr.OutOfPlaneRecoveryWarning, match="SomeFutureTri9"):
        assert pr.inject_out_of_plane(
            cols, np.array([1]), prefix="stress", model=model,
        ) is False
    assert "stress_zz" not in cols


def test_warns_when_plane_strain_material_nu_unreadable():
    model = _model(
        [_elem("quad", (0.5, "PlaneStrain", 1), fem_eid=1)],
        [_mat("SomeExoticNDMaterial", 1, (1.0, 2.0, 3.0))],   # no ν to read
    )
    cols = _stress_cols([10.0], [0.0])
    with pytest.warns(pr.OutOfPlaneRecoveryWarning, match="Poisson"):
        pr.inject_out_of_plane(cols, np.array([1]), prefix="stress", model=model)
    np.testing.assert_allclose(cols["stress_zz"], [[0.0]])


def test_no_warning_when_zero_is_the_exact_answer():
    """Plane stress → σ_zz = 0 exactly; nothing to report."""
    model = _model(
        [_elem("quad", (0.5, "PlaneStress", 1), fem_eid=1)],
        [_mat("ElasticIsotropic", 1, (1.0, 0.3, 0.0))],
    )
    cols = _stress_cols([10.0], [0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", pr.OutOfPlaneRecoveryWarning)
        pr.inject_out_of_plane(cols, np.array([1]), prefix="stress", model=model)


def test_warning_fires_once_per_situation():
    """The viewer re-reads every frame — one warning, not one per redraw."""
    model = _model([_elem("SomeFutureTri9", (7,), fem_eid=1)], [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            pr.inject_out_of_plane(
                _stress_cols([10.0], [0.0]), np.array([1]),
                prefix="stress", model=model,
            )
    assert sum(
        issubclass(w.category, pr.OutOfPlaneRecoveryWarning) for w in caught
    ) == 1


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


# ---------------------------------------------------------------------
# A recorded σ_zz: finite is silent, NaN names the material
# ---------------------------------------------------------------------

def _recorded_zz_model():
    return _model(
        [_elem("LadrunoLST", (1,), fem_eid=1)],
        [_mat("LadrunoJ2", 1, (1.333e8, 8e7, 1.2e5))],
    )


def test_recorded_finite_zz_raises_no_warning():
    """A genuinely recorded σ_zz is the answer — nothing to report."""
    cols = {"stress_xx": np.array([[10.0]]), "stress_yy": np.array([[2.0]]),
            "stress_zz": np.array([[7.0]])}
    with warnings.catch_warnings():
        warnings.simplefilter("error", pr.OutOfPlaneRecoveryWarning)
        assert pr.inject_out_of_plane(
            cols, np.array([1]), prefix="stress",
            model=_recorded_zz_model(),
        ) is False


def test_recorded_nan_zz_warns_about_the_material_not_the_model():
    """A NaN σ_zz means ``NDMaterial::getStressZZ`` had no override — the
    element and its idealization parsed fine.  Pointing the user at
    ``plane=`` / ``nu=`` (the unclassifiable-element remedy) would send
    them the wrong way, so the message must be a different one."""
    cols = {"stress_xx": np.array([[10.0]]), "stress_yy": np.array([[0.0]]),
            "stress_zz": np.array([[np.nan]])}
    with pytest.warns(pr.OutOfPlaneRecoveryWarning) as caught:
        pr.inject_out_of_plane(
            cols, np.array([1]), prefix="stress", model=_recorded_zz_model(),
        )
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert "getStressZZ" in messages[0]
    assert "could not be classified" not in messages[0]


def test_recorded_nan_zz_warning_fires_once():
    """Same dedupe contract as the unrecovered warning — the viewer
    re-reads every frame."""
    model = _recorded_zz_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(4):
            pr.inject_out_of_plane(
                {"stress_xx": np.array([[10.0]]),
                 "stress_yy": np.array([[0.0]]),
                 "stress_zz": np.array([[np.nan]])},
                np.array([1]), prefix="stress", model=model,
            )
    assert sum(
        issubclass(w.category, pr.OutOfPlaneRecoveryWarning) for w in caught
    ) == 1
