"""``/modes`` (ADR 0130 D6, A1, A6): the table, its round trip, the row rule."""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._internal.schema_version import SchemaVersionError
from apeGmsh.results.user import (
    UserFile,
    UserWriteRefused,
    ZoneRefused,
    mode_table,
    staleness,
    write_user_file,
)

from tests.results.user._user_helpers import MODES_LAMBDA, REPO, mode_rows

MODES_LADRUNO = REPO / "apeGmshViewer" / "fixtures" / "ladruno_modes.ladruno"

#: A two-storey shear frame: storey masses and stiffnesses.
M1, M2, K1, K2 = 2.0e4, 1.5e4, 3.0e7, 2.0e7
#: Each mode's eigenvector is scaled by this, so its generalised mass is SCALE²,
#: not the 1 of a mass-normalised mode.
SCALE = (3.7, -0.2)


def _shear_frame_properties() -> tuple[dict, np.ndarray]:
    """A ``modalProperties -return`` dict built from the closed form, and ``φᵀMφ``.

    ``Γ = φᵀMr / φᵀMφ``, ``M_eff = (φᵀMr)² / φᵀMφ`` and the ratios in percent,
    as ``DomainModalProperties`` returns them; the frame moves along X only.
    """
    m = np.diag([M1, M2])
    k = np.array([[K1 + K2, -K2], [-K2, K2]])
    lam, phi = np.linalg.eig(np.linalg.solve(m, k))
    order = np.argsort(lam)
    lam, phi = lam[order], phi[:, order]
    phi = phi / np.sqrt(np.einsum("im,ij,jm->m", phi, m, phi)) * np.array(SCALE)
    mgen = np.einsum("im,ij,jm->m", phi, m, phi)
    lx = phi.T @ m @ np.ones(2)
    gamma = lx / mgen
    meff = lx * lx / mgen
    pct = 100.0 * meff / (M1 + M2)
    zero = [0.0, 0.0]
    props = {
        "eigenLambda": lam.tolist(), "eigenPeriod": (2 * np.pi / np.sqrt(lam)).tolist(),
        "partiFactorMX": gamma.tolist(), "partiFactorMY": zero, "partiFactorRMZ": zero,
        "partiMassMX": meff.tolist(), "partiMassMY": zero, "partiMassRMZ": zero,
        "partiMassRatiosMX": pct.tolist(), "partiMassRatiosMY": zero, "partiMassRatiosRMZ": zero,
        "partiMassRatiosCumuMX": np.cumsum(pct).tolist(), "partiMassRatiosCumuMY": zero,
        "partiMassRatiosCumuRMZ": zero, "totalMass": [M1 + M2, M1 + M2, 0.0],
    }
    return props, mgen


# ---- the table from modalProperties -----------------------------------------


def test_the_table_is_modal_properties_with_fractions() -> None:
    props, _ = _shear_frame_properties()
    t = mode_table(props, host="stage_0")
    assert t.components == ("MX", "MY", "RMZ")
    assert t.lam.tobytes() == np.asarray(props["eigenLambda"]).tobytes()
    assert t.period.tobytes() == np.asarray(props["eigenPeriod"]).tobytes()
    assert t.participation[:, 0].tobytes() == np.asarray(props["partiFactorMX"]).tobytes()
    assert np.array_equal(t.mass_ratio[:, 0], np.asarray(props["partiMassRatiosMX"]) / 100.0)
    assert np.array_equal(t.cumulative_ratio[:, 0], np.asarray(props["partiMassRatiosCumuMX"]) / 100.0)
    assert t.mode.tolist() == [1, 2] and t.step.tolist() == [-1, -1]
    assert t.kind == ("eigen", "eigen") and t.host == ("stage_0", "stage_0")


def test_generalized_mass_is_phi_m_phi() -> None:
    """A1: effective mass over Γ² is ``φᵀMφ``, here SCALE², not 1."""
    props, mgen = _shear_frame_properties()
    t = mode_table(props, host="stage_0")
    np.testing.assert_allclose(t.generalized_mass, mgen, rtol=1e-12)
    np.testing.assert_allclose(t.generalized_mass, np.square(SCALE), rtol=1e-12)


def test_generalized_mass_is_nan_where_every_gamma_is_zero() -> None:
    props, _ = _shear_frame_properties()
    props["partiFactorMX"] = [props["partiFactorMX"][0], 0.0]
    t = mode_table(props, host="stage_0")
    assert np.isfinite(t.generalized_mass[0]) and np.isnan(t.generalized_mass[1])


def test_mode_kinds_follow_the_fork_tags() -> None:
    props, _ = _shear_frame_properties()
    props["modeKind"] = [0, 1]
    assert mode_table(props, host="stage_0").kind == ("eigen", "residual")
    props["modeKind"] = [0, 7]
    with pytest.raises(ValueError, match="modeKind"):
        mode_table(props, host="stage_0")


@pytest.mark.parametrize("drop", ["eigenLambda", "eigenPeriod", "partiMassMX"])
def test_a_dict_missing_a_series_refuses(drop: str) -> None:
    props, _ = _shear_frame_properties()
    del props[drop]
    with pytest.raises(ValueError, match=drop):
        mode_table(props, host="stage_0")


def test_a_non_positive_lambda_refuses() -> None:
    props, _ = _shear_frame_properties()
    props["eigenLambda"] = [0.0, props["eigenLambda"][1]]
    with pytest.raises(ValueError, match="lambda"):
        mode_table(props, host="stage_0")


# ---- the zone ---------------------------------------------------------------


def test_the_table_round_trips_bit_for_bit(make_model, tmp_path: Path) -> None:
    t = mode_rows()
    path = write_user_file(tmp_path / "s.user.h5", make_model(), modes=t)
    back = UserFile(path).modes()
    assert back.same_bits(t)
    assert np.isnan(back.generalized_mass[1])
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["modes_schema_version"] == "1.0.0"
        assert f["modes/step"].dtype == np.dtype("i4")
        assert f["modes"].attrs["components"] == "MX,MY,RMZ"


def test_a_second_write_replaces_the_table(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), modes=mode_rows())
    one = mode_rows(host=("stage_0",), kind=("eigen",), step=np.array([-1], dtype="i4"),
                    mode=np.array([1], dtype="i4"), lam=np.array([MODES_LAMBDA[0]]),
                    period=np.array([1.0]), generalized_mass=np.array([2.0]),
                    participation=np.ones((1, 3)), mass_ratio=np.ones((1, 3)) / 2,
                    cumulative_ratio=np.ones((1, 3)) / 2)
    write_user_file(path, make_model(), modes=one)
    assert len(UserFile(path).modes()) == 1


def test_a_write_without_modes_keeps_them(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), modes=mode_rows())
    write_user_file(path, make_model(), units="kN,m")
    assert UserFile(path).modes().same_bits(mode_rows())


def test_rewriting_modes_under_a_new_model_needs_no_drop(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), modes=mode_rows())
    other = make_model("other.h5", fem="other")
    with pytest.raises(UserWriteRefused, match="modes"):
        write_user_file(path, other)
    write_user_file(path, other, modes=mode_rows())
    assert UserFile(path).fem_hash == "other"
    assert not staleness(path, model_h5=other).zones["modes"].stale


@pytest.mark.parametrize("kw, word", [
    ({"components": ("MX", "MQ", "RMZ")}, "components"),
    ({"kind": ("eigen", "spooky")}, "kinds"),
    ({"mode": np.array([1, 1], dtype="i4")}, "same"),
    ({"mode": np.array([0, 1], dtype="i4")}, "1-based"),
    ({"lam": np.array([381.9, -1.0])}, "lambda"),
    ({"period": np.array([1.0, np.nan])}, "period"),
    ({"host": ("stage_0", " stage_0")}, "host"),
    ({"mass_ratio": np.ones((2, 2))}, "mass_ratio"),
])
def test_a_bad_table_refuses_the_write(make_model, tmp_path: Path, kw, word) -> None:
    with pytest.raises(UserWriteRefused, match=word):
        write_user_file(tmp_path / "s.user.h5", make_model(), modes=mode_rows(**kw))
    assert not (tmp_path / "s.user.h5").exists()


def test_malformed_columns_refuse_the_zone_and_the_rest_reads(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), modes=mode_rows(),
                           combinations=[])
    with h5py.File(path, "a") as f:
        del f["modes/lambda"]
        f["modes"].create_dataset("lambda", data=np.array([1, 2], dtype="i8"))
    uf = UserFile(path)
    with pytest.raises(ZoneRefused, match="lambda"):
        uf.modes()
    assert uf.combinations().rows == ()


def test_a_newer_zone_refuses_to_read_and_to_be_rewritten(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), modes=mode_rows())
    with h5py.File(path, "a") as f:
        f["meta"].attrs["modes_schema_version"] = "2.0.0"
    with pytest.raises(SchemaVersionError):
        UserFile(path).modes()
    before = path.read_bytes()
    with pytest.raises(SchemaVersionError):
        write_user_file(path, make_model(), modes=mode_rows())
    assert path.read_bytes() == before


def test_a_file_without_modes_refuses_the_read(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model())
    with pytest.raises(ZoneRefused, match="no modal table"):
        UserFile(path).modes()


# ---- A6: a row is stale when its lambda moved ---------------------------------


@pytest.fixture
def modes_set(make_model, tmp_path: Path):
    ladruno = tmp_path / "modes.ladruno"
    shutil.copy(MODES_LADRUNO, ladruno)
    model = make_model()
    return model, ladruno


def _rows(path: Path, model: Path, ladruno: Path):
    return staleness(path, model_h5=model, results_file=ladruno).modes


def test_rows_matching_the_ladruno_are_current(modes_set, tmp_path: Path) -> None:
    model, ladruno = modes_set
    path = write_user_file(tmp_path / "s.user.h5", model, modes=mode_rows())
    rows = _rows(path, model, ladruno)
    assert [r.stale for r in rows] == [False, False], rows


@pytest.mark.parametrize("rel, stale", [(5e-10, False), (2e-9, True), (-2e-9, True)])
def test_a_row_whose_lambda_moved_past_1e9_is_stale(modes_set, tmp_path, rel, stale) -> None:
    model, ladruno = modes_set
    lam = np.array([MODES_LAMBDA[0], MODES_LAMBDA[1] * (1.0 + rel)])
    path = write_user_file(tmp_path / "s.user.h5", model, modes=mode_rows(lam=lam))
    rows = _rows(path, model, ladruno)
    assert not rows[0].stale
    assert rows[1].stale is stale
    if stale:
        assert "relative" in rows[1].reason


def test_a_rerun_with_other_eigenvalues_stales_the_rows(modes_set, tmp_path: Path) -> None:
    model, ladruno = modes_set
    path = write_user_file(tmp_path / "s.user.h5", model, modes=mode_rows())
    with h5py.File(ladruno, "a") as f:
        for group in ("MODES_OF_VIBRATION(U)", "MODES_OF_VIBRATION(R)"):
            ds = f[f"MODEL_STAGE[1]/RESULTS/ON_NODES/{group}/DATA/STEP_0/MODE_0"]
            ds.attrs["LAMBDA"] = np.asarray(ds.attrs["LAMBDA"]) * 1.01
    rows = _rows(path, model, ladruno)
    assert [r.stale for r in rows] == [True, False]


@pytest.mark.parametrize("kw, word", [
    ({"host": ("gravity", "gravity")}, "stage_<i>"),
    ({"host": ("stage_1", "stage_1")}, "no modes on stage_1"),
    ({"step": np.array([1, 1], dtype="i4")}, "STEP_1"),
    ({"mode": np.array([1, 3], dtype="i4")}, "no mode 3"),
])
def test_a_row_the_ladruno_cannot_match_is_stale(modes_set, tmp_path, kw, word) -> None:
    model, ladruno = modes_set
    path = write_user_file(tmp_path / "s.user.h5", model, modes=mode_rows(**kw))
    rows = _rows(path, model, ladruno)
    assert rows[-1].stale and word in rows[-1].reason, rows


def test_an_explicit_step_matches_its_set(modes_set, tmp_path: Path) -> None:
    model, ladruno = modes_set
    path = write_user_file(tmp_path / "s.user.h5", model,
                           modes=mode_rows(step=np.array([0, 0], dtype="i4")))
    assert not any(r.stale for r in _rows(path, model, ladruno))


def test_rows_of_a_stale_zone_are_stale(modes_set, make_model, tmp_path: Path) -> None:
    model, ladruno = modes_set
    path = write_user_file(tmp_path / "s.user.h5", model, modes=mode_rows())
    got = staleness(path, model_h5=make_model("other.h5", fem="other"), results_file=ladruno)
    assert got.zones["modes"].stale
    assert all(r.stale and "fem_hash" in r.reason for r in got.modes)


def test_rows_with_no_results_file_are_stale(modes_set, tmp_path: Path) -> None:
    model, _ = modes_set
    path = write_user_file(tmp_path / "s.user.h5", model, modes=mode_rows())
    rows = staleness(path, model_h5=model).modes
    assert all(r.stale and "no results file" in r.reason for r in rows)
