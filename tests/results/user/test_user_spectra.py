"""``/spectra`` (ADR 0130 D5, A1, A7): cases from the driver's result, round trip.

The oracle of the card: a case read back holds the driver's in-memory values
bit for bit, and CQC over what the file holds, computed here from the textbook
formula (Der Kiureghian 1981, rigid-mode grouping of ADR 213 D3b), equals the
driver's own :meth:`SpectrumModalResult.combine`.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._internal.schema_version import SchemaVersionError
from apeGmsh.opensees.analysis.gravity_modal import GravityModalResult
from apeGmsh.opensees.analysis.modal import ModalPropertiesResult
from apeGmsh.opensees.analysis.spectrum import (
    DesignSpectrum,
    NodeDisp,
    NodeReaction,
    SpectrumModalResult,
)
from apeGmsh.results.user import (
    UserFile,
    UserWriteRefused,
    spectrum_cases,
    staleness,
    write_user_file,
)
from apeGmsh.results.user._spectra import same_bits

from tests.fixtures.schema import SPECTRA_CURRENT
from tests.results.user._user_helpers import case, cqc_reference

LAM = np.array([40.0, 400.0, 900.0, 2500.0])
TN, SA, SCALE = (0.0, 0.5, 1.0, 2.0), (3.0, 6.0, 4.0, 2.0), 0.75
XI = 0.05
#: The residual pseudo-modes (kind 1) read Sa at this period (ADR 213 D3b).
T_RES = 0.04
DISP = NodeDisp(nodes=[4, 7, 9], dofs=(1, 2))


def _rng_values(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


def _result(*, residual: bool = True, function: bool = False) -> SpectrumModalResult:
    """A result as ``spectrum_after_gravity`` builds it, with known values."""
    n = LAM.size
    props = {
        "eigenLambda": LAM.tolist(), "eigenPeriod": (2 * np.pi / np.sqrt(LAM)).tolist(),
        "partiFactorMX": [1.3, -0.4, 0.2, 0.05], "partiFactorMY": [0.1, 1.1, -0.3, 0.02],
        "modeKind": [0, 0, 1, 1] if residual else [0] * n,
    }
    modal = ModalPropertiesResult(eigenvalues=LAM, properties=props, _live=None)  # type: ignore[arg-type]
    gravity = GravityModalResult(modal=modal, gravity=None, p_delta=True,  # type: ignore[arg-type]
                                 gravity_steps=10, _live=None)  # type: ignore[arg-type]
    second = (DesignSpectrum(lambda t: 2.0 + 3.0 * np.asarray(t), direction=2, scale=2.0)
              if function else DesignSpectrum(TN, SA, direction=2, scale=2.0))
    spectra = (DesignSpectrum(TN, SA, direction=1, scale=SCALE), second)
    per_mode = {
        "disp": _rng_values(1, (2, n, 6)),
        "drift": _rng_values(2, (2, n, 1)),
        "cols": _rng_values(3, (2, n, 5)),
        "base_shear": _rng_values(4, (2, n, 2)),
    }
    labels = {"disp": DISP.labels, "drift": ("roof drift",), "cols": None, "base_shear": None}
    gravity_values = {k: _rng_values(10 + i, (v.shape[2],))
                      for i, (k, v) in enumerate(per_mode.items())}
    return SpectrumModalResult(
        modal=gravity, spectra=spectra, gravity_values=gravity_values, per_mode=per_mode,
        labels=labels, restore_drift=(0.0, 0.0),
        sa_periods=np.where(np.asarray(props["modeKind"]) == 1, T_RES, 0.0) if residual else None,
    )


# ---- the cases from the driver's result ---------------------------------------


def test_each_spectrum_is_a_case_holding_the_drivers_values() -> None:
    res = _result()
    cases = spectrum_cases(res, ["E_x", "E_y"], damping=XI, nodal=["disp"])
    assert [c.name for c in cases] == ["E_x", "E_y"]
    for k, c in enumerate(cases):
        assert (c.direction, c.scale) == (res.spectra[k].direction, res.spectra[k].scale)
        assert c.mode.tolist() == [1, 2, 3, 4] and c.kind == ("eigen", "eigen", "residual", "residual")
        assert c.lam.tobytes() == LAM.tobytes()
        assert c.sa_period.tolist() == [0.0, 0.0, T_RES, T_RES]
        token = ("partiFactorMX", "partiFactorMY")[k]
        assert c.gamma.tolist() == res.modal.modal.properties[token]
        assert c.damping.tolist() == [XI] * 4
        for name, label in (("drift", "drift"), ("cols", "cols"), ("base_shear", "base_shear")):
            q = c.result(label)
            assert q.target == "domain"
            assert q.data.tobytes() == res.per_mode[name][k][:, None, :].tobytes()
        disp = c.result("DISPLACEMENT")
        assert disp.target == "nodes" and disp.components == ("Ux", "Uy")
        assert disp.ids.tolist() == [4, 7, 9]
        assert disp.data.tobytes() == res.per_mode["disp"][k].reshape(4, 3, 2).tobytes()
        ref = {q.label: q for q in c.reference}
        assert ref["DISPLACEMENT"].data.tobytes() == res.gravity_values["disp"].tobytes()
        assert ref["drift"].data.reshape(-1).tolist() == res.gravity_values["drift"].tolist()


def test_components_are_the_labels_or_one_based_indices() -> None:
    c = spectrum_cases(_result(), ["E_x", "E_y"], damping=XI, nodal=["disp"])[0]
    assert c.result("drift").components == ("roof drift",)
    assert c.result("cols").components == ("1", "2", "3", "4", "5")
    assert c.result("base_shear").components == ("1", "2")


def test_sa_is_the_applied_spectrum_and_the_table_is_unscaled() -> None:
    c = spectrum_cases(_result(), ["E_x", "E_y"], damping=XI, nodal=["disp"])[0]
    at = np.where(c.sa_period > 0, c.sa_period, 2 * np.pi / np.sqrt(LAM))
    np.testing.assert_allclose(c.sa, SCALE * np.interp(at, TN, SA), rtol=1e-15)
    assert c.spectrum_periods.tolist() == list(TN) and c.spectrum_sa.tolist() == list(SA)


def test_a_function_spectrum_is_sampled_where_the_modes_read_it() -> None:
    c = spectrum_cases(_result(function=True), ["E_x", "E_y"], damping=XI, nodal=["disp"])[1]
    at = np.unique(np.where(c.sa_period > 0, c.sa_period, 2 * np.pi / np.sqrt(LAM)))
    assert c.spectrum_periods.tolist() == at.tolist()
    np.testing.assert_allclose(c.spectrum_sa, 2.0 + 3.0 * at, rtol=1e-15)
    np.testing.assert_allclose(c.sa, 2.0 * (2.0 + 3.0 * np.where(
        c.sa_period > 0, c.sa_period, 2 * np.pi / np.sqrt(LAM))), rtol=1e-15)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("name, label", [("drift", "drift"), ("base_shear", "base_shear"),
                                         ("disp", "DISPLACEMENT")])
def test_cqc_over_the_file_equals_the_drivers_combination(
        make_model, tmp_path: Path, residual: bool, name: str, label: str) -> None:
    res = _result(residual=residual)
    path = write_user_file(tmp_path / "s.user.h5", make_model(),
                           spectra=spectrum_cases(res, ["E_x", "E_y"], damping=XI, nodal=["disp"]))
    driver = res.combine(name, rule="CQC", damping=XI)
    for k, c in enumerate(UserFile(path).spectra().cases):
        per_mode = c.result(label).data.reshape(c.mode.size, -1)
        mine = cqc_reference(per_mode, c.lam, c.damping, c.sa_period)
        np.testing.assert_allclose(mine, driver[k], rtol=1e-12, atol=0.0)


def test_per_mode_damping_is_kept_per_mode(make_model, tmp_path: Path) -> None:
    xi = np.array([0.02, 0.05, 0.05, 0.07])
    res = _result()
    path = write_user_file(tmp_path / "s.user.h5", make_model(),
                           spectra=spectrum_cases(res, ["E_x", "E_y"], damping=xi, nodal=["disp"]))
    c = UserFile(path).spectra().cases[0]
    assert c.damping.tolist() == xi.tolist()
    mine = cqc_reference(c.result("drift").data.reshape(4, -1), c.lam, c.damping, c.sa_period)
    np.testing.assert_allclose(mine, res.combine("drift", damping=xi)[0], rtol=1e-12)


def test_no_damping_writes_nan() -> None:
    c = spectrum_cases(_result(), ["E_x", "E_y"], nodal=["disp"])[0]
    assert np.all(np.isnan(c.damping))


@pytest.mark.parametrize("damping", [0.0, 1.0, -0.1, np.nan, [0.05, 0.05]])
def test_damping_outside_0_1_refuses(damping) -> None:
    with pytest.raises(ValueError):
        spectrum_cases(_result(), ["E_x", "E_y"], damping=damping, nodal=["disp"])


# ---- A7: nodal fields on request, never by default, never dropped silently ----


def test_a_nodal_extractor_named_nowhere_refuses() -> None:
    with pytest.raises(ValueError, match="nodal="):
        spectrum_cases(_result(), ["E_x", "E_y"], damping=XI)


def test_omit_leaves_a_quantity_out() -> None:
    c = spectrum_cases(_result(), ["E_x", "E_y"], damping=XI, omit=["disp"])[0]
    assert {q.label for q in c.results} == {"drift", "cols", "base_shear"}
    assert {q.label for q in c.reference} == {"drift", "cols", "base_shear"}


@pytest.mark.parametrize("kw, word", [
    ({"nodal": ["nope"]}, "not extractors"),
    ({"omit": ["nope"]}, "not extractors"),
    ({"nodal": ["disp"], "omit": ["disp"]}, "both"),
    ({"nodal": ["drift"], "omit": ["disp"]}, "NodeDisp"),
])
def test_bad_requests_refuse(kw, word) -> None:
    with pytest.raises(ValueError, match=word):
        spectrum_cases(_result(), ["E_x", "E_y"], damping=XI, **kw)


def _with_labels(labels: tuple[str, ...]) -> SpectrumModalResult:
    res = _result()
    return SpectrumModalResult(
        modal=res.modal, spectra=res.spectra, gravity_values=res.gravity_values,
        per_mode=res.per_mode, labels={**res.labels, "disp": labels},
        restore_drift=res.restore_drift, sa_periods=res.sa_periods)


def test_reactions_and_rotations_are_not_nodal_fields_here() -> None:
    reaction = _with_labels(NodeReaction(nodes=[4, 7, 9], dofs=(1, 2)).labels)
    with pytest.raises(ValueError, match="NodeDisp"):
        spectrum_cases(reaction, ["E_x", "E_y"], damping=XI, nodal=["disp"])
    rotation = _with_labels(NodeDisp(nodes=[4, 7], dofs=(1, 2, 3)).labels)
    with pytest.raises(ValueError, match="translational"):
        spectrum_cases(rotation, ["E_x", "E_y"], damping=XI, nodal=["disp"])


@pytest.mark.parametrize("names, word", [(["E_x"], "2 spectra"), (["E", "E"], "repeat"),
                                         (["E_x", " E_y"], "whitespace")])
def test_bad_case_names_refuse(names, word) -> None:
    with pytest.raises(ValueError, match=word):
        spectrum_cases(_result(), names, damping=XI, nodal=["disp"])


def test_anything_but_the_drivers_result_refuses() -> None:
    with pytest.raises(TypeError, match="SpectrumModalResult"):
        spectrum_cases(object(), ["E_x"])  # type: ignore[arg-type]


# ---- the zone ---------------------------------------------------------------


def test_a_case_round_trips_bit_for_bit(make_model, tmp_path: Path) -> None:
    c = case()
    path = write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[c])
    index = UserFile(path).spectra()
    assert index.refused == () and same_bits(index.by_name("E_x (DBE)"), c)
    with h5py.File(path, "r") as f:
        g = f["spectra/s0"]
        assert g.attrs["name"] == "E_x (DBE)" and g.attrs["KIND"] == "static"
        assert g.attrs["direction"].dtype == np.dtype("i4")
        assert g["RESULTS/ON_NODES/DISPLACEMENT/ID"].shape == (2, 1)
        assert g["reference/RESULTS/ON_DOMAIN/base_shear/DATA"].shape == (1, 1, 2)
        assert g["RESULTS/ON_DOMAIN/base_shear/STEP"][...].tolist() == [0, 1]
        assert f["meta"].attrs["spectra_schema_version"] == SPECTRA_CURRENT


def test_spectra_replace_the_zone_and_none_keeps_it(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[case("a"), case("b")])
    write_user_file(path, make_model(), units="kN,m")
    assert [c.name for c in UserFile(path).spectra().cases] == ["a", "b"]
    write_user_file(path, make_model(), spectra=[case("c")])
    assert [c.name for c in UserFile(path).spectra().cases] == ["c"]


def test_two_cases_of_one_name_refuse_the_write(make_model, tmp_path: Path) -> None:
    with pytest.raises(UserWriteRefused, match="twice"):
        write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[case("a"), case("a")])


@pytest.mark.parametrize("kw, word", [
    ({"lam": np.array([40.0, 0.0])}, "lambda"),
    ({"damping": np.array([0.05, 1.5])}, "damping"),
    ({"spectrum_periods": np.array([0.0, 2.0, 0.5])}, "increasing"),
    ({"sa": np.array([4.5, -1.0])}, "sa must"),
    ({"direction": 0}, "1-based"),
    ({"kind": ("eigen", "unknown")}, "kind"),
])
def test_a_bad_case_refuses_the_write(make_model, tmp_path: Path, kw, word) -> None:
    with pytest.raises(UserWriteRefused, match=word):
        write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[case(**kw)])


def test_reference_must_mirror_the_results(make_model, tmp_path: Path) -> None:
    c = case()
    bad = case(reference=c.reference[:1])
    with pytest.raises(UserWriteRefused, match="reference"):
        write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[bad])


def test_a_damaged_case_refuses_alone(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[case("a"), case("b")])
    with h5py.File(path, "a") as f:
        del f["spectra/s0/modes/gamma"]
    index = UserFile(path).spectra()
    assert [c.name for c in index.cases] == ["b"]
    assert [(r.name, r.reason) for r in index.refused] == [("a", "bad-param")]


def test_namesakes_both_refuse_as_duplicate(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[case("a"), case("b")])
    with h5py.File(path, "a") as f:
        f["spectra/s1"].attrs["name"] = "a"
    index = UserFile(path).spectra()
    assert index.cases == ()
    assert sorted(r.reason for r in index.refused) == ["duplicate", "duplicate"]


def test_a_newer_zone_refuses(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[case()])
    with h5py.File(path, "a") as f:
        f["meta"].attrs["spectra_schema_version"] = "2.0.0"
    with pytest.raises(SchemaVersionError):
        UserFile(path).spectra()
    assert staleness(path, model_h5=make_model()).zones["spectra"].stale is False, (
        "staleness of /spectra is by the model hashes alone (A6)")


def test_spectra_go_stale_with_the_model_and_a_rewrite_needs_no_drop(make_model, tmp_path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), spectra=[case()])
    other = make_model("other.h5", fem="other")
    assert staleness(path, model_h5=other).zones["spectra"].stale
    with pytest.raises(UserWriteRefused, match="spectra"):
        write_user_file(path, other)
    write_user_file(path, other, spectra=[case()])
    assert not staleness(path, model_h5=other).zones["spectra"].stale
