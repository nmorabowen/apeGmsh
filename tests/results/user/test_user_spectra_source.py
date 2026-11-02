"""Spectrum terms read through ``/spectra`` (ADR 0130 D5, A6, A7; slice VU2b).

Oracles:

* the conformance goldens of ``tests/fixtures/user_results/conformance.json``,
  with the fixture's cases written to a ``.user.h5`` and read back through
  :class:`UserSpectra`, one recorded component at a time;
* the kernels of ``apeGmsh.opensees.analysis.spectrum`` (``combine_modes`` with
  ``sa_periods``, ``combine_directions``, ``seismic_envelope``) over the values
  read back;
* ``Results.combinations()`` with no ``spectra=``, against the formula CQC of
  the case's own columns (``cqc_reference``) plus the plain ``Results`` read.
"""
from __future__ import annotations

import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pytest

from apeGmsh.opensees.analysis.spectrum import (
    combine_directions,
    combine_modes,
    seismic_envelope,
)
from apeGmsh.results import Results
from apeGmsh.results.user import (
    CaseQuantity,
    Combination,
    SpectrumCase,
    StateAddress,
    Term,
    UserSpectra,
    write_user_file,
)
from apeGmsh.results.user._descriptor import resolve
from apeGmsh.results.user._evaluate import Case, CombinationRefused, Field, evaluate
from apeGmsh.results.user._states import node_keys
from tests.results.user._user_helpers import FEM_HASH, LADRUNO, case, cqc_reference

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "user_results"
DOC = json.loads((FIXTURE / "conformance.json").read_text(encoding="utf-8"))
RTOL = DOC["rtol"]
#: The fixture's columns, and the recorded component each one is read as.
COLUMNS = {"ux": "displacement_x", "uy": "displacement_y"}


def _nan(v: Optional[float]) -> float:
    return math.nan if v is None else float(v)


def _values(f: dict[str, Any]) -> np.ndarray:
    return np.array([[_nan(v) for v in row] for row in f["values"]], dtype="f8")


def _model(path: Path, fem_hash: str = FEM_HASH) -> Path:
    with h5py.File(path, "w") as f:
        f.create_group("meta/lineage").attrs["fem_hash"] = fem_hash
    return path


# ---- the conformance cases, written and read back ------------------------------


def _fixture_case(name: str, direction: int, reference: np.ndarray) -> SpectrumCase:
    """A fixture case as ``ON_NODES/DISPLACEMENT`` (``Ux``, ``Uy``) on nodes 1-3.

    The fixture holds what the evaluator reads (``mode``, ``lambda``,
    ``sa_period``, ``damping``, the increments); the rest is any valid value.
    """
    c = DOC["cases"][name]
    m = len(c["mode"])
    for f in c["increments"]:
        assert f["keys"] == [[1], [2], [3]] and f["components"] == ["ux", "uy"]
    data = np.stack([_values(f) for f in c["increments"]])          # (M, 3, 2)
    ids = np.array([1, 2, 3], dtype="i4")
    return SpectrumCase(
        name=name, direction=direction, scale=1.0,
        spectrum_periods=np.array([0.0, 1.0]), spectrum_sa=np.array([1.0, 1.0]),
        mode=np.array(c["mode"], dtype="i4"), kind=("eigen",) * m,
        lam=np.array(c["lambda"], dtype="f8"), sa_period=np.array(c["sa_period"], dtype="f8"),
        gamma=np.ones(m), sa=np.ones(m),
        damping=np.array([_nan(v) for v in c["damping"]], dtype="f8"),
        results=(CaseQuantity("DISPLACEMENT", "nodes", ("Ux", "Uy"), data, ids),),
        reference=(CaseQuantity("DISPLACEMENT", "nodes", ("Ux", "Uy"), reference[None], ids),),
    )


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> UserSpectra:
    folder = tmp_path_factory.mktemp("conformance")
    ex_reference = _values(DOC["states"]["reference|Ex|-1|0"])
    cases = [_fixture_case("Ex", 1, ex_reference), _fixture_case("Ey", 2, np.zeros((3, 2)))]
    model = _model(folder / "conf.h5")
    path = write_user_file(folder / "conf.user.h5", model, spectra=cases)
    return UserSpectra(path, model_h5=model)


class ReaderSource:
    """One recorded column: stages and modes from the fixture, spectra from the file."""

    linear = True

    def __init__(self, spectra: UserSpectra, column: str) -> None:
        self.spectra = spectra
        self.column = column
        self.quantity = ("nodes", COLUMNS[column])

    def stored(self, name: str) -> Optional[Field]:
        return None

    def field(self, address: StateAddress) -> Field:
        if address.kind in ("spectrum", "reference"):
            return self.spectra.field(address, self.quantity)
        key = f"{address.kind}|{address.source}|{address.step}|{address.mode}"
        f = DOC["states"].get(key) or DOC["other_states"][key]
        col = f["components"].index(self.column)
        return Field(node_keys([k[0] for k in f["keys"]]), (self.quantity[1],),
                     _values(f)[:, col:col + 1])

    def eigenvalue(self, address: StateAddress) -> float:
        return float(DOC["eigenvalues"][
            f"{address.kind}|{address.source}|{address.step}|{address.mode}"])

    def case(self, name: str) -> Case:
        return self.spectra.case(name, self.quantity)


def _table():
    def rows(doc_rows):
        return [Combination(
            c["name"], c["rule"],
            tuple(Term(t["kind"], t["source"], t["step"], t["mode"], t["factor"]) for t in c["terms"]),
            envelope=c["envelope"], fraction=c["fraction"], damping=c["damping"]) for c in doc_rows]
    return resolve(rows(DOC["combinations"]), rows(DOC["model_combinations"]))


def _reaches_a_written_case(table, name: str) -> bool:
    row = table.by_name(name)
    return any(t.kind in ("spectrum", "reference") and t.source in ("Ex", "Ey")
               or t.kind == "combination" and _reaches_a_written_case(table, t.source)
               for t in row.terms)


TABLE = _table()
SPECTRUM_GOLDENS = sorted(
    name for name, want in DOC["goldens"]["recorded"].items()
    if "refused" not in want and _reaches_a_written_case(TABLE, name))


def test_the_goldens_read_through_the_file_are_every_spectrum_golden() -> None:
    assert set(SPECTRUM_GOLDENS) >= {
        "Ex_cqc", "Ex_cqc_d02", "Ex_srss", "Ex_abs", "Ey_cqc", "dir", "plus", "minus",
        "seismic_max", "seismic_min", "k_cqc", "k_linear"}


@pytest.mark.parametrize("name", SPECTRUM_GOLDENS)
@pytest.mark.parametrize("column", sorted(COLUMNS))
def test_a_spectrum_golden_holds_through_the_written_file(written, name: str, column: str) -> None:
    want = DOC["goldens"]["recorded"][name]
    got = evaluate(TABLE, name, ReaderSource(written, column))
    assert got.signed is want["signed"]
    assert list(got.badges) == want["badges"]
    assert got.field.components == (COLUMNS[column],)
    assert [list(k) for k in got.field.keys.tolist()] == want["keys"]
    expected = _values(want)[:, want["components"].index(column)]
    np.testing.assert_allclose(got.field.values[:, 0], expected, rtol=RTOL, atol=0.0)


def _value(written: UserSpectra, name: str) -> np.ndarray:
    """``(3, 2)``: both columns of combination ``name``, read through the file."""
    return np.stack([evaluate(TABLE, name, ReaderSource(written, c)).field.values[:, 0]
                     for c in ("ux", "uy")], axis=1)


def _increments(written: UserSpectra, name: str) -> tuple[np.ndarray, Case]:
    """``(M, 3, 2)`` per-mode increments read back, and the case of the ``ux`` read."""
    cols = [written.case(name, ("nodes", COLUMNS[c])) for c in ("ux", "uy")]
    stack = np.stack([np.concatenate([cols[0].increments[i].values, cols[1].increments[i].values],
                                     axis=1) for i in range(len(cols[0].increments))])
    return stack, cols[0]


def test_whole_case_cqc_read_back_equals_combine_modes(written) -> None:
    r, ex = _increments(written, "Ex")
    want = combine_modes(r, rule="CQC", omega=np.sqrt(ex.lam), damping=ex.damping,
                         sa_periods=ex.sa_period)
    np.testing.assert_allclose(_value(written, "Ex_cqc"), want, rtol=RTOL)
    srss = 2.0 * combine_modes(r, rule="SRSS", sa_periods=ex.sa_period)
    np.testing.assert_allclose(_value(written, "Ex_srss"), srss, rtol=RTOL)


def test_directional_read_back_equals_combine_directions(written) -> None:
    per_direction = np.stack([_value(written, "Ex_cqc"), _value(written, "Ey_cqc")])
    want = combine_directions(per_direction, method="100-30")
    np.testing.assert_allclose(_value(written, "dir"), want, rtol=RTOL)


def test_the_seismic_envelope_read_back_equals_seismic_envelope(written) -> None:
    up, lo = seismic_envelope(_value(written, "ULS"), _value(written, "dir"))
    np.testing.assert_allclose(_value(written, "seismic_max"), up, rtol=RTOL)
    np.testing.assert_allclose(_value(written, "seismic_min"), lo, rtol=RTOL)


def test_the_case_read_back_is_the_one_written(written) -> None:
    ex = written.case("Ex", ("nodes", "displacement_y"))
    c = DOC["cases"]["Ex"]
    assert ex.mode.tolist() == c["mode"]
    assert ex.lam.tolist() == c["lambda"] and ex.sa_period.tolist() == c["sa_period"]
    for got, f in zip(ex.increments, c["increments"]):
        assert got.components == ("displacement_y",)
        np.testing.assert_array_equal(got.values[:, 0], _values(f)[:, 1])
    ref = written.field(StateAddress("reference", "Ex"), ("nodes", "displacement_x"))
    np.testing.assert_array_equal(ref.values[:, 0], _values(DOC["states"]["reference|Ex|-1|0"])[:, 0])


def test_a_case_with_lambda_zero_is_never_written(tmp_path: Path) -> None:
    """``bad_lambda`` stays a fixture-only golden: the writer refuses its case."""
    with pytest.raises(Exception, match="lambda"):
        write_user_file(tmp_path / "bad.user.h5", _model(tmp_path / "bad.h5"),
                        spectra=[_fixture_case("Ebad", 1, np.zeros((3, 2)))])


# ---- the mapping: (level, component) -> result and column ----------------------


@pytest.fixture
def helper_case(tmp_path: Path) -> UserSpectra:
    """``_user_helpers.case``: ``base_shear`` (``1``, ``2``) and DISPLACEMENT on nodes 4, 7."""
    model = _model(tmp_path / "m.h5")
    path = write_user_file(tmp_path / "m.user.h5", model, spectra=[case("E_x")])
    return UserSpectra(path, model_h5=model)


def _reason(fn, *args) -> str:
    with pytest.raises(CombinationRefused) as err:
        fn(*args)
    return err.value.reason


def test_a_nodal_component_reads_its_recorder_column(helper_case) -> None:
    written = case("E_x").result("DISPLACEMENT")
    for component, col in (("displacement_x", 0), ("displacement_y", 1)):
        got = helper_case.case("E_x", ("nodes", component))
        assert [f.components for f in got.increments] == [(component,)] * 2
        assert got.increments[0].keys["id"].tolist() == [4, 7]
        for i, f in enumerate(got.increments):
            np.testing.assert_array_equal(f.values[:, 0], written.data[i, :, col])
        step = helper_case.field(StateAddress("spectrum", "E_x", -1, 2), ("nodes", component))
        np.testing.assert_array_equal(step.values[:, 0], written.data[1, :, col])


def test_a_domain_label_reads_every_column_on_one_row(helper_case) -> None:
    shear = case("E_x").result("base_shear")
    got = helper_case.case("E_x", ("domain", "base_shear"))
    assert got.increments[0].components == ("1", "2")
    assert got.increments[0].keys["id"].tolist() == [0]
    for i, f in enumerate(got.increments):
        np.testing.assert_array_equal(f.values, shear.data[i])
    ref = helper_case.field(StateAddress("reference", "E_x"), ("domain", "base_shear"))
    np.testing.assert_array_equal(ref.values, [[0.0, 9.81]])


def test_a_quantity_the_case_does_not_hold_is_unresolved(helper_case) -> None:
    for quantity in (("nodes", "displacement_z"), ("nodes", "rotation_z"),
                     ("nodes", "Ux"), ("domain", "storey_shear"), ("domain", "1"),
                     ("elements", "axial_force"), ("gauss", "stress_xx"),
                     ("line_stations", "axial_force")):
        assert _reason(helper_case.case, "E_x", quantity) == "unresolved", quantity


def test_an_unknown_level_raises(helper_case) -> None:
    with pytest.raises(ValueError, match="level"):
        helper_case.case("E_x", ("regions", "displacement_x"))


def test_the_address_is_checked(helper_case) -> None:
    q = ("nodes", "displacement_x")
    assert _reason(helper_case.field, StateAddress("spectrum", "E_x", -1, 3), q) == "unresolved"
    assert _reason(helper_case.field, StateAddress("spectrum", "E_x", -1, 0), q) == "bad-term"
    assert _reason(helper_case.field, StateAddress("stage", "E_x"), q) == "bad-term"
    assert _reason(helper_case.case, "E_y", q) == "unresolved"


def test_a_refused_case_is_refused(tmp_path: Path) -> None:
    model = _model(tmp_path / "m.h5")
    path = write_user_file(tmp_path / "m.user.h5", model, spectra=[case("E_x"), case("E_y")])
    with h5py.File(path, "a") as f:
        del f["spectra/s1/modes/lambda"]
    provider = UserSpectra(path, model_h5=model)
    assert _reason(provider.case, "E_y", ("nodes", "displacement_x")) == "refused"
    assert provider.case("E_x", ("nodes", "displacement_x")).mode.tolist() == [1, 2]


def test_a_stale_zone_refuses_unless_asked(tmp_path: Path) -> None:
    """A6: ``/spectra`` is stale when the model hashes differ, or with no model."""
    path = write_user_file(tmp_path / "m.user.h5", _model(tmp_path / "m.h5"), spectra=[case()])
    other = _model(tmp_path / "other.h5", fem_hash="0" * 32)
    q = ("nodes", "displacement_x")
    provider = UserSpectra(path, model_h5=other)
    assert _reason(provider.case, "E_x (DBE)", q) == "stale"
    assert _reason(UserSpectra(path).case, "E_x (DBE)", q) == "stale"
    address = StateAddress("spectrum", "E_x (DBE)", -1, 1)
    assert _reason(provider.field, address, q) == "stale"
    assert provider.case("E_x (DBE)", q, allow_stale=True).mode.tolist() == [1, 2]
    assert provider.field(address, q, allow_stale=True).values.shape == (2, 1)
    # The verdict is the call's, not the provider's: asking once does not stick.
    assert _reason(provider.case, "E_x (DBE)", q) == "stale"


def test_no_user_file_is_unresolved() -> None:
    assert _reason(UserSpectra(None).case, "E_x", ("nodes", "displacement_x")) == "unresolved"


# ---- Results.combinations() passes the provider by default ----------------------


def _truss_set(folder: Path) -> Results:
    shutil.copy(LADRUNO / "truss2d.ladruno", folder / "run.ladruno")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Results.from_ladruno(folder / "run.ladruno")


SHEAR = np.array([[[120.0, -3.0]], [[-45.0, 8.0]], [[10.0, 2.5]]])  # (M, 1, ndm)


def _truss_case() -> SpectrumCase:
    """Three modes on the truss's nodes 1-3, two sharing a rigid ``sa_period``,
    and a base shear on the domain."""
    lam = np.array([40.0, 160.0, 900.0])
    data = np.array([[[0.0, 0.0], [0.4, 0.1], [1.0, 0.2]],
                     [[0.0, 0.0], [-0.7, 0.3], [0.5, -0.1]],
                     [[0.0, 0.0], [0.2, -0.05], [0.1, 0.3]]])
    ids = np.array([1, 2, 3], dtype="i4")
    return case(
        "Ex", mode=np.array([1, 2, 3], dtype="i4"), kind=("eigen",) * 3, lam=lam,
        sa_period=np.array([0.0, 0.2, 0.2]), gamma=np.ones(3), sa=np.ones(3),
        damping=np.array([0.05, 0.04, 0.03]),
        results=(CaseQuantity("base_shear", "domain", ("1", "2"), SHEAR),
                 CaseQuantity("DISPLACEMENT", "nodes", ("Ux", "Uy"), data, ids)),
        reference=(CaseQuantity("base_shear", "domain", ("1", "2"), np.array([[[0.0, 9.81]]])),
                   CaseQuantity("DISPLACEMENT", "nodes", ("Ux", "Uy"), np.zeros((1, 3, 2)), ids)))


def _truss_user(tmp_path: Path, model: Path) -> None:
    rows = [
        Combination("E", "cqc", (Term("spectrum", "Ex"),)),
        Combination("D+E", "linear", (Term("stage", "stage_0"), Term("combination", "E"))),
        Combination("k2", "linear", (Term("spectrum", "Ex", -1, 2, 3.0),)),
    ]
    write_user_file(tmp_path / "run.user.h5", model, results_file=tmp_path / "run.ladruno",
                    combinations=rows, spectra=[_truss_case()])


def test_results_combinations_reads_spectrum_terms_by_default(tmp_path: Path) -> None:
    run = _truss_set(tmp_path)
    model = _model(tmp_path / "run.h5")
    _truss_user(tmp_path, model)
    combos = run.combinations(model_h5=model)
    c = _truss_case()
    data = c.result("DISPLACEMENT").data
    for col, component in enumerate(("displacement_x", "displacement_y")):
        e = combos.evaluate("E", component)
        want = cqc_reference(data[:, :, col], c.lam, c.damping, c.sa_period)
        np.testing.assert_allclose(e.field.values[:, 0], want, rtol=1e-13)
        assert e.field.keys["id"].tolist() == [1, 2, 3] and not e.signed
        stage = run.stage("stage_0").nodes.get(component=component, time=-1).values[-1]
        both = combos.evaluate("D+E", component).field
        assert both.components == (component,)
        np.testing.assert_allclose(both.values[:, 0], stage + want, rtol=1e-13)
        k2 = combos.evaluate("k2", component)
        assert k2.signed
        np.testing.assert_array_equal(k2.field.values[:, 0], 3.0 * data[1, :, col])


def test_results_combinations_refuses_a_stale_spectra_zone(tmp_path: Path) -> None:
    run = _truss_set(tmp_path)
    _truss_user(tmp_path, _model(tmp_path / "run.h5"))
    other = _model(tmp_path / "other.h5", fem_hash="0" * 32)
    combos = run.combinations(tmp_path / "run.user.h5", model_h5=other)
    for args in (("E", "displacement_x"), ("E", "base_shear")):
        level = "domain" if args[1] == "base_shear" else "nodes"
        with pytest.raises(CombinationRefused) as err:
            combos.evaluate(*args, level=level)
        assert err.value.reason == "stale"
    fresh = run.combinations(tmp_path / "run.user.h5", model_h5=tmp_path / "run.h5")
    for component, level in (("displacement_x", "nodes"), ("base_shear", "domain")):
        got = combos.evaluate("E", component, level=level, allow_stale=True).field.values
        want = fresh.evaluate("E", component, level=level).field.values
        np.testing.assert_array_equal(got, want)
    with pytest.raises(CombinationRefused) as err:  # asking once does not stick
        combos.evaluate("E", "displacement_x")
    assert err.value.reason == "stale"


def test_results_combinations_reaches_a_domain_quantity(tmp_path: Path) -> None:
    """``base_shear`` through ``evaluate(level="domain")``: CQC, an increment, the reference."""
    run = _truss_set(tmp_path)
    model = _model(tmp_path / "run.h5")
    _truss_user(tmp_path, model)
    combos = run.combinations(model_h5=model)
    c = _truss_case()
    e = combos.evaluate("E", "base_shear", level="domain").field
    assert e.components == ("1", "2") and e.keys["id"].tolist() == [0]
    want = cqc_reference(SHEAR[:, 0, :], c.lam, c.damping, c.sa_period)
    np.testing.assert_allclose(e.values[0], want, rtol=1e-13)
    k2 = combos.evaluate("k2", "base_shear", level="domain").field
    np.testing.assert_array_equal(k2.values[0], 3.0 * SHEAR[1, 0])
    ref = combos.evaluate(StateAddress("reference", "Ex"), "base_shear", level="domain")
    np.testing.assert_array_equal(ref.field.values[0], [0.0, 9.81])


def test_a_stage_or_mode_term_of_a_domain_quantity_is_unresolved(tmp_path: Path) -> None:
    """``Results`` records no domain value: ``D+E`` cannot be read on the domain."""
    run = _truss_set(tmp_path)
    model = _model(tmp_path / "run.h5")
    _truss_user(tmp_path, model)
    combos = run.combinations(model_h5=model)
    for state in ("D+E", StateAddress("stage", "stage_0"), StateAddress("mode", "stage_0", -1, 1)):
        with pytest.raises(CombinationRefused) as err:
            combos.evaluate(state, "base_shear", level="domain")
        assert err.value.reason == "unresolved", state
        assert "domain" in err.value.detail


def test_no_provider_says_so() -> None:
    from apeGmsh.results.user._states import _RecordedSource
    source = _RecordedSource(None, "nodes", "displacement_x", None, None, "x")  # type: ignore[arg-type]
    for call in (lambda: source.case("Ex"),
                 lambda: source.field(StateAddress("spectrum", "Ex", -1, 1))):
        with pytest.raises(CombinationRefused) as err:
            call()
        assert err.value.reason == "unresolved"
        assert "no spectra provider was given" in err.value.detail
        assert "no reader yet" not in err.value.detail


def test_an_element_quantity_of_a_spectrum_term_is_unresolved(tmp_path: Path) -> None:
    run = _truss_set(tmp_path)
    model = _model(tmp_path / "run.h5")
    _truss_user(tmp_path, model)
    with pytest.raises(CombinationRefused) as err:
        run.combinations(model_h5=model).evaluate("E", "axial_force", level="line_stations")
    assert err.value.reason == "unresolved"
    assert "ON_NODES and ON_DOMAIN only" in err.value.detail
