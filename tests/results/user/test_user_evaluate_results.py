"""``Results.combinations()``: terms read from a real ``.ladruno`` set (ADR 0130 A3, A7).

The set is built from the committed ``truss2d.ladruno`` with h5py: stage ``D`` is
the recorded one, stage ``L`` its copy times ``-0.5``, and an ``eigen`` stage holds
two modal sets at ``STEP_3`` and ``STEP_9``, so a set's ``STEP`` id is never its
index. Each test's oracle is the value read back through the plain
``Results`` API and combined by hand, or ``combine_modes``.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._internal import _combinations_h5 as archive
from apeGmsh.opensees._internal.schema_version import OPENSEES_KEY
from apeGmsh.opensees.analysis.spectrum import combine_modes
from apeGmsh.results import Results
from apeGmsh.results.user import (
    Combination,
    StateAddress,
    Term,
    UserResult,
    write_user_file,
)
from apeGmsh.results.user._evaluate import SUPERPOSITION_BADGE, CombinationRefused
from tests.fixtures.schema import OPENSEES_CURRENT
from tests.results.user._user_helpers import LADRUNO

#: The mode shapes written, by (STEP id, 1-based mode): ``(Ux, Uy)`` at nodes 1-3.
SHAPES = {
    (3, 1): [[0.0, 0.0], [0.4, 0.1], [1.0, 0.2]],
    (3, 2): [[0.0, 0.0], [-0.7, 0.3], [0.5, -0.1]],
    (9, 1): [[0.0, 0.0], [0.6, -0.2], [0.9, 0.4]],
    (9, 2): [[0.0, 0.0], [0.2, 0.5], [-0.8, 0.3]],
}
LAMBDA = {(3, 1): 40.0, (3, 2): 160.0, (9, 1): 36.0, (9, 2): 150.0}


def _build_set(folder: Path, name: str = "run") -> Path:
    path = folder / f"{name}.ladruno"
    shutil.copy(LADRUNO / "truss2d.ladruno", path)
    with h5py.File(path, "a") as f:
        f.copy(f["MODEL_STAGE[1]"], "MODEL_STAGE[2]")
        for ds in ("RESULTS/ON_NODES/DISPLACEMENT/DATA",
                   "RESULTS/ON_ELEMENTS/basicForce/12-Truss[1:0:0]/DATA"):
            f["MODEL_STAGE[2]"][ds][...] = -0.5 * f["MODEL_STAGE[1]"][ds][...]
        f.copy(f["MODEL_STAGE[1]/MODEL"], "MODEL_STAGE[3]/MODEL")
        stage = f["MODEL_STAGE[3]"]
        for key in ("STEP", "TIME"):
            stage.attrs[key] = f["MODEL_STAGE[1]"].attrs[key]
        stage.attrs["KIND"] = np.bytes_("eigen")
        on_nodes = stage.create_group("RESULTS/ON_NODES")
        modal = on_nodes.create_group("MODES_OF_VIBRATION(U)")
        modal.attrs["COMPONENTS"] = np.bytes_("Ux,Uy")
        modal.create_dataset("ID", data=np.array([[1], [2], [3]], dtype="i4"))
        for (step, mode), shape in SHAPES.items():
            block = modal.create_dataset(f"DATA/STEP_{step}/MODE_{mode - 1}", data=np.array(shape))
            block.attrs["LAMBDA"] = LAMBDA[(step, mode)]
    return path


@pytest.fixture
def run(tmp_path: Path) -> Results:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Results.from_ladruno(_build_set(tmp_path))


def _model(path: Path, rows: tuple[archive.Combination, ...] = ()) -> Path:
    """A model archive with lineage, and ``/opensees/combinations`` when given."""
    with h5py.File(path, "w") as f:
        f.create_group("meta/lineage").attrs["fem_hash"] = "fedcba9876543210fedcba9876543210"
        f["meta"].attrs[OPENSEES_KEY] = OPENSEES_CURRENT
        archive.write_combinations_into(f, rows)
    return path


def _user(tmp_path: Path, rows: list[Combination], results=()) -> Path:
    return write_user_file(
        tmp_path / "run.user.h5", _model(tmp_path / "model.h5"),
        results_file=tmp_path / "run.ladruno", combinations=rows, results=results)


def _stage(source: str, factor: float = 1.0, step: int = -1) -> Term:
    return Term("stage", source, step, 0, factor)


def _read(run: Results, stage: str, component: str = "displacement_x", **kw) -> np.ndarray:
    return run.stage(stage).nodes.get(component=component, time=-1, **kw).values[-1]


# ---- stages ------------------------------------------------------------------


def test_a_linear_combination_is_the_factored_sum_of_the_stage_reads(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("ULS", "linear", (_stage("stage_0", 1.2), _stage("stage_1", 1.6)))])
    combos = run.combinations()
    got = combos.evaluate("ULS", "displacement_x")
    want = 1.2 * _read(run, "stage_0") + 1.6 * _read(run, "stage_1")
    np.testing.assert_allclose(got.field.values[:, 0], want, rtol=1e-15)
    assert got.field.keys["id"].tolist() == [1, 2, 3]
    assert got.signed and got.badges == (SUPERPOSITION_BADGE,)


def test_a_combined_force_comes_from_the_per_term_forces(run, tmp_path: Path) -> None:
    """ADR 0124 D6: the axial force of ULS is the factored sum of the stages' forces."""
    _user(tmp_path, [Combination("ULS", "linear", (_stage("stage_0", 1.2), _stage("stage_1", 1.6)))])
    got = run.combinations().evaluate("ULS", "axial_force", level="line_stations")
    per = [run.stage(s).elements.line_stations.get(component="axial_force", time=-1).values[-1]
           for s in ("stage_0", "stage_1")]
    np.testing.assert_allclose(got.field.values[:, 0], 1.2 * per[0] + 1.6 * per[1], rtol=1e-15)
    assert got.field.keys["id"].tolist() == [1, 2]


def test_a_stage_step_is_a_time_index(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("first", "linear", (_stage("stage_0", 1.0, step=0),)),
                     Combination("beyond", "linear", (_stage("stage_0", 1.0, step=4),))])
    combos = run.combinations()
    want = run.stage("stage_0").nodes.get(component="displacement_x", time=0).values[0]
    np.testing.assert_array_equal(combos.evaluate("first", "displacement_x").field.values[:, 0], want)
    with pytest.raises(CombinationRefused) as err:
        combos.evaluate("beyond", "displacement_x")
    assert err.value.reason == "unresolved"


def test_a_stage_resolves_by_name_too(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("by_name", "linear", (_stage("MODEL_STAGE[2]", 2.0),))])
    got = run.combinations().evaluate("by_name", "displacement_x").field.values[:, 0]
    np.testing.assert_allclose(got, 2.0 * _read(run, "stage_1"), rtol=1e-15)


# ---- modes: STEP ids, never indices --------------------------------------------


def _mode(step: int, mode: int, factor: float = 1.0) -> Term:
    return Term("mode", "stage_2", step, mode, factor)


@pytest.mark.parametrize("step, mode", sorted(SHAPES))
def test_a_mode_term_names_its_set_by_step_id(run, tmp_path: Path, step: int, mode: int) -> None:
    """The STEP-id-for-index mutant reads STEP_9 for step 1, or nothing for step 3."""
    _user(tmp_path, [Combination("m", "linear", (_mode(step, mode, 2.0),))])
    got = run.combinations().evaluate("m", "displacement_x").field.values[:, 0]
    np.testing.assert_allclose(got, 2.0 * np.array(SHAPES[(step, mode)])[:, 0], rtol=1e-15)


def test_a_host_with_two_sets_needs_a_step_id(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("only", "linear", (_mode(-1, 1),)),
                     Combination("index", "linear", (_mode(1, 1),))])
    combos = run.combinations()
    for name in ("only", "index"):
        with pytest.raises(CombinationRefused) as err:
            combos.evaluate(name, "displacement_x")
        assert err.value.reason == "unresolved"


def test_cqc_over_the_modes_of_one_set_equals_combine_modes(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("cqc9", "cqc", (_mode(9, 1), _mode(9, 2)), damping=0.05)])
    got = run.combinations().evaluate("cqc9", "displacement_x").field.values[:, 0]
    per_mode = np.array([SHAPES[(9, 1)], SHAPES[(9, 2)]])[:, :, 0]
    want = combine_modes(per_mode, rule="CQC", omega=np.sqrt([36.0, 150.0]), damping=0.05)
    np.testing.assert_allclose(got, want, rtol=1e-12)


# ---- what is refused -------------------------------------------------------------


def test_a_state_handle_of_another_results_is_refused(run, tmp_path: Path) -> None:
    other_dir = tmp_path / "variant"
    other_dir.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = Results.from_ladruno(_build_set(other_dir))
    combos = run.combinations()
    mine = combos.evaluate(run.stage("stage_0"), "displacement_x").field.values[:, 0]
    np.testing.assert_array_equal(mine, _read(run, "stage_0"))
    with pytest.raises(CombinationRefused) as err:
        combos.evaluate(other.stage("stage_0"), "displacement_x")
    assert err.value.reason == "foreign-handle"


def test_a_combination_handle_of_another_set_is_refused(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("ULS", "linear", (_stage("stage_0"),))])
    first, second = run.combinations(), run.combinations()
    assert first.evaluate(first.state("ULS"), "displacement_x").signed
    with pytest.raises(CombinationRefused) as err:
        first.evaluate(second.state("ULS"), "displacement_x")
    assert err.value.reason == "foreign-handle"


def test_a_spectrum_term_without_a_spectra_reader_is_unresolved(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("E", "cqc", (Term("spectrum", "Ex"),))])
    with pytest.raises(CombinationRefused) as err:
        run.combinations().evaluate("E", "displacement_x")
    assert err.value.reason == "unresolved"


def test_model_h5_combinations_evaluate_without_a_user_file(run, tmp_path: Path) -> None:
    model = _model(tmp_path / "model.h5", (
        archive.Combination("ULS", "linear", (archive.CombinationTerm("stage", "stage_0", 3.0),)),))
    combos = run.combinations(model_h5=model)
    assert combos.names == ("ULS",)
    got = combos.evaluate("ULS", "displacement_x").field.values[:, 0]
    np.testing.assert_allclose(got, 3.0 * _read(run, "stage_0"), rtol=1e-15)


# ---- a user result of the set (A5, A6) ------------------------------------------


def _wood(basis: str = "model", linear: bool = False) -> UserResult:
    data = np.array([[[1.0], [2.0], [3.0]], [[10.0], [-20.0], [30.0]]])
    return UserResult(
        name="wood_armer_top", target="nodes", basis=basis, linear=linear,
        states=(StateAddress("stage", "stage_0"), StateAddress("stage", "stage_1")),
        components=("w",), data=data, ids=np.array([1, 2, 3], dtype="i4"))


def test_a_user_result_reads_its_stored_rows(run, tmp_path: Path) -> None:
    rows = [Combination("ULS", "linear", (_stage("stage_0", 1.2), _stage("stage_1", 1.6))),
            Combination("env", "envelope", (_stage("stage_0"), _stage("stage_1")), envelope="absmax")]
    _user(tmp_path, rows, results=[_wood()])
    # @basis=model goes stale without a model to compare with: none is drawn on no proof.
    with pytest.raises(CombinationRefused) as err:
        run.combinations().evaluate("env", user_result="wood_armer_top")
    assert err.value.reason == "stale"
    combos = run.combinations(model_h5=tmp_path / "model.h5")
    with pytest.raises(CombinationRefused) as err:
        combos.evaluate("ULS", user_result="wood_armer_top")
    assert err.value.reason == "not-linear"
    got = combos.evaluate("env", user_result="wood_armer_top").field.values[:, 0]
    np.testing.assert_array_equal(got, [10.0, 20.0, 30.0])


def test_a_stale_user_result_is_refused_unless_asked(run, tmp_path: Path) -> None:
    rows = [Combination("env", "envelope", (_stage("stage_0"), _stage("stage_1")), envelope="max")]
    _user(tmp_path, rows, results=[_wood(basis="results")])
    combos = run.combinations()
    np.testing.assert_array_equal(
        combos.evaluate("env", user_result="wood_armer_top").field.values[:, 0], [10.0, 2.0, 30.0])
    run.close()
    with h5py.File(tmp_path / "run.ladruno", "a") as f:  # a re-run moves the sampled slab
        f["MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA"][-1, 0, 0] += 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rerun = Results.from_ladruno(tmp_path / "run.ladruno")
    combos = rerun.combinations()
    with pytest.raises(CombinationRefused) as err:
        combos.evaluate("env", user_result="wood_armer_top")
    assert err.value.reason == "stale"
    assert combos.evaluate("env", user_result="wood_armer_top", allow_stale=True).field.values[0, 0] == 10.0


def test_a_state_address_alone_is_read_and_a_bad_one_refused(run) -> None:
    combos = run.combinations()
    got = combos.evaluate(StateAddress("mode", "stage_2", 9, 2), "displacement_x")
    np.testing.assert_array_equal(got.field.values[:, 0], np.array(SHAPES[(9, 2)])[:, 0])
    assert got.signed
    with pytest.raises(CombinationRefused) as err:
        combos.evaluate(StateAddress("mode", "stage_2", 9, 0), "displacement_x")
    assert err.value.reason == "bad-term"
