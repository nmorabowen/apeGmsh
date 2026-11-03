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
        tmp_path / "run.user.h5", _model(tmp_path / "run.h5"),
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


@pytest.mark.parametrize("step", [-1, 0])
def test_a_stage_with_no_steps_refuses_with_a_reason(run, tmp_path: Path, step: int) -> None:
    """#2750: the eigen stage records no steps (San Ramón's static stage is one), so
    even its last step is no state: the term refuses, never an ``IndexError``."""
    assert run.stage("stage_2").n_steps == 0
    _user(tmp_path, [Combination("over_eigen", "linear", (_stage("stage_2", 1.0, step=step),
                                                           _stage("stage_0", 1.0)))])
    with pytest.raises(CombinationRefused) as err:
        run.combinations().evaluate("over_eigen", "displacement_x")
    assert err.value.reason == "unresolved"
    assert "stage 'stage_2' records no steps" in err.value.detail


def test_an_unstamped_stage_is_named_stage_i_only(run, tmp_path: Path) -> None:
    """A3: with no stamp the source is ``stage_<i>``; the recorder's group name is not one."""
    _user(tmp_path, [Combination("by_group", "linear", (_stage("MODEL_STAGE[2]", 2.0),)),
                     Combination("by_id", "linear", (_stage("stage_1", 2.0),))])
    combos = run.combinations()
    got = combos.evaluate("by_id", "displacement_x").field.values[:, 0]
    np.testing.assert_allclose(got, 2.0 * _read(run, "stage_1"), rtol=1e-15)
    with pytest.raises(CombinationRefused) as err:
        combos.evaluate("by_group", "displacement_x")
    assert err.value.reason == "unresolved"


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


def test_a_spectrum_term_over_a_case_the_set_lacks_is_unresolved(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("E", "cqc", (Term("spectrum", "Ex"),))])
    with pytest.raises(CombinationRefused) as err:
        run.combinations().evaluate("E", "displacement_x")
    assert err.value.reason == "unresolved"
    assert "no spectrum case named 'Ex'" in err.value.detail


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
    combos = run.combinations(model_h5=tmp_path / "run.h5")
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


# ---- review of #2711 ------------------------------------------------------------


def _stamp(run: Results) -> None:
    """Stamp the run's stage names, as model_h5= does from /opensees/stages."""
    run._reader.attach_stage_names(["gravity", "live", "modal"], ["static", "static", "static"])
    run._stages_cache = None


def test_one_state_alone_has_no_rule(run, tmp_path: Path) -> None:
    """A lone state of a non-linear result reads as stored, and is not badged (finding 1)."""
    _user(tmp_path, [Combination("ULS", "linear", (_stage("stage_0", 1.2),))],
          results=[_wood(basis="model", linear=False)])
    combos = run.combinations(model_h5=tmp_path / "run.h5")
    for state in (StateAddress("stage", "stage_0"), run.stage("stage_0")):
        got = combos.evaluate(state, user_result="wood_armer_top")
        np.testing.assert_array_equal(got.field.values[:, 0], [1.0, 2.0, 3.0])
        assert got.signed and got.badges == ()
    assert combos.evaluate(run.stage("stage_0"), "displacement_x").badges == ()
    with pytest.raises(CombinationRefused) as err:  # a declared linear keeps the check
        combos.evaluate("ULS", user_result="wood_armer_top")
    assert err.value.reason == "not-linear"


def test_a_stamped_stage_is_named_by_its_stamp(run, tmp_path: Path) -> None:
    """A3: the stamped name, else stage_<i>, on the term and the STATES side (finding 2)."""
    _stamp(run)
    rows = [Combination("ULS", "linear", (_stage("gravity", 1.2), _stage("live", 1.6))),
            Combination("by_id", "linear", (_stage("stage_0", 1.2),)),
            Combination("m", "linear", (Term("mode", "modal", 9, 2),))]
    stored = UserResult(
        name="mx", target="nodes", basis="model", linear=True,
        states=(StateAddress("stage", "gravity"), StateAddress("stage", "live")),
        components=("w",), data=np.array([[[1.0], [2.0], [3.0]], [[10.0], [-20.0], [30.0]]]),
        ids=np.array([1, 2, 3], dtype="i4"))
    _user(tmp_path, rows, results=[stored])
    combos = run.combinations(model_h5=tmp_path / "run.h5")
    got = combos.evaluate("ULS", user_result="mx").field.values[:, 0]
    np.testing.assert_allclose(got, [1.2 + 16.0, 2.4 - 32.0, 3.6 + 48.0], rtol=1e-15)
    for quantity in ({"component": "displacement_x"}, {"user_result": "mx"}):
        with pytest.raises(CombinationRefused) as err:
            combos.evaluate("by_id", **quantity)
        assert err.value.reason == "unresolved" and "stamped" in err.value.detail
    handle = combos.evaluate(run.stage("gravity"), user_result="mx").field.values[:, 0]
    np.testing.assert_array_equal(handle, [1.0, 2.0, 3.0])
    shape = combos.evaluate("m", "displacement_x").field.values[:, 0]
    np.testing.assert_array_equal(shape, np.array(SHAPES[(9, 2)])[:, 0])
    mode_handle = next(s for s in run.stages if s.id == "stage_2_step_9_mode_2")
    got = combos.evaluate(run.stage(mode_handle.id), "displacement_x").field.values[:, 0]
    np.testing.assert_array_equal(got, shape)


@pytest.mark.parametrize("results_name, user_name", [
    ("yoo_slab_N56_75.results.h5", "yoo_slab_N56_75.user.h5"),
    ("truss2d.part-0.ladruno", "truss2d.user.h5"),
    ("run.ladruno", "run.user.h5"),
])
def test_the_user_file_is_found_by_the_set_stem(tmp_path: Path, results_name: str, user_name: str) -> None:
    """Finding 4: the stem drops .results and .part-<N>."""
    from apeGmsh.results.user._states import user_file_beside

    (tmp_path / user_name).write_bytes(b"")
    assert user_file_beside(tmp_path / results_name, None) == tmp_path / user_name
    assert user_file_beside(tmp_path / "elsewhere" / results_name, None) is None


def test_the_model_stem_names_the_user_file(tmp_path: Path) -> None:
    from apeGmsh.results.user._states import user_file_beside

    (tmp_path / "frame.user.h5").write_bytes(b"")
    (tmp_path / "run.user.h5").write_bytes(b"")
    assert user_file_beside(tmp_path / "run.ladruno", tmp_path / "frame.h5") == tmp_path / "frame.user.h5"


def test_a_partitioned_set_finds_its_user_file(tmp_path: Path) -> None:
    for k in (0, 1):
        shutil.copy(LADRUNO / f"truss2d.part-{k}.ladruno", tmp_path / f"truss2d.part-{k}.ladruno")
    write_user_file(tmp_path / "truss2d.user.h5", _model(tmp_path / "truss2d.h5"),
                    combinations=[Combination("ULS", "linear", (_stage("stage_0", 1.2),))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = Results.from_ladruno(tmp_path / "truss2d.part-0.ladruno")
    assert run.combinations().names == ("ULS",)
    explicit = run.combinations(tmp_path / "truss2d.user.h5")
    assert explicit.user_h5 == tmp_path / "truss2d.user.h5"


def test_a_deep_chain_and_a_wide_diamond_evaluate(run) -> None:
    """Finding 5: no recursion, and each combination is evaluated once."""
    import time

    from apeGmsh.results.user._descriptor import resolve
    from apeGmsh.results.user._evaluate import evaluate, is_signed

    from tests.results.user.test_user_conformance import FixtureSource

    chain = [Combination("c0", "linear", (Term("stage", "D"),))]
    chain += [Combination(f"c{i}", "linear", (Term("combination", f"c{i - 1}"),))
              for i in range(1, 1000)]
    t0 = time.perf_counter()
    got = evaluate(resolve(chain), "c999", FixtureSource()).field.values
    np.testing.assert_array_equal(got[0], [1.0, 2.0])
    diamond = [Combination("d0", "linear", (Term("stage", "D"),))]
    diamond += [Combination(f"d{i}", "linear", (Term("combination", f"d{i - 1}"),
                                                Term("combination", f"d{i - 1}", factor=-1.0)))
                for i in range(1, 22)]
    table = resolve(diamond)
    assert is_signed(table, "d21")
    assert evaluate(table, "d21", FixtureSource()).field.values[0, 0] == 0.0
    assert time.perf_counter() - t0 < 1.0


# ---- delta review of #2711 --------------------------------------------------------


@pytest.mark.parametrize("model_at", ["model.h5", "results/from_fem/0123abcd.model.h5"])
def test_a_model_of_another_stem_or_folder_does_not_hide_the_user_file(
        run, tmp_path: Path, model_at: str) -> None:
    """Delta finding 1: <model stem>.user.h5 only if it is beside the results file."""
    model = tmp_path / model_at
    model.parent.mkdir(parents=True, exist_ok=True)
    write_user_file(tmp_path / "run.user.h5", _model(model), results_file=tmp_path / "run.ladruno",
                    combinations=[Combination("ULS", "linear", (_stage("stage_0", 1.2),))])
    combos = run.combinations(model_h5=model)
    assert combos.user_h5 == tmp_path / "run.user.h5"
    assert combos.names == ("ULS",)


def _program(path: Path, *kinds: str) -> Path:
    """A bridge model.h5 of the truss whose stages are gravity, live and modal."""
    from tests.test_results_ladruno_stage_binding import _model_h5

    return _model_h5(path, *zip(("gravity", "live", "modal"), kinds))


def _program_set(tmp_path: Path, *kinds: str) -> tuple[Path, Path]:
    path = _build_set(tmp_path)
    model = _program(tmp_path / "run.h5", *kinds)
    write_user_file(tmp_path / "run.user.h5", model, results_file=path, combinations=[
        Combination("ULS", "linear", (_stage("gravity", 1.2), _stage("live", 1.6))),
        Combination("ULS_ids", "linear", (_stage("stage_0", 1.2), _stage("stage_1", 1.6))),
        Combination("m", "linear", (Term("mode", "stage_2", 9, 2),))])
    return path, model


def test_stage_terms_refuse_until_the_model_names_are_bound(tmp_path: Path) -> None:
    """Delta finding 2: a read never renames the caller's stages, and never silently differs.

    Opened bare beside a model that names its stages, every stage and mode term
    refuses and says to open with ``model_h5=``; opened with it, the names bind.
    """
    path, model = _program_set(tmp_path, "static", "static", "static")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = Results.from_ladruno(path)
        stamped = Results.from_ladruno(path, model_h5=model)
    combos = bare.combinations(model_h5=model)
    for state in ("ULS", "ULS_ids", "m", bare.stage("stage_0")):
        with pytest.raises(CombinationRefused) as err:
            combos.evaluate(state, "displacement_x")
        assert err.value.reason == "unresolved" and "model_h5=" in err.value.detail
    assert [s.name for s in bare.stages][:2] == ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]  # untouched

    combos = stamped.combinations()
    got = combos.evaluate("ULS", "displacement_x").field.values[:, 0]
    want = 1.2 * _read(stamped, "gravity") + 1.6 * _read(stamped, "live")
    np.testing.assert_allclose(got, want, rtol=1e-15)
    with pytest.raises(CombinationRefused) as err:
        combos.evaluate("ULS_ids", "displacement_x")
    assert "stamped" in err.value.detail


def test_from_fem_refuses_like_a_bare_open(tmp_path: Path, monkeypatch) -> None:
    """A from_fem Results binds a neutral cache archive that names no stage, so the
    program model's names are unbound there too: refused, never read the other way."""
    path, model = _program_set(tmp_path, "static", "static", "static")
    monkeypatch.chdir(tmp_path)  # from_fem caches under <cwd>/results/from_fem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = Results.from_ladruno(path)
        ff = Results.from_fem(bare.fem, path)
    assert ff._model_path is not None  # the neutral cache archive
    combos = ff.combinations(model_h5=model)
    for state in ("ULS", "ULS_ids", "m"):
        with pytest.raises(CombinationRefused) as err:
            combos.evaluate(state, "displacement_x")
        assert err.value.reason == "unresolved" and "model_h5=" in err.value.detail
    # Its own cache archive declares no stages, so on its own stage_<i> stands.
    combos = ff.combinations()
    got = combos.evaluate("ULS_ids", "displacement_x").field.values[:, 0]
    want = 1.2 * _read(ff, "stage_0") + 1.6 * _read(ff, "stage_1")
    np.testing.assert_allclose(got, want, rtol=1e-15)


def test_without_names_from_either_source_stage_i_stays(run, tmp_path: Path) -> None:
    _user(tmp_path, [Combination("ULS_ids", "linear", (_stage("stage_0", 2.0),))])
    combos = run.combinations(model_h5=tmp_path / "run.h5")  # declares no stages
    got = combos.evaluate("ULS_ids", "displacement_x").field.values[:, 0]
    np.testing.assert_allclose(got, 2.0 * _read(run, "stage_0"), rtol=1e-15)


def test_a_reader_that_declines_the_names_keeps_stage_i(tmp_path: Path) -> None:
    """The .ladruno records KIND static; a program calling live transient is misaligned."""
    from apeGmsh.results.readers._ladruno import StageKindMismatchWarning

    path, model = _program_set(tmp_path, "static", "transient", "static")
    with pytest.warns(StageKindMismatchWarning):
        run = Results.from_ladruno(path, model_h5=model)
    combos = run.combinations()
    want = 1.2 * _read(run, "stage_0") + 1.6 * _read(run, "stage_1")
    np.testing.assert_allclose(
        combos.evaluate("ULS_ids", "displacement_x").field.values[:, 0], want, rtol=1e-15)
    with pytest.raises(CombinationRefused) as err:
        combos.evaluate("ULS", "displacement_x")
    assert err.value.reason == "unresolved"


def test_a_ten_thousand_deep_chain_evaluates_in_about_the_judges_time() -> None:
    """Delta finding 4: by_name is a lookup, so evaluation is linear in the chain."""
    import time

    from apeGmsh.results.user._descriptor import resolve
    from apeGmsh.results.user._evaluate import evaluate

    from tests.results.user.test_user_conformance import FixtureSource

    rows = [Combination("c0", "linear", (Term("stage", "D"),))]
    rows += [Combination(f"c{i}", "linear", (Term("combination", f"c{i - 1}"),))
             for i in range(1, 10_000)]
    t0 = time.perf_counter()
    table = resolve(rows)
    judge = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = evaluate(table, "c9999", FixtureSource()).field.values
    spent = time.perf_counter() - t0
    np.testing.assert_array_equal(got[0], [1.0, 2.0])
    assert spent < max(2.0, 20 * judge), (spent, judge)


def _named(path: Path, *names: str) -> Path:
    from tests.test_results_ladruno_stage_binding import _model_h5

    return _model_h5(path, *((n, "static") for n in names))


@pytest.mark.parametrize("other", [("live", "gravity", "modal"), ("gravity", "wind", "snow")])
def test_another_models_stage_names_are_refused(tmp_path: Path, other: tuple[str, ...]) -> None:
    """Delta finding (p12): one shared name must not lift the refusal.

    Opened with A = [gravity, live, modal], a model B that orders the names
    differently, or shares only one, would read B's gravity as A's stage_0.
    """
    path, model = _program_set(tmp_path, "static", "static", "static")
    (tmp_path / "b").mkdir()
    b = _named(tmp_path / "b" / "b.h5", *other)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = Results.from_ladruno(path, model_h5=model)
    combos = run.combinations(tmp_path / "run.user.h5", model_h5=b)
    for state in ("ULS", "ULS_ids", "m"):
        with pytest.raises(CombinationRefused) as err:
            combos.evaluate(state, "displacement_x")
        assert err.value.reason == "unresolved"
        assert "bound to another model's stage names" in err.value.detail
        assert "model_h5=" in err.value.detail
    exact = run.combinations(model_h5=model)
    want = 1.2 * _read(run, "gravity") + 1.6 * _read(run, "live")
    np.testing.assert_allclose(exact.evaluate("ULS", "displacement_x").field.values[:, 0], want,
                               rtol=1e-15)


def test_a_partial_run_is_its_models_binding(tmp_path: Path) -> None:
    """Four stages declared, three ran: the three stamped names are a prefix of the four."""
    path = _build_set(tmp_path)
    model = _named(tmp_path / "run.h5", "gravity", "live", "modal", "wind")
    write_user_file(tmp_path / "run.user.h5", model, results_file=path, combinations=[
        Combination("ULS", "linear", (_stage("gravity", 1.2), _stage("live", 1.6)))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = Results.from_ladruno(path, model_h5=model)
    assert [s.name for s in run.stages if s.kind != "mode"] == ["gravity", "live", "modal"]
    got = run.combinations().evaluate("ULS", "displacement_x").field.values[:, 0]
    want = 1.2 * _read(run, "gravity") + 1.6 * _read(run, "live")
    np.testing.assert_allclose(got, want, rtol=1e-15)
