"""A native ``results.h5`` design field is not linear (ADR 0130 A5, owner ruling VU2c).

The YOO slab's native file stores ``wood_armer_*`` beside ``displacement_z`` as
an ordinary component, and a summing combination summed it. Here a native file
written with ``NativeWriter`` holds two stages, ``D`` and ``L``, of a recorder
spelling (``displacement_z``) and a design field (``wood_armer_pos_x``, values
that are not linear in the stage's load). The oracle for every value is the
plain ``Results`` read of each stage, combined by hand.

A derived scalar (von Mises, a principal value) is not linear on any source:
the last tests pin that on a native Gauss file and on ``quad2d.ladruno``.
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
from apeGmsh.results import Results
from apeGmsh.results.user import Combination, StateAddress, Term, write_user_file
from apeGmsh.results.user._evaluate import CombinationRefused
from apeGmsh.results.user._states import recorded_is_linear
from apeGmsh.results.writers import NativeWriter
from tests.conftest import _open_model_from_h5
from tests.fixtures.schema import OPENSEES_CURRENT
from tests.results.user._user_helpers import LADRUNO

NODES = np.array([1, 2, 3], dtype=np.int64)
#: Per stage: (displacement_z, wood_armer_pos_x) at nodes 1-3.
STAGES = {
    "D": ([-1.0, -2.0, -3.0], [0.0, 4.0, 9.0]),
    "L": ([-0.5, -1.5, -2.5], [1.0, 0.0, 2.0]),
}
ULS = Combination("ULS", "linear", (Term("stage", "D", -1, 0, 1.2), Term("stage", "L", -1, 0, 1.6)))
ENV = Combination("ENV", "envelope", (Term("stage", "D", -1, 0, 1.0), Term("stage", "L", -1, 0, 1.0)),
                  envelope="max")


def _native(path: Path) -> Path:
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        for name, (uz, wa) in STAGES.items():
            sid = w.begin_stage(name=name, kind="static", time=np.array([0.0]))
            w.write_nodes(sid, "partition_0", node_ids=NODES, components={
                "displacement_z": np.array([uz]), "wood_armer_pos_x": np.array([wa])})
            w.end_stage()
    return path


def _model(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        f.create_group("meta/lineage").attrs["fem_hash"] = "fedcba9876543210fedcba9876543210"
        f["meta"].attrs[OPENSEES_KEY] = OPENSEES_CURRENT
        archive.write_combinations_into(f, ())
    return path


@pytest.fixture
def native_set(tmp_path: Path) -> Path:
    results = _native(tmp_path / "run.results.h5")
    write_user_file(tmp_path / "run.user.h5", _model(tmp_path / "run.model.h5"),
                    results_file=results, combinations=[ULS, ENV])
    return results


def _open(path: Path) -> Results:
    return Results.from_native(path, model=_open_model_from_h5(path))


def _read(run: Results, stage: str, component: str) -> np.ndarray:
    return np.asarray(run.stage(stage).nodes.get(component=component, time=-1).values[-1])


def test_a_recorder_spelling_still_sums(native_set: Path) -> None:
    run = _open(native_set)
    got = run.combinations().evaluate("ULS", "displacement_z")
    want = 1.2 * _read(run, "D", "displacement_z") + 1.6 * _read(run, "L", "displacement_z")
    np.testing.assert_allclose(got.field.values[:, 0], want, rtol=0, atol=0)


def test_a_design_field_refuses_a_summing_rule(native_set: Path) -> None:
    with pytest.raises(CombinationRefused) as exc:
        _open(native_set).combinations().evaluate("ULS", "wood_armer_pos_x")
    assert exc.value.reason == "not-linear"
    assert "recorder spelling" in str(exc.value)


def test_a_design_field_still_envelopes_and_reads_alone(native_set: Path) -> None:
    run = _open(native_set)
    combos = run.combinations()
    env = combos.evaluate("ENV", "wood_armer_pos_x")
    want = np.maximum(_read(run, "D", "wood_armer_pos_x"), _read(run, "L", "wood_armer_pos_x"))
    np.testing.assert_array_equal(env.field.values[:, 0], want)
    lone = combos.evaluate(StateAddress("stage", "D", -1, 0), "wood_armer_pos_x")
    np.testing.assert_array_equal(lone.field.values[:, 0], _read(run, "D", "wood_armer_pos_x"))
    assert lone.badges == ()


@pytest.mark.parametrize("component", ["", "no_such_component", "Wood_Armer_Pos_X"])
def test_an_unknown_or_empty_component_refuses(native_set: Path, component: str) -> None:
    with pytest.raises(CombinationRefused) as exc:
        _open(native_set).combinations().evaluate("ULS", component)
    assert exc.value.reason == "not-linear"


def test_every_way_of_opening_the_file_judges_it_alike(native_set: Path, tmp_path: Path) -> None:
    """Reload: the file itself names the verdict, not the door it was opened through."""
    sidecar = tmp_path / "sidecar.model.h5"
    shutil.copy(native_set, sidecar)
    doors = [
        _open(native_set),
        Results.from_native(native_set, model=_open_model_from_h5(sidecar), model_path=sidecar),
        _open(native_set).stage("D"),
    ]
    user = tmp_path / "run.user.h5"
    for run in doors:
        combos = run.combinations(user)
        with pytest.raises(CombinationRefused, match="recorder spelling"):
            combos.evaluate("ULS", "wood_armer_pos_x")
        np.testing.assert_allclose(
            combos.evaluate("ULS", "displacement_z").field.values[:, 0],
            1.2 * np.array(STAGES["D"][0]) + 1.6 * np.array(STAGES["L"][0]), rtol=1e-15)


@pytest.mark.parametrize(("results_file", "level", "component", "linear"), [
    ("run.results.h5", "nodes", "displacement_z", True),
    ("run.results.h5", "gauss", "bending_moment_xx", True),
    ("run.results.h5", "gauss", "wood_armer_pos_x", False),
    ("run.results.h5", "gauss", "rib_shear_x", False),
    ("run.results.h5", "nodes", "deflection_dc_L480", False),
    ("run.results.h5", "nodes", "", False),
    ("model.h5", "nodes", "state_variable_3", True),
    (None, "nodes", "wood_armer_pos_x", False),  # no file: judged by name
    ("run.ladruno", "gauss", "wood_armer_pos_x", True),  # .ladruno unchanged
    ("run.part-0.ladruno", "nodes", "", True),
    ("run.part-1.MPCO", "gauss", "anything", True),
    ("run.results.h5", "domain", "base_shear", True),  # /spectra only
    # A derived scalar is not linear on any source.
    ("run.ladruno", "gauss", "von_mises_stress", False),
    ("run.ladruno", "gauss", "stress_xx", True),
    ("run.ladruno", "gauss", "equivalent_plastic_strain", True),  # stored, not derived
    ("run.results.h5", "gauss", "principal_stress_1", False),
    ("run.part-0.mpco", "gauss", "max_shear_strain", False),
    (None, "gauss", "von_mises_shell", False),
])
def test_the_rule(results_file, level: str, component: str, linear: bool) -> None:
    path = None if results_file is None else Path(results_file)
    assert recorded_is_linear(path, level, component) is linear


# ---- derived scalars, on every source ----------------------------------------

#: One Gauss point per element at 101 and 102; stage ``L`` is ``-0.5`` times ``D``,
#: so its von Mises is half of ``D``'s and never adds under a sum.
SIGMA_D = {"stress_xx": [10.0, -4.0], "stress_yy": [2.0, 6.0], "stress_zz": [0.0, 1.0],
           "stress_xy": [3.0, -1.0], "stress_yz": [0.0, 0.5], "stress_xz": [1.0, 0.0]}
DERIVED = ("von_mises_stress", "principal_stress_1")


def _native_gauss(path: Path) -> Path:
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        for name, scale in (("D", 1.0), ("L", -0.5)):
            sid = w.begin_stage(name=name, kind="static", time=np.array([0.0]))
            w.write_gauss_group(
                sid, "partition_0", group_id="solids", class_tag=33002, int_rule=3,
                element_index=np.array([101, 102], dtype=np.int64),
                natural_coords=np.zeros((1, 3)),
                components={c: scale * np.array(v).reshape(1, 2, 1) for c, v in SIGMA_D.items()})
            w.end_stage()
    return path


@pytest.fixture(params=["native", "ladruno"])
def derived_set(request, tmp_path: Path) -> tuple[Results, Path, tuple[Term, Term]]:
    """``(Results, user.h5, (a, b))``: ``ULS = 1.2 a + 1.6 b`` and ``ENV = max(a, b)``.

    Native: stages ``D`` and ``L`` of :func:`_native_gauss`. ``.ladruno``: the
    two steps of ``quad2d.ladruno``'s one stage, each a state.
    """
    if request.param == "native":
        results_file = _native_gauss(tmp_path / "g.results.h5")
        run = _open(results_file)
        a, b = Term("stage", "D", -1, 0, 1.2), Term("stage", "L", -1, 0, 1.6)
    else:
        results_file = tmp_path / "quad.ladruno"
        shutil.copy(LADRUNO / "quad2d.ladruno", results_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run = Results.from_ladruno(results_file)
        a, b = Term("stage", "stage_0", 0, 0, 1.2), Term("stage", "stage_0", 1, 0, 1.6)
    unit = tuple(Term(t.kind, t.source, t.step, 0, 1.0) for t in (a, b))
    user = write_user_file(
        tmp_path / "set.user.h5", _model(tmp_path / "set.model.h5"), results_file=results_file,
        combinations=[Combination("ULS", "linear", (a, b)),
                      Combination("ENV", "envelope", unit, envelope="max")])
    return run, user, (a, b)


def _gauss(run: Results, term: Term, component: str) -> np.ndarray:
    scoped = run.stage(term.source)
    return np.asarray(scoped.elements.gauss.get(component=component, time=term.step).values[-1])


@pytest.mark.parametrize("component", DERIVED)
def test_a_derived_scalar_refuses_a_summing_rule_on_every_source(derived_set, component) -> None:
    run, user, _ = derived_set
    with pytest.raises(CombinationRefused) as exc:
        run.combinations(user).evaluate("ULS", component, level="gauss")
    assert exc.value.reason == "not-linear"
    assert "derived scalar" in str(exc.value)


def test_the_tensor_a_derived_scalar_comes_from_still_sums(derived_set) -> None:
    run, user, (a, b) = derived_set
    got = run.combinations(user).evaluate("ULS", "stress_xx", level="gauss")
    want = 1.2 * _gauss(run, a, "stress_xx") + 1.6 * _gauss(run, b, "stress_xx")
    np.testing.assert_allclose(got.field.values[:, 0], want, rtol=1e-15)


@pytest.mark.filterwarnings("ignore:Out-of-plane stress_zz recovery")  # quad2d's plane is unknown
@pytest.mark.parametrize("component", DERIVED)
def test_a_derived_scalar_still_envelopes_and_reads_alone(derived_set, component) -> None:
    run, user, (a, b) = derived_set
    combos = run.combinations(user)
    first, second = _gauss(run, a, component), _gauss(run, b, component)
    assert not np.array_equal(first, second)  # the envelope picks between two states
    env = combos.evaluate("ENV", component, level="gauss")
    np.testing.assert_array_equal(env.field.values[:, 0], np.maximum(first, second))
    lone = combos.evaluate(StateAddress(a.kind, a.source, a.step, 0), component, level="gauss")
    np.testing.assert_array_equal(lone.field.values[:, 0], first)
    assert lone.badges == ()
