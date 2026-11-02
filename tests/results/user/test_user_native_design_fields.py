"""A native ``results.h5`` design field is not linear (ADR 0130 A5, owner ruling VU2c).

The YOO slab's native file stores ``wood_armer_*`` beside ``displacement_z`` as
an ordinary component, and a summing combination summed it. Here a native file
written with ``NativeWriter`` holds two stages, ``D`` and ``L``, of a recorder
spelling (``displacement_z``) and a design field (``wood_armer_pos_x``, values
that are not linear in the stage's load). The oracle for every value is the
plain ``Results`` read of each stage, combined by hand.
"""
from __future__ import annotations

import shutil
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
])
def test_the_rule(results_file, level: str, component: str, linear: bool) -> None:
    path = None if results_file is None else Path(results_file)
    assert recorded_is_linear(path, level, component) is linear
