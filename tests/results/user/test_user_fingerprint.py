"""``rf1`` (ADR 0130 A6): goldens from outside this package, and what moves it.

The goldens were computed by two implementations independent of
``results/user/_fingerprint.py``: VU0's Python prototype and its TypeScript twin
through the app's h5wasm and ``node:crypto``, both extended to index every family
under ``RESULTS``. The fallback and part-set rows are the TypeScript one's alone,
since the Python prototype has no rule for them. A digest change here breaks the
app's slice VU3a too.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.user import FingerprintRefused, results_fingerprint

from .conftest import LADRUNO, REPO

GOLDEN = {
    "beam3d": "48ab1eaaa17047157458d3afb319f82940c10b1c8d377d988e5ff48d343748ac",
    "bezier_tet10": "97b1afb18534d0462a7674b8c5133dae2830cd8bae0d7b9f99bab292bad6b3fe",
    "bezier_tri6": "844792face2e8e8640b900127a288d446613be00ed8c0a0f4f5fb95409ee24de",
    "energy": "b893d0b39fed406013e9b06fd1390f668c5adb50f0eca0f0b8071a14f2f976f3",
    "fiberbeam": "a991d7666d23d8a9e1d7b8d954262058ccf019f3bf93ba1ea1bdb801d946a09a",
    "node_envelope": "732318ca4c1a9aac26c28132c18219fb63bb850636437b28f329d8e9085f4b22",
    "quad2d": "2d3998a3fd7273399536406d9fcfcc5b88cc0af9249ba4b6070e5806ddd22bc6",
    "truss2d": "76a404478cbb9ef919c2d79a676fbe1a0a300712819b344edd9a43e7289cfcc3",
}
GOLDEN_PART_SET = "d62d58956af134bd46e7bd4e236f03808c777b99d33eff93ecec750d35621ce5"
GOLDEN_MODES = "eecf2a003117de0df4521ba47d9b5e38142d76f1a597d10729057a86e65435fa"
GOLDEN_ELEMENT_ONLY = "b3ef4e3f7e5d786e087af7346a786ab847f1445e9e412dbb04c4c3900067ca3c"

DISP = "MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT"
BUCKET = "MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/basicForce/12-Truss[1:0:0]"


@pytest.mark.parametrize("stem", sorted(GOLDEN))
def test_rf1_equals_the_golden(stem: str) -> None:
    assert results_fingerprint(LADRUNO / f"{stem}.ladruno") == f"rf1:{GOLDEN[stem]}"


def test_rf1_of_modes_hashes_lambda_and_equals_the_golden() -> None:
    path = REPO / "apeGmshViewer" / "fixtures" / "ladruno_modes.ladruno"
    assert results_fingerprint(path) == f"rf1:{GOLDEN_MODES}"


@pytest.mark.parametrize("part", ["truss2d.part-0.ladruno", "truss2d.part-1.ladruno"])
def test_rf1_of_a_part_is_the_rf1_of_its_set(part: str) -> None:
    assert results_fingerprint(LADRUNO / part) == f"rf1:{GOLDEN_PART_SET}"


# ---- the element fallback (the amendment's addition, measured here) -------


def _element_only(path: Path) -> Path:
    with h5py.File(path, "a") as f:
        del f["MODEL_STAGE[1]/RESULTS/ON_NODES"]
    return path


def test_fallback_equals_the_cross_language_golden(ladruno: Path) -> None:
    assert results_fingerprint(_element_only(ladruno)) == f"rf1:{GOLDEN_ELEMENT_ONLY}"


def test_fallback_sees_the_last_slab_of_an_element_only_stage(ladruno: Path) -> None:
    before = results_fingerprint(_element_only(ladruno))
    with h5py.File(ladruno, "a") as f:
        data = f[f"{BUCKET}/DATA"]
        data[data.shape[0] - 1] = data[data.shape[0] - 1] + 1.0
    assert results_fingerprint(ladruno) != before


def test_nodal_sample_wins_so_the_element_slab_is_not_read_when_a_node_result_exists(
    ladruno: Path,
) -> None:
    """The fallback is a fallback: with a nodal result the bucket's values do
    not enter the digest, which is what the measured algorithm hashes."""
    before = results_fingerprint(ladruno)
    with h5py.File(ladruno, "a") as f:
        data = f[f"{BUCKET}/DATA"]
        data[data.shape[0] - 1] = data[data.shape[0] - 1] + 1.0
    assert results_fingerprint(ladruno) == before


# ---- what moves it, and what does not -------------------------------------


def test_stamping_leaves_rf1_unchanged_with_the_real_stamp(ladruno: Path, tmp_path: Path) -> None:
    from apeGmsh.results.writers import stamp_provenance

    from tests.test_results_ladruno_provenance import _write_model

    before = results_fingerprint(ladruno)
    model = _write_model(tmp_path / "bridge.h5")
    whole_file_before = ladruno.read_bytes()
    stamp_provenance(ladruno, model)
    assert ladruno.read_bytes() != whole_file_before, "the stamp wrote nothing: the test is vacuous"
    with h5py.File(ladruno, "r") as f:
        assert "model_hash" in f["INFO/PROVENANCE"].attrs
    assert results_fingerprint(ladruno) == before


@pytest.mark.parametrize("change", ["last_slab", "extra_step", "time", "run_id"])
def test_a_rerun_changes_rf1(ladruno: Path, change: str) -> None:
    before = results_fingerprint(ladruno)
    with h5py.File(ladruno, "a") as f:
        group = f[DISP]
        data = group["DATA"]
        if change == "last_slab":
            data[data.shape[0] - 1] = data[data.shape[0] - 1] * 1.001 + 1e-9
        elif change == "extra_step":
            t = data.shape[0]
            for name in ("DATA", "TIME", "STEP"):
                group[name].resize(t + 1, axis=0)
                group[name][t] = group[name][t - 1]
        elif change == "time":
            group["TIME"][0] = group["TIME"][0] + 0.5
        else:
            f["INFO"].attrs["RUN_ID"] = "another-run"
    assert results_fingerprint(ladruno) != before


def test_a_changed_lambda_changes_rf1(tmp_path: Path) -> None:
    path = tmp_path / "modes.ladruno"
    shutil.copy(REPO / "apeGmshViewer" / "fixtures" / "ladruno_modes.ladruno", path)
    before = results_fingerprint(path)
    with h5py.File(path, "a") as f:
        mode = f["MODEL_STAGE[1]/RESULTS/ON_NODES/MODES_OF_VIBRATION(U)/DATA/STEP_0/MODE_1"]
        mode.attrs["LAMBDA"] = np.asarray(mode.attrs["LAMBDA"]) * 1.01
    assert results_fingerprint(path) != before


def test_the_stated_limit_a_rewrite_of_earlier_steps_is_not_seen(ladruno: Path) -> None:
    """A6 states this limit; a test keeps the statement honest."""
    before = results_fingerprint(ladruno)
    with h5py.File(ladruno, "a") as f:
        f[f"{DISP}/DATA"][0] = f[f"{DISP}/DATA"][0] + 1.0
    assert results_fingerprint(ladruno) == before


def _energy(tmp_path: Path) -> Path:
    path = tmp_path / "energy.ladruno"
    shutil.copy(LADRUNO / "energy.ladruno", path)
    return path


def _envelope(tmp_path: Path) -> Path:
    path = tmp_path / "env.ladruno"
    shutil.copy(LADRUNO / "node_envelope.ladruno", path)
    return path


ON_DOMAIN = "MODEL_STAGE[1]/RESULTS/ON_DOMAIN/energyBalance"
ON_REGIONS = "MODEL_STAGE[1]/RESULTS/ON_REGIONS/energyBalance"
ENV_MAX = "MODEL_STAGE[1]/RESULTS/ENVELOPES/ON_NODES/DISPLACEMENT/MAX"


@pytest.mark.parametrize("target", [ON_DOMAIN, ON_REGIONS])
def test_deleting_a_result_of_any_family_moves_rf1(tmp_path: Path, target: str) -> None:
    path = _energy(tmp_path)
    before = results_fingerprint(path)
    with h5py.File(path, "a") as f:
        del f[target]
    assert results_fingerprint(path) != before


def test_deleting_an_envelope_moves_rf1(tmp_path: Path) -> None:
    path = _envelope(tmp_path)
    before = results_fingerprint(path)
    with h5py.File(path, "a") as f:
        del f[ENV_MAX]
    assert results_fingerprint(path) != before


def test_a_new_time_or_shape_in_on_domain_moves_rf1(tmp_path: Path) -> None:
    path = _energy(tmp_path)
    before = results_fingerprint(path)
    with h5py.File(path, "a") as f:
        f[f"{ON_DOMAIN}/TIME"][0] = f[f"{ON_DOMAIN}/TIME"][0] + 1.0
    assert results_fingerprint(path) != before
    path = _energy(tmp_path)
    with h5py.File(path, "a") as f:
        group = f[ON_DOMAIN]
        shape = group["DATA"].shape
        del group["DATA"]
        group.create_dataset("DATA", data=np.zeros((shape[0], shape[1], shape[2] + 1)))
    assert results_fingerprint(path) != before


def test_a_family_rf1_does_not_know_yet_is_indexed(ladruno: Path) -> None:
    before = results_fingerprint(ladruno)
    with h5py.File(ladruno, "a") as f:
        f.create_group("MODEL_STAGE[1]/RESULTS/ON_SOMETHING/thing").create_dataset(
            "x", data=np.zeros(3))
    assert results_fingerprint(ladruno) != before


def test_a_dtype_rf1_cannot_encode_in_a_family_refuses(ladruno: Path) -> None:
    with h5py.File(ladruno, "a") as f:
        f.create_group("MODEL_STAGE[1]/RESULTS/ON_SOMETHING").create_dataset(
            "x", data=["text"], dtype=h5py.string_dtype())
    with pytest.raises(FingerprintRefused, match="dtype"):
        results_fingerprint(ladruno)


@pytest.mark.parametrize(
    "make, target",
    [(_energy, f"{ON_DOMAIN}/DATA"), (_envelope, ENV_MAX)],
)
def test_the_stated_limit_values_outside_the_sample_are_not_seen(tmp_path: Path, make, target) -> None:
    """Every family is indexed, and the sampled slab rule is as the schema has it,
    so a rewrite that keeps each shape and ``TIME`` is not seen in these families."""
    path = make(tmp_path)
    before = results_fingerprint(path)
    with h5py.File(path, "a") as f:
        f[target][...] = f[target][...] + 1.0
    assert results_fingerprint(path) == before


def test_the_sample_stays_on_nodes_with_another_family_present(tmp_path: Path) -> None:
    path = _energy(tmp_path)
    before = results_fingerprint(path)
    with h5py.File(path, "a") as f:
        data = f[f"{DISP}/DATA"]
        data[data.shape[0] - 1] = data[data.shape[0] - 1] + 1.0
    assert results_fingerprint(path) != before


# ---- refusals --------------------------------------------------------------


def test_a_file_that_is_not_hdf5_refuses(tmp_path: Path) -> None:
    path = tmp_path / "junk.ladruno"
    path.write_bytes(b"not hdf5")
    with pytest.raises(FingerprintRefused, match="cannot be fingerprinted"):
        results_fingerprint(path)


def test_a_dangling_link_inside_a_result_refuses(ladruno: Path) -> None:
    with h5py.File(ladruno, "a") as f:
        del f[f"{DISP}/TIME"]
        f[f"{DISP}/TIME"] = h5py.SoftLink("/nowhere")
    with pytest.raises(FingerprintRefused):
        results_fingerprint(ladruno)


def test_a_result_that_is_a_dataset_refuses(ladruno: Path) -> None:
    with h5py.File(ladruno, "a") as f:
        del f[DISP]
        f[DISP] = np.zeros(3)
    with pytest.raises(FingerprintRefused):
        results_fingerprint(ladruno)


def test_missing_file_refuses(tmp_path: Path) -> None:
    with pytest.raises(FingerprintRefused, match="no such"):
        results_fingerprint(tmp_path / "absent.ladruno")


def test_a_stage_without_kind_refuses(ladruno: Path) -> None:
    with h5py.File(ladruno, "a") as f:
        del f["MODEL_STAGE[1]"].attrs["KIND"]
    with pytest.raises(FingerprintRefused, match="KIND"):
        results_fingerprint(ladruno)


def test_a_file_that_is_neither_refuses(tmp_path: Path) -> None:
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f.create_group("nodes")
    with pytest.raises(FingerprintRefused, match="neither"):
        results_fingerprint(path)


def test_a_gap_in_a_part_set_refuses(tmp_path: Path) -> None:
    shutil.copy(LADRUNO / "truss2d.part-0.ladruno", tmp_path / "t.part-0.ladruno")
    shutil.copy(LADRUNO / "truss2d.part-1.ladruno", tmp_path / "t.part-2.ladruno")
    with pytest.raises(FingerprintRefused, match="contiguous"):
        results_fingerprint(tmp_path / "t.part-0.ladruno")


def test_a_native_file_uses_its_lineage_hash(tmp_path: Path) -> None:
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as f:
        f.create_group("meta/lineage").attrs["results_hash"] = "abc123"
    assert results_fingerprint(path) == "native:abc123"
