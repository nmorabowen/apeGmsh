"""Layered ShellMITC3 and stored shell frames: MPCO and ``.ladruno`` agree (#2090).

ShellMITC3 is a layered-shell class no build has: the pinned fork (FORK_PIN
c744d6e70) answers ``element type ShellMITC3 is unknown``, and upstream has
no such element either, so the bridge dropped it (#2092). A file from another
writer could still name it, so it stays here as a foreign class: it has no
``LAYER_CATALOG`` row, and both readers warn
:class:`LayerBucketUnreadWarning` by class for a layered bucket they leave
out, where MPCO used to drop it with no word (``.ladruno`` warns since #1995).

A stored ``.ladruno`` shell frame whose ``FRAME`` size disagrees with its
``ID`` used to be dropped silently and recomputed from coordinates; it now
warns :class:`ShellFrameUnavailableWarning`, as MPCO's ``_stored_quaternions``
does, and the frames are still computed (closed form: ``shell_quaternion``
of the file's own nodes).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import element as _bridge_elements
from apeGmsh.opensees._element_capabilities import Unknown, element_capability
from apeGmsh.opensees._response_catalog import (
    ELE_TAG_ASDShellQ4,
    LAYER_CATALOG,
    IntRule,
)
from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_layer_io as _mlayer
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_element_io import (
    LayerBucketUnreadWarning as LadrunoLayerBucketUnreadWarning,
)
from apeGmsh.results.readers._mpco_layer_io import (
    LayerBucketUnreadWarning,
    ShellFrameUnavailableWarning,
)
from apeGmsh.results.readers._protocol import ResultLevel
from tests.conftest import _stub_model_h5_path
from tests.results.readers.test_ladruno_reader import (
    N_LAYERS,
    SHELL_TAGS,
    _frame,
    _layered_shell_file,
)
from tests.test_results_mpco_layer_mock import (
    _add_layer_bucket,
    _add_layered_section,
    _add_shell_connectivity,
    _create_skeleton,
)

#: A class tag for a mock ShellMITC3 bucket: no build has ShellMITC3,
#: so no real tag exists (the readers key on the class name).
_MOCK_TAG = 999


# ---------------------------------------------------------------------------
# The roster: ShellMITC3 is a foreign class, outside the bridge (#2092)
# ---------------------------------------------------------------------------


def test_shellmitc3_is_a_foreign_class_without_a_layer_row() -> None:
    assert not hasattr(_bridge_elements, "ShellMITC3")
    assert element_capability("ShellMITC3") is Unknown
    assert ("ShellMITC3", "fiber_stress") not in LAYER_CATALOG
    assert ("ShellMITC3", "fiber_strain") not in LAYER_CATALOG


def test_one_warning_class_for_both_formats() -> None:
    assert LadrunoLayerBucketUnreadWarning is LayerBucketUnreadWarning


# ---------------------------------------------------------------------------
# MPCO: every layered bucket discovery leaves out warns by class
# ---------------------------------------------------------------------------


def _mpco_with_bucket(
    path: Path, *, class_name: str, class_tag: int, rule: int,
    token: str = "material.fiber.stress", header: int = 0,
) -> Path:
    """A one-stage MPCO file holding one two-element layered bucket."""
    eids = np.array([20, 21], dtype=np.int64)
    f, stage_name = _create_skeleton(
        path, node_ids=np.arange(1, 9, dtype=np.int64),
        coords=np.zeros((8, 3)), n_steps=1, dt=1.0,
    )
    stage = f[stage_name]
    _add_shell_connectivity(
        stage, class_tag=class_tag, class_name=class_name,
        surface_int_rule=rule, custom_rule_idx=0, element_ids=eids,
    )
    _add_layered_section(
        stage, section_tag=1, section_class="LayeredShellFiberSection",
        element_sgp_pairs=[(int(e), g) for e in eids for g in range(3)],
        layer_thickness=np.array([0.01, 0.02]),
        layer_material=np.array([1, 1], dtype=np.int64),
    )
    key = _add_layer_bucket(
        stage, class_tag=class_tag, class_name=class_name,
        surface_int_rule=rule, custom_rule_idx=0, element_ids=eids,
        n_sgp=3, n_layers=2, flat_data=np.ones((1, 2, 6)), token=token,
    )
    if header:
        grp = stage[f"RESULTS/ON_ELEMENTS/{token}"]
        grp.move(key, key.replace(":0]", f":{header}]"))
    f.close()
    return path


def _discover(path: Path, canonical: str = "fiber_stress") -> list:
    with h5py.File(path, "r") as f:
        _, buckets = _mlayer.discover_layer_buckets(
            f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"],
            canonical_component=canonical,
        )
    return buckets


@pytest.mark.parametrize("token", ["material.fiber.stress",
                                   "section.fiber.stress"])
def test_mpco_shellmitc3_bucket_warns_by_class(
    tmp_path: Path, token: str,
) -> None:
    path = _mpco_with_bucket(
        tmp_path / "mitc3.mpco", class_name="ShellMITC3",
        class_tag=_MOCK_TAG, rule=IntRule.Triangle_GL_2C, token=token,
    )
    with pytest.warns(LayerBucketUnreadWarning,
                      match=r"'ShellMITC3' has no layered-shell entry .*"
                            r"LAYER_CATALOG") as rec:
        assert _discover(path) == []
    assert token in str(rec[0].message)


def test_mpco_shellmitc3_warns_through_the_public_reader(
    tmp_path: Path,
) -> None:
    path = _mpco_with_bucket(
        tmp_path / "mitc3.mpco", class_name="ShellMITC3",
        class_tag=_MOCK_TAG, rule=IntRule.Triangle_GL_2C,
    )
    results = Results.from_mpco(str(path), model_h5=_stub_model_h5_path())
    try:
        reader = results._reader
        with pytest.warns(LayerBucketUnreadWarning, match="ShellMITC3"):
            assert reader.available_components(
                reader.stages()[0].id, ResultLevel.LAYERS) == []
        with pytest.warns(LayerBucketUnreadWarning, match="ShellMITC3"):
            slab = results.elements.layers.get(component="fiber_stress")
        assert slab.values.shape[1] == 0
    finally:
        results._reader.close()


def test_mpco_header_index_above_zero_warns(tmp_path: Path) -> None:
    path = _mpco_with_bucket(
        tmp_path / "hdr.mpco", class_name="ASDShellQ4",
        class_tag=ELE_TAG_ASDShellQ4, rule=IntRule.Quad_GL_2, header=1,
    )
    with pytest.warns(LayerBucketUnreadWarning,
                      match=r"'ASDShellQ4' bucket has header index 1"):
        assert _discover(path) == []


def test_mpco_rule_other_than_the_surface_rule_warns(tmp_path: Path) -> None:
    path = _mpco_with_bucket(
        tmp_path / "rule.mpco", class_name="ASDShellQ4",
        class_tag=ELE_TAG_ASDShellQ4, rule=IntRule.Quad_GL_3,
    )
    with pytest.warns(
        LayerBucketUnreadWarning,
        match=rf"'ASDShellQ4' bucket has integration rule {IntRule.Quad_GL_3}"
              rf".*surface rule {IntRule.Quad_GL_2}",
    ):
        assert _discover(path) == []


def test_mpco_unparseable_key_warns(tmp_path: Path) -> None:
    path = _mpco_with_bucket(
        tmp_path / "key.mpco", class_name="ASDShellQ4",
        class_tag=ELE_TAG_ASDShellQ4, rule=IntRule.Quad_GL_2,
    )
    with h5py.File(path, "r+") as f:
        grp = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/material.fiber.stress"]
        grp.move(next(iter(grp)), "garbage")
    with pytest.warns(LayerBucketUnreadWarning, match="garbage"):
        assert _discover(path) == []


def test_mpco_catalogued_shell_and_beam_fibers_stay_silent(
    tmp_path: Path,
) -> None:
    """The warning fires only where data is left unread: a catalogued
    layered shell is read, and a beam fiber-section bucket in the shared
    ``section.fiber.stress`` group belongs to the fiber level."""
    path = _mpco_with_bucket(
        tmp_path / "ok.mpco", class_name="ASDShellQ4",
        class_tag=ELE_TAG_ASDShellQ4, rule=IntRule.Quad_GL_2,
    )
    with h5py.File(path, "r+") as f:
        grp = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"].create_group(
            "section.fiber.stress/74-ForceBeamColumn3d[1000:0:0]")
        grp.create_dataset("ID", data=np.array([[30]], dtype=np.int32))
    with warnings.catch_warnings():
        warnings.simplefilter("error", LayerBucketUnreadWarning)
        (bucket,) = _discover(path)
    assert bucket.elem_key.class_name == "ASDShellQ4"


# ---------------------------------------------------------------------------
# .ladruno: the same ShellMITC3 bucket warns by class (since #1995)
# ---------------------------------------------------------------------------


def test_ladruno_shellmitc3_bucket_warns_by_class(tmp_path: Path) -> None:
    path = _layered_shell_file(tmp_path, class_name="ShellMITC3")
    with LadrunoReader(path) as r:
        with pytest.warns(LayerBucketUnreadWarning,
                          match=r"'ShellMITC3' has no layered-shell entry"):
            assert r.available_components("stage_0", ResultLevel.LAYERS) == []


# ---------------------------------------------------------------------------
# .ladruno: a stored frame the reader cannot use warns, as MPCO does
# ---------------------------------------------------------------------------


def _with_stored_frame(path: Path, **datasets: np.ndarray) -> Path:
    tag = LAYER_CATALOG[("ASDShellQ4", "fiber_stress")].class_tag
    with h5py.File(path, "r+") as f:
        stage = f[next(k for k in f if k.startswith("MODEL_STAGE["))]
        grp = stage["MODEL/LOCAL_AXES"].create_group(f"{tag}-ASDShellQ4")
        for name, data in datasets.items():
            grp.create_dataset(name, data=data)
    return path


def _assert_frames_computed(path: Path, match: str) -> None:
    with LadrunoReader(path) as r:
        with pytest.warns(ShellFrameUnavailableWarning, match=match):
            q = r.read_layers("stage_0", "fiber_stress_0").local_axes_quaternion
    per_e = 4 * N_LAYERS
    want = np.concatenate([
        np.tile(_frame("ASDShellQ4", e), (per_e, 1))
        for e in range(len(SHELL_TAGS))
    ])
    np.testing.assert_allclose(q, want, atol=1e-14)


def test_ladruno_stored_frame_size_mismatch_warns(tmp_path: Path) -> None:
    path = _with_stored_frame(
        _layered_shell_file(tmp_path),
        ID=np.array(SHELL_TAGS, dtype=np.int32),
        FRAME=np.array([[0.0, 0.0, 0.0, 1.0]]),
    )
    _assert_frames_computed(path, r"2 IDs but FRAME has shape \(1, 4\)")


def test_ladruno_stored_frame_missing_dataset_warns(tmp_path: Path) -> None:
    path = _with_stored_frame(
        _layered_shell_file(tmp_path),
        ID=np.array(SHELL_TAGS, dtype=np.int32),
    )
    _assert_frames_computed(path, r"expected ID and FRAME datasets")


def test_ladruno_consistent_stored_frame_is_silent(tmp_path: Path) -> None:
    stored = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    path = _with_stored_frame(
        _layered_shell_file(tmp_path),
        ID=np.array(SHELL_TAGS, dtype=np.int32), FRAME=stored,
    )
    with LadrunoReader(path) as r, warnings.catch_warnings():
        warnings.simplefilter("error", ShellFrameUnavailableWarning)
        q = r.read_layers("stage_0", "fiber_stress_0").local_axes_quaternion
    per_e = 4 * N_LAYERS
    np.testing.assert_array_equal(
        q, np.concatenate([np.tile(s, (per_e, 1)) for s in stored]))
