"""#2072: frame-beam nodal forces on MPCO, DomainCapture and ``.out``.

An MPCO read of ``nodal_resisting_*`` on a force-based beam used to return
an empty slab with no warning: the fork writes a frame beam's
``globalForce`` / ``localForce`` bucket under a custom-rule key, and the
reader skipped every custom-rule bucket, then every class outside
``NODAL_FORCE_CATALOG``. The catalog now lists the frame beams, the reader
takes any custom-rule index, and a bucket it leaves unread warns by class.

The bucket keys below are the ones the fork's MPCO recorder wrote on build
79e0623670 for these classes (``_FRAME_KEYS``). ``NODAL_FORCE_CATALOG``
also gates DomainCapture and the ``.out`` transcoder, so each added class
is read on all three routes here, from one synthetic end-force history.
The fork lane solves the same classes for real
(``tests/opensees/integration_ladruno/test_mpco_frame_nodal_forces_live.py``).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    NODAL_FORCE_CATALOG,
    is_line_station_synthesis_catalogued,
    synthesize_line_station_layout_for_elastic_beam,
)
from apeGmsh.results import Results
from apeGmsh.results.capture._domain import DomainCapture
from apeGmsh.results.capture.spec import ResolvedDomainCaptureRecord
from apeGmsh.results.readers._mpco_nodal_io import (
    NodalForceBucketSkippedWarning,
)
from apeGmsh.results.spec._resolved import ResolvedRecorderRecord
from apeGmsh.results.transcoders._recorder import RecorderTranscoder

from tests.conftest import _open_model_from_h5, _stub_model_h5_path
from tests.fixtures.capture_tags import identity_tags
from tests.test_results_domain_capture_nodal_forces import (
    _FakeOpsBeams,
    _MockFem,
    _spec_with,
)
from tests.test_results_mpco_nodal_mock import _create_skeleton
from tests.test_results_recorder_transcoder_nodal_forces import (
    _make_spec,
    _MinimalFem,
    _write_element_out,
)

#: class -> (MPCO bucket key the fork writes, ndm). Measured on 79e0623670.
_FRAME_KEYS: dict[str, tuple[str, int]] = {
    "ForceBeamColumn2d": ("73-ForceBeamColumn2d[1000:1:0]", 2),
    "ForceBeamColumn3d": ("74-ForceBeamColumn3d[1000:1:0]", 3),
    "DispBeamColumn2d": ("62-DispBeamColumn2d[1000:1:0]", 2),
    "DispBeamColumn3d": ("64-DispBeamColumn3d[1000:1:0]", 3),
    "LadrunoDispBeamColumn2d": ("33013-LadrunoDispBeamColumn2d[0:0:0]", 2),
    "LadrunoDispBeamColumn3d": ("33014-LadrunoDispBeamColumn3d[0:0:0]", 3),
}
_CLASSES = sorted(_FRAME_KEYS)

_GLOBAL = {
    2: ("nodal_resisting_force_x", "nodal_resisting_force_y",
        "nodal_resisting_moment_z"),
    3: ("nodal_resisting_force_x", "nodal_resisting_force_y",
        "nodal_resisting_force_z", "nodal_resisting_moment_x",
        "nodal_resisting_moment_y", "nodal_resisting_moment_z"),
}
_LOCAL = {
    2: ("nodal_resisting_force_local_x", "nodal_resisting_force_local_y",
        "nodal_resisting_moment_local_z"),
    3: ("nodal_resisting_force_local_x", "nodal_resisting_force_local_y",
        "nodal_resisting_force_local_z", "nodal_resisting_moment_local_x",
        "nodal_resisting_moment_local_y", "nodal_resisting_moment_local_z"),
}
_FRAMES = {"globalForce": _GLOBAL, "localForce": _LOCAL}

_EIDS = np.array([7, 11], dtype=np.int64)
_T = 2


def _flat(ndm: int, frame: str) -> np.ndarray:
    """``(T, E, 2 * n)`` end forces, every slot distinct:
    ``frame_offset + t * 1000 + eid * 10 + n * n_comp + k``."""
    n_comp = 3 if ndm == 2 else 6
    off = 0.0 if frame == "globalForce" else 0.5
    flat = np.empty((_T, _EIDS.size, 2 * n_comp))
    for t in range(_T):
        for i, e in enumerate(_EIDS):
            flat[t, i] = off + t * 1000 + e * 10 + np.arange(2 * n_comp)
    return flat


def _expected(ndm: int, frame: str, k: int) -> np.ndarray:
    """``(T, E, 2)``: component ``k`` at end i and end j."""
    n_comp = 3 if ndm == 2 else 6
    return _flat(ndm, frame)[:, :, [k, k + n_comp]]


def _add_bucket(stage: "h5py.Group", frame: str, key: str,
                flat: np.ndarray) -> None:
    """A single-block nodal-force bucket, as the fork's MPCO recorder
    writes it (NUM_COLUMNS, META with GAUSS_IDS = [-1], ID, DATA)."""
    n_total = flat.shape[2]
    bucket = stage["RESULTS/ON_ELEMENTS"].require_group(frame).create_group(key)
    bucket.attrs["NUM_COLUMNS"] = np.array([n_total], dtype=np.int32)
    meta = bucket.create_group("META")
    meta.create_dataset("MULTIPLICITY", data=np.array([[1]], dtype=np.int32))
    meta.create_dataset("GAUSS_IDS", data=np.array([[-1]], dtype=np.int32))
    meta.create_dataset("NUM_COMPONENTS",
                        data=np.array([[n_total]], dtype=np.int32))
    meta.create_dataset("COMPONENTS", data=np.array([b"0.unused"]))
    bucket.create_dataset("ID", data=_EIDS.reshape(-1, 1).astype(np.int32))
    data = bucket.create_group("DATA")
    for t in range(flat.shape[0]):
        ds = data.create_dataset(f"STEP_{t}", data=flat[t])
        ds.attrs["STEP"] = t
        ds.attrs["TIME"] = float(t + 1)


def _mpco(tmp_path: Path, buckets: list[tuple[str, str, np.ndarray]]) -> Path:
    path = tmp_path / "frame.mpco"
    f, stage_name = _create_skeleton(
        path, node_ids=np.array([1, 2, 3], dtype=np.int32),
        coords=np.zeros((3, 3)), n_steps=_T, dt=1.0,
    )
    try:
        for frame, key, flat in buckets:
            _add_bucket(f[stage_name], frame, key, flat)
    finally:
        f.close()
    return path


# =====================================================================
# The catalog
# =====================================================================

@pytest.mark.parametrize("cls", _CLASSES)
def test_catalog_lists_the_frame_beam_in_both_frames(cls: str) -> None:
    key, ndm = _FRAME_KEYS[cls]
    tag = int(key.split("-", 1)[0])
    for frame, names in (("global", _GLOBAL), ("local", _LOCAL)):
        lay = NODAL_FORCE_CATALOG[(cls, f"{frame}_force")]
        assert (lay.class_tag, lay.frame, lay.n_nodes_per_element) == (
            tag, frame, 2)
        assert lay.component_layout == names[ndm]


@pytest.mark.parametrize("cls", _CLASSES)
def test_frame_beam_stations_stay_on_section_force(cls: str) -> None:
    """A frame beam's ``localForce`` is catalogued for nodal forces only:
    its line stations come from ``section force``, never the elastic-beam
    synthesis."""
    assert not is_line_station_synthesis_catalogued(cls)
    assert synthesize_line_station_layout_for_elastic_beam(cls) is None
    assert is_line_station_synthesis_catalogued("ElasticBeam3d")


# =====================================================================
# MPCO
# =====================================================================

@pytest.mark.parametrize("cls", _CLASSES)
def test_mpco_reads_the_frame_beam_bucket(tmp_path: Path, cls: str) -> None:
    key, ndm = _FRAME_KEYS[cls]
    path = _mpco(tmp_path, [(fr, key, _flat(ndm, fr)) for fr in _FRAMES])
    with warnings.catch_warnings():
        warnings.simplefilter("error", NodalForceBucketSkippedWarning)
        with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
            avail = set(r.elements.available_components())
            for fr, names in _FRAMES.items():
                assert set(names[ndm]) <= avail
                for k, comp in enumerate(names[ndm]):
                    slab = r.elements.get(component=comp)
                    np.testing.assert_array_equal(slab.element_ids, _EIDS)
                    np.testing.assert_array_equal(
                        slab.values, _expected(ndm, fr, k), err_msg=comp)


def test_mpco_uncatalogued_class_warns_by_name(tmp_path: Path) -> None:
    """Beside a readable ForceBeamColumn3d bucket, an uncatalogued class
    warns naming itself, and the read keeps the force beam's rows."""
    path = _mpco(tmp_path, [
        ("globalForce", _FRAME_KEYS["ForceBeamColumn3d"][0],
         _flat(3, "globalForce")),
        ("globalForce", "999-MysteryBeam3d[1000:1:0]",
         _flat(3, "globalForce")),
    ])
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.warns(NodalForceBucketSkippedWarning,
                          match="'MysteryBeam3d'.*NODAL_FORCE_CATALOG"):
            slab = r.elements.get(component="nodal_resisting_force_x")
    np.testing.assert_array_equal(slab.element_ids, _EIDS)
    np.testing.assert_array_equal(slab.values, _expected(3, "globalForce", 0))


def test_mpco_lone_uncatalogued_class_warns_before_the_empty_slab(
    tmp_path: Path,
) -> None:
    """The silent-empty case of #2072: the only bucket is unreadable."""
    path = _mpco(tmp_path, [
        ("globalForce", "999-MysteryBeam3d[1000:1:0]", _flat(3, "globalForce")),
    ])
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.warns(NodalForceBucketSkippedWarning, match="MysteryBeam3d"):
            slab = r.elements.get(component="nodal_resisting_force_x")
    assert slab.values.shape[1] == 0


def test_mpco_second_header_warns_by_name(tmp_path: Path) -> None:
    path = _mpco(tmp_path, [
        ("globalForce", "74-ForceBeamColumn3d[1000:1:1]",
         _flat(3, "globalForce")),
    ])
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.warns(NodalForceBucketSkippedWarning,
                          match="'ForceBeamColumn3d'.*response header 1"):
            r.elements.get(component="nodal_resisting_force_x")


def test_mpco_unread_bucket_outside_the_id_filter_stays_quiet(
    tmp_path: Path,
) -> None:
    """A filter that requests none of the unread bucket's elements reads
    the force beam alone, with no warning."""
    other = _flat(3, "globalForce")
    path = _mpco(tmp_path, [
        ("globalForce", _FRAME_KEYS["ForceBeamColumn3d"][0],
         _flat(3, "globalForce")),
    ])
    with h5py.File(path, "a") as f:
        stage = f["MODEL_STAGE[1]"]
        _add_bucket(stage, "globalForce", "999-MysteryBeam3d[1000:1:0]", other)
        stage["RESULTS/ON_ELEMENTS/globalForce/999-MysteryBeam3d[1000:1:0]/ID"][
            ...] = np.array([[40], [41]], dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NodalForceBucketSkippedWarning)
        with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
            slab = r.elements.get(component="nodal_resisting_force_x",
                                  ids=[7])
    np.testing.assert_array_equal(slab.element_ids, [7])


# =====================================================================
# DomainCapture and .out: the other two routes the catalog gates
# =====================================================================

@pytest.mark.parametrize("fr", list(_FRAMES))
@pytest.mark.parametrize("cls", _CLASSES)
def test_domain_capture_records_the_frame_beam(
    tmp_path: Path, cls: str, fr: str,
) -> None:
    ndm = _FRAME_KEYS[cls][1]
    names = _FRAMES[fr][ndm]
    fem = _MockFem([1, 2, 3])
    spec = _spec_with(
        ResolvedDomainCaptureRecord(
            category="elements", name="frame", components=names,
            dt=None, n_steps=None, element_ids=_EIDS,
        ),
        snapshot_id=fem.snapshot_id,
    )
    ops = _FakeOpsBeams()
    flat = _flat(ndm, fr)
    for e in _EIDS:
        ops.ele_class[int(e)] = cls
    path = tmp_path / "cap.h5"
    with DomainCapture(spec, path, fem, ops=ops,
                       tag_map=identity_tags(spec)) as cap:
        cap.begin_stage("s", kind="static")
        for t in range(_T):
            for i, e in enumerate(_EIDS):
                ops.ele_response[(int(e), fr)] = flat[t, i]
            cap.step(t=float(t + 1))
        cap.end_stage()
        (nfc,) = cap._nodal_force_capturers
        assert nfc.skipped_elements == []
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        for k, comp in enumerate(names):
            slab = r.elements.get(component=comp)
            np.testing.assert_array_equal(slab.element_ids, _EIDS)
            np.testing.assert_array_equal(
                slab.values, _expected(ndm, fr, k), err_msg=comp)


@pytest.mark.parametrize("fr", list(_FRAMES))
@pytest.mark.parametrize("cls", _CLASSES)
def test_out_transcoder_reads_the_frame_beam(
    tmp_path: Path, cls: str, fr: str,
) -> None:
    ndm = _FRAME_KEYS[cls][1]
    names = _FRAMES[fr][ndm]
    fem = _MinimalFem([1, 2, 3])
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="elements", name="frame", components=names,
            dt=None, n_steps=None, element_ids=_EIDS,
            element_class_name=cls,
        ),
        snapshot_id=fem.snapshot_id,
    )
    _write_element_out(tmp_path / "frame_elements.out",
                       times=np.array([1.0, 2.0]), flat=_flat(ndm, fr))
    target = tmp_path / "out.h5"
    RecorderTranscoder(
        spec, output_dir=tmp_path, target_path=target, fem=fem,
        stage_name="s", stage_kind="static",
    ).run()
    with Results.from_native(target, model=_open_model_from_h5(target)) as r:
        for k, comp in enumerate(names):
            slab = r.elements.get(component=comp)
            np.testing.assert_array_equal(slab.element_ids, _EIDS)
            np.testing.assert_array_equal(
                slab.values, _expected(ndm, fr, k), err_msg=comp)
