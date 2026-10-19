"""An MPCO nodal-force read refuses an element id held by two buckets (#2173).

MPCO writes each element to exactly one bucket of a token. A file where one
element id sits in two buckets is corrupt: the reader used to stitch both
buckets and return that element twice, with no warning. It now raises
:class:`NodalForceDuplicateElementError`, naming the id, both bucket paths
and the file.

The oracle is the #2172 reproducer: header 0 holds elements ``[7, 11]`` and
header 1 holds ``[11, 33]``; read before the fix, it returned
``[7, 11, 11, 33]``. The same split by custom rule (``[1000:1:0]`` and
``[1000:2:0]``) and by class must refuse too, and a file where each element
appears once must read each element's own values, as before.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import NODAL_FORCE_CATALOG
from apeGmsh.results import Results
from apeGmsh.results.readers._mpco_nodal_io import (
    NodalForceBucketSkippedWarning,
    NodalForceDuplicateElementError,
)
from tests.conftest import _stub_model_h5_path
from tests.test_results_mpco_nodal_mock import _create_skeleton

_T = 2
_G3 = ("Px", "Py", "Pz", "Mx", "My", "Mz")
_L3 = ("N", "Vy", "Vz", "T", "My", "Mz")


def _names(bases: tuple[str, ...]) -> str:
    return "0." + ",".join(f"{b}_{n}" for n in (1, 2) for b in bases)


def _flat(eids: np.ndarray, salt: float) -> np.ndarray:
    """``(T, E, 12)``, every slot distinct: ``salt + t*1000 + eid*10 + col``."""
    flat = np.empty((_T, eids.size, 12))
    for t in range(_T):
        for i, e in enumerate(eids):
            flat[t, i] = salt + t * 1000 + e * 10 + np.arange(12)
    return flat


def _add_bucket(
    stage: "h5py.Group", frame: str, key: str, eids: np.ndarray,
    flat: np.ndarray, components: str,
) -> None:
    """A single-block 12-column nodal-force bucket, as MPCO writes it."""
    grp = stage["RESULTS/ON_ELEMENTS"].require_group(frame).create_group(key)
    grp.attrs["NUM_COLUMNS"] = np.array([12], dtype=np.int32)
    meta = grp.create_group("META")
    meta.create_dataset("MULTIPLICITY", data=np.array([[1]], dtype=np.int32))
    meta.create_dataset("GAUSS_IDS", data=np.array([[-1]], dtype=np.int32))
    meta.create_dataset("NUM_COMPONENTS",
                        data=np.array([[12]], dtype=np.int32))
    meta.create_dataset("COMPONENTS",
                        data=np.array([components.encode("ascii")]))
    grp.create_dataset("ID", data=eids.reshape(-1, 1).astype(np.int32))
    data = grp.create_group("DATA")
    for t in range(flat.shape[0]):
        ds = data.create_dataset(f"STEP_{t}", data=flat[t])
        ds.attrs["STEP"] = t
        ds.attrs["TIME"] = float(t + 1)


def _file(
    tmp_path: Path, frame: str, keys: tuple[str, str],
    eids: tuple[list[int], list[int]], *, comps1: str | None = None,
) -> Path:
    """Two buckets under one token; ``comps1`` overrides the second
    bucket's COMPONENTS."""
    comps = _names(_G3 if frame == "globalForce" else _L3)
    path = tmp_path / "dup_ids.mpco"
    f, stage_name = _create_skeleton(
        path, node_ids=np.array([1, 2, 3], dtype=np.int32),
        coords=np.zeros((3, 3)), n_steps=_T, dt=1.0,
    )
    try:
        stage = f[stage_name]
        for i, (key, ids) in enumerate(zip(keys, eids)):
            arr = np.asarray(ids, dtype=np.int64)
            _add_bucket(stage, frame, key, arr, _flat(arr, 0.25 * i),
                        comps if i == 0 or comps1 is None else comps1)
    finally:
        f.close()
    return path


def _component(frame: str) -> str:
    token = "global_force" if frame == "globalForce" else "local_force"
    return NODAL_FORCE_CATALOG[("ElasticBeam3d", token)].component_layout[0]


#: (id, frame, the two bucket keys): split by header index, by custom
#: rule, and by class.
_SPLITS = [
    ("header-0-vs-1-global", "globalForce",
     ("5-ElasticBeam3d[1:0:0]", "5-ElasticBeam3d[1:0:1]")),
    ("header-0-vs-1-local", "localForce",
     ("5-ElasticBeam3d[1:0:0]", "5-ElasticBeam3d[1:0:1]")),
    ("custom-rule-1-vs-2-global", "globalForce",
     ("74-ForceBeamColumn3d[1000:1:0]", "74-ForceBeamColumn3d[1000:2:0]")),
    ("custom-rule-1-vs-2-local", "localForce",
     ("74-ForceBeamColumn3d[1000:1:0]", "74-ForceBeamColumn3d[1000:2:0]")),
    ("two-classes-global", "globalForce",
     ("5-ElasticBeam3d[1:0:0]", "74-ForceBeamColumn3d[1000:1:0]")),
]
_SPLIT_ARGS = [s[1:] for s in _SPLITS]
_SPLIT_IDS = [s[0] for s in _SPLITS]


# =====================================================================
# One element in two buckets: the read refuses, by name
# =====================================================================

@pytest.mark.parametrize(("frame", "keys"), _SPLIT_ARGS, ids=_SPLIT_IDS)
def test_duplicate_element_id_refuses_by_name(
    tmp_path: Path, frame: str, keys: tuple[str, str],
) -> None:
    path = _file(tmp_path, frame, keys, ([7, 11], [11, 33]))
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.raises(NodalForceDuplicateElementError) as exc:
            r.elements.get(component=_component(frame))
    msg = str(exc.value)
    assert "element 11 " in msg, msg
    for key in keys:
        assert f"/RESULTS/ON_ELEMENTS/{frame}/{key}" in msg, msg
    assert path.name in msg, msg
    assert isinstance(exc.value, ValueError)


@pytest.mark.parametrize(("frame", "keys"), _SPLIT_ARGS, ids=_SPLIT_IDS)
def test_duplicate_refuses_even_when_the_id_filter_skips_it(
    tmp_path: Path, frame: str, keys: tuple[str, str],
) -> None:
    """The file is corrupt whichever elements a read asks for."""
    path = _file(tmp_path, frame, keys, ([7, 11], [11, 33]))
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.raises(NodalForceDuplicateElementError):
            r.elements.get(component=_component(frame), ids=[7])


def test_message_counts_the_other_repeated_ids(tmp_path: Path) -> None:
    path = _file(tmp_path, "globalForce",
                 ("5-ElasticBeam3d[1:0:0]", "5-ElasticBeam3d[1:0:1]"),
                 ([7, 11, 21, 40], [11, 21, 33, 40]))
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.raises(NodalForceDuplicateElementError,
                           match=r"element 11 .* 2 more element id"):
            r.elements.get(component="nodal_resisting_force_x")


def test_listing_still_names_the_components(tmp_path: Path) -> None:
    """A listing stitches no rows, so it does not refuse."""
    path = _file(tmp_path, "globalForce",
                 ("5-ElasticBeam3d[1:0:0]", "5-ElasticBeam3d[1:0:1]"),
                 ([7, 11], [11, 33]))
    layout = NODAL_FORCE_CATALOG[("ElasticBeam3d", "global_force")]
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        assert set(layout.component_layout) <= set(
            r.elements.available_components())


def test_a_skipped_bucket_is_not_a_duplicate(tmp_path: Path) -> None:
    """A header-1 bucket whose META does not spell the catalog is left
    unread with a warning; it stitches no rows, so element 11 reads once,
    from header 0."""
    swapped = _names(("Py", "Px", "Pz", "Mx", "My", "Mz"))
    path = _file(tmp_path, "globalForce",
                 ("5-ElasticBeam3d[1:0:0]", "5-ElasticBeam3d[1:0:1]"),
                 ([7, 11], [11, 33]), comps1=swapped)
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.warns(NodalForceBucketSkippedWarning):
            slab = r.elements.get(component="nodal_resisting_force_x")
    np.testing.assert_array_equal(slab.element_ids, [7, 11])


# =====================================================================
# Each element once: every element reads its own values, as before
# =====================================================================

@pytest.mark.parametrize(("frame", "keys"), _SPLIT_ARGS, ids=_SPLIT_IDS)
def test_unique_ids_read_as_before(
    tmp_path: Path, frame: str, keys: tuple[str, str],
) -> None:
    eids = ([7, 11], [21, 33])
    path = _file(tmp_path, frame, keys, eids)
    comp = _component(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NodalForceBucketSkippedWarning)
        with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
            slab = r.elements.get(component=comp)
    got = {int(e): slab.values[:, i, :]
           for i, e in enumerate(slab.element_ids)}
    assert sorted(got) == [7, 11, 21, 33]
    for b, ids in enumerate(eids):
        want = _flat(np.asarray(ids, dtype=np.int64), 0.25 * b)
        for i, e in enumerate(ids):
            # Component 0 sits in column 0 at end i and column 6 at end j.
            np.testing.assert_array_equal(
                got[e], want[:, i, [0, 6]], err_msg=f"bucket {b} element {e}")
