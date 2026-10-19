"""An MPCO nodal-force read refuses an element id held by two buckets (#2173).

MPCO writes each element once, to exactly one bucket of a token. A file
where one element id sits in two buckets, or twice in one bucket's ``ID``,
is corrupt: the reader used to stitch the rows and return that element
twice, with no warning. It now raises :class:`MPCOElementInTwoBuckets`,
naming the id, the bucket path(s) and the file. The check is filter-aware:
it refuses only for ids the read returns.

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
from apeGmsh.results.readers import _mpco_nodal_io as _mn
from apeGmsh.results.readers._mpco_nodal_io import (
    NodalForceBucketSkippedWarning,
)
from apeGmsh.results.readers._mpco_refusals import (
    MPCOElementInTwoBuckets,
    refuse_element_in_two_buckets,
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
        with pytest.raises(MPCOElementInTwoBuckets) as exc:
            r.elements.get(component=_component(frame))
    msg = str(exc.value)
    assert "element 11 " in msg, msg
    for key in keys:
        assert f"/RESULTS/ON_ELEMENTS/{frame}/{key}" in msg, msg
    assert path.name in msg, msg
    assert isinstance(exc.value, ValueError)


@pytest.mark.parametrize(("frame", "keys"), _SPLIT_ARGS, ids=_SPLIT_IDS)
def test_id_filter_that_skips_the_duplicate_reads(
    tmp_path: Path, frame: str, keys: tuple[str, str],
) -> None:
    """The check is filter-aware: a read that returns elements 7 and 33
    alone returns each once, with its own values; asking for 11 refuses."""
    path = _file(tmp_path, frame, keys, ([7, 11], [11, 33]))
    comp = _component(frame)
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        slab = r.elements.get(component=comp, ids=[7, 33])
        with pytest.raises(MPCOElementInTwoBuckets, match="element 11 "):
            r.elements.get(component=comp, ids=[11])
    np.testing.assert_array_equal(slab.element_ids, [7, 33])
    want7 = _flat(np.array([7, 11]), 0.0)[:, 0, [0, 6]]
    want33 = _flat(np.array([11, 33]), 0.25)[:, 1, [0, 6]]
    np.testing.assert_array_equal(slab.values[:, 0, :], want7)
    np.testing.assert_array_equal(slab.values[:, 1, :], want33)


# =====================================================================
# One element twice in one bucket: the read refuses, by name
# =====================================================================

def _one_bucket_file(tmp_path: Path, eids: list[int]) -> tuple[Path, str]:
    key = "5-ElasticBeam3d[1:0:0]"
    path = tmp_path / "dup_in_one.mpco"
    f, stage_name = _create_skeleton(
        path, node_ids=np.array([1, 2, 3], dtype=np.int32),
        coords=np.zeros((3, 3)), n_steps=_T, dt=1.0,
    )
    try:
        arr = np.asarray(eids, dtype=np.int64)
        _add_bucket(f[stage_name], "globalForce", key, arr,
                    _flat(arr, 0.0), _names(_G3))
    finally:
        f.close()
    return path, key


def test_repeat_inside_one_bucket_refuses_by_name(tmp_path: Path) -> None:
    path, key = _one_bucket_file(tmp_path, [7, 11, 11])
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.raises(MPCOElementInTwoBuckets) as exc:
            r.elements.get(component="nodal_resisting_force_x")
    msg = str(exc.value)
    assert "element 11 appears twice in the MPCO bucket " in msg, msg
    assert f"/RESULTS/ON_ELEMENTS/globalForce/{key}" in msg, msg
    assert path.name in msg, msg


def test_repeat_inside_one_bucket_is_filter_aware(tmp_path: Path) -> None:
    path, _key = _one_bucket_file(tmp_path, [7, 11, 11])
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        slab = r.elements.get(component="nodal_resisting_force_x", ids=[7])
    np.testing.assert_array_equal(slab.element_ids, [7])


# =====================================================================
# The helper and the per-read ledger
# =====================================================================

def test_helper_names_both_paths_and_the_file() -> None:
    with pytest.raises(MPCOElementInTwoBuckets) as exc:
        refuse_element_in_two_buckets(
            {"/a": np.array([3, 9]), "/b": np.array([5, 9, 3])},
            file="f.mpco",
        )
    msg = str(exc.value)
    assert msg.startswith("f.mpco: element 3 appears in two MPCO buckets: "
                          "/a and /b."), msg
    assert "1 more element id(s)" in msg, msg
    refuse_element_in_two_buckets(
        {"/a": np.array([3, 9]), "/b": np.array([5])}, file="f.mpco")
    refuse_element_in_two_buckets({}, file="f.mpco")


def test_a_scan_serves_one_read(tmp_path: Path) -> None:
    """The ledger is per scan: reading a bucket twice from one scan is a
    caller error, raised by name rather than refused as a duplicate."""
    path, _key = _one_bucket_file(tmp_path, [7, 11])
    with h5py.File(path, "r") as f:
        on_el = f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"]
        token, buckets, _ = _mn.scan_nodal_force_buckets(
            on_el, canonical_component="nodal_resisting_force_x")
        grp = on_el[token][buckets[0].bracket_key]
        kw = dict(t_idx=np.arange(_T), element_ids=None)
        _mn.read_nodal_bucket_slab(
            grp, buckets[0], "nodal_resisting_force_x", **kw)
        with pytest.raises(RuntimeError, match="read twice from one scan"):
            _mn.read_nodal_bucket_slab(
                grp, buckets[0], "nodal_resisting_force_x", **kw)


def test_message_counts_the_other_repeated_ids(tmp_path: Path) -> None:
    path = _file(tmp_path, "globalForce",
                 ("5-ElasticBeam3d[1:0:0]", "5-ElasticBeam3d[1:0:1]"),
                 ([7, 11, 21, 40], [11, 21, 33, 40]))
    with Results.from_mpco(path, model_h5=_stub_model_h5_path()) as r:
        with pytest.raises(MPCOElementInTwoBuckets,
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
