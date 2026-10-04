"""``.ladruno`` reader against the fork's WP-163/164/165 writer changes.

Three behaviours, on synthetic files written with ``h5py`` (no fork
build, no MPI run):

* **Hyperslab reads.** ``DATA[T × nIds × nComp]`` is read only where the
  caller asked (steps, ids, components), and the values equal the same
  selection taken from a full read.
* **Part-set validation.** The ``.part-N`` files of one set must agree
  on ``NUM_PARTITIONS`` (= the file count, ``PARTITION_ID`` = 0..N-1) and,
  when ``RUN_ID_SCOPE`` is not ``"process"``, on ``RUN_ID``. Stale parts
  from an earlier run are refused; files that predate ``RUN_ID`` pass.
* **EMPTY_PARTITION.** A stage marked ``EMPTY_PARTITION = 1`` has
  zero-length ``MODEL/NODES`` and no result groups; it stitches as an
  empty contribution.

Contract: fork ``Ladruno_implementation/ladruno_schema_v1.md`` §2 (INFO
``RUN_ID`` / ``RUN_ID_SCOPE`` rows, the ``EMPTY_PARTITION`` paragraph) and
``ladruno_apegmsh_contract.md`` ("Validate the part set (WP-165)").
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers import _ladruno_hyperslab as hs
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_multi import (
    LadrunoMultiPartitionReader,
    StageOrderMatchWarning,
)

_T = 6
_TIME = np.linspace(0.1, 0.6, _T)


def _disp(ids: np.ndarray) -> np.ndarray:
    """Deterministic, entry-unique DATA[T, n, 3] for the given node ids."""
    t = np.arange(_T, dtype=np.float64)[:, None, None]
    n = ids.astype(np.float64)[None, :, None]
    c = np.arange(3, dtype=np.float64)[None, None, :]
    return 1000.0 * t + n + 0.1 * c


def _write_part(
    path: Path, node_ids, *,
    part: Optional[int] = None, num_parts: Optional[int] = None,
    run_id: Optional[str] = None, run_scope: Optional[str] = None,
    empty: bool = False, chunks=None, stamps=(1,),
) -> Path:
    """One part file. Stage ``k`` (in ``stamps`` order) holds
    ``_disp(ids) + 1e6 * k`` so a test can tell the stages apart."""
    ids = np.asarray(node_ids, dtype=np.int64)
    with h5py.File(path, "w") as f:
        info = f.create_group("INFO")
        info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
        info.attrs["FORMAT_VERSION"] = 1
        info.attrs["SPATIAL_DIM"] = 3
        if num_parts is not None:
            info.attrs["PARTITIONED"] = 1
            info.attrs["PARTITION_ID"] = part
            info.attrs["NUM_PARTITIONS"] = num_parts
        if run_id is not None:
            info.attrs["RUN_ID"] = np.bytes_(run_id.encode())
        if run_scope is not None:
            info.attrs["RUN_ID_SCOPE"] = np.bytes_(run_scope.encode())
        for k, stamp in enumerate(stamps):
            stage = f.create_group(f"MODEL_STAGE[{stamp}]")
            stage.attrs["KIND"] = np.bytes_(b"static")
            nodes = stage.create_group("MODEL/NODES")
            nodes.create_dataset("ID", data=ids.reshape(-1, 1))
            nodes.create_dataset(
                "COORDINATES",
                data=np.column_stack(
                    [ids, np.zeros_like(ids), np.zeros_like(ids)],
                ).astype(np.float64).reshape(-1, 3),
            )
            stage.create_group("MODEL/ELEMENTS")
            if empty:
                # WP-165 MP-8: zero-length NODES, no node/element results.
                stage.attrs["EMPTY_PARTITION"] = 1
                continue
            res = stage.create_group("RESULTS/ON_NODES/DISPLACEMENT")
            res.attrs["COMPONENTS"] = np.array([b"Ux,Uy,Uz"])
            res.create_dataset("ID", data=ids.reshape(-1, 1))
            res.create_dataset(
                "DATA", data=_disp(ids) + 1e6 * k, chunks=chunks,
            )
            res.create_dataset("TIME", data=_TIME)
            res.create_dataset("STEP", data=np.arange(_T))
    return path


# ---------------------------------------------------------------------
# A. Hyperslab reads
# ---------------------------------------------------------------------

_SELECTIONS = [
    (np.arange(_T), None, None),
    (np.array([5]), np.array([7]), np.array([1])),
    (np.array([4, 1, 1]), np.array([0, 3, 99]), np.array([2, 0])),  # sparse rows
    (np.array([0, 2]), np.arange(10, 60), np.array([0, 1, 2])),     # dense rows
    (np.arange(_T), np.array([42, 3, 42]), None),                   # unsorted, repeat
    (np.array([], dtype=np.int64), np.array([1]), None),
]


@pytest.mark.parametrize("t_idx, rows, cols", _SELECTIONS)
def test_hyperslab_equals_full_read(tmp_path: Path, t_idx, rows, cols) -> None:
    ids = np.arange(1, 101)
    p = _write_part(tmp_path / "one.ladruno", ids, chunks=(2, 16, 3))
    with h5py.File(p, "r") as f:
        ds = f["MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA"]
        full = ds[...]
        got = hs.read_hyperslab(ds, t_idx, rows, cols)
    ref = full[t_idx]
    if rows is not None:
        ref = ref[:, rows]
    if cols is not None:
        ref = ref[:, :, cols]
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)


def test_read_nodes_matches_full_read(tmp_path: Path) -> None:
    ids = np.arange(1, 201)
    p = _write_part(tmp_path / "one.ladruno", ids, chunks=(_T, 32, 3))
    full = _disp(ids)
    with LadrunoReader(p) as r:
        all_slab = r.read_nodes("stage_0", "displacement_y")
        np.testing.assert_array_equal(all_slab.values, full[:, :, 1])
        np.testing.assert_array_equal(all_slab.node_ids, ids)
        # Unsorted request: the reader keeps file order, as before.
        sub = r.read_nodes(
            "stage_0", "displacement_z",
            node_ids=np.array([150, 3, 77]), time_slice=[4, 0],
        )
        assert sub.node_ids.tolist() == [3, 77, 150]
        rows = np.array([2, 76, 149])
        np.testing.assert_array_equal(sub.values, full[[4, 0]][:, rows, 2])
        np.testing.assert_allclose(sub.time, _TIME[[4, 0]])


def test_single_node_read_does_not_load_full_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One node's history reads one id column, not ``T × nIds × nComp``."""
    ids = np.arange(1, 5001)
    p = _write_part(tmp_path / "big.ladruno", ids, chunks=(_T, 256, 3))
    blocks: list[tuple[int, ...]] = []
    real = hs._read_block

    def spy(ds, key):
        out = real(ds, key)
        blocks.append(out.shape)
        return out

    monkeypatch.setattr(hs, "_read_block", spy)
    with LadrunoReader(p) as r:
        slab = r.read_nodes(
            "stage_0", "displacement_x", node_ids=np.array([2500]),
        )
    np.testing.assert_array_equal(slab.values[:, 0], _disp(ids)[:, 2499, 0])
    assert blocks == [(_T, 1, 1)]


# ---------------------------------------------------------------------
# B. Part-set validation
# ---------------------------------------------------------------------

def _pair(tmp_path: Path, **kw) -> list[Path]:
    """Two parts; per-part overrides as ``<attr>0`` / ``<attr>1``."""
    out = []
    for k in (0, 1):
        opts = {
            "part": k, "num_parts": 2,
            "run_id": "slurm-81234.0", "run_scope": "launcher",
        }
        for name in ("num_parts", "run_id", "run_scope", "part"):
            if f"{name}{k}" in kw:
                opts[name] = kw[f"{name}{k}"]
        ids = [1, 2, 3] if k == 0 else [3, 4, 5]
        out.append(_write_part(tmp_path / f"run.part-{k}.ladruno", ids, **opts))
    return out


def test_matching_part_set_passes(tmp_path: Path) -> None:
    with LadrunoMultiPartitionReader(_pair(tmp_path)) as r:
        assert r.read_nodes("stage_0", "displacement_x").node_ids.tolist() == [
            1, 2, 3, 4, 5,
        ]


def test_mismatched_run_id_is_refused(tmp_path: Path) -> None:
    paths = _pair(tmp_path, run_id1="slurm-79999.0")
    with pytest.raises(ValueError, match="RUN_ID differs") as exc:
        LadrunoMultiPartitionReader(paths)
    assert "run.part-0.ladruno" in str(exc.value)
    assert "run.part-1.ladruno" in str(exc.value)


def test_mismatched_num_partitions_is_refused(tmp_path: Path) -> None:
    paths = _pair(tmp_path, num_parts1=3)
    with pytest.raises(ValueError, match="NUM_PARTITIONS") as exc:
        LadrunoMultiPartitionReader(paths)
    assert "run.part-1.ladruno=3" in str(exc.value)


def test_num_partitions_must_equal_file_count(tmp_path: Path) -> None:
    """Both parts say 3 but only 2 were found: part 2 is missing."""
    paths = _pair(tmp_path, num_parts0=3, num_parts1=3)
    with pytest.raises(ValueError, match="NUM_PARTITIONS"):
        LadrunoMultiPartitionReader(paths)


def test_duplicate_partition_id_is_refused(tmp_path: Path) -> None:
    paths = _pair(tmp_path, part1=0)
    with pytest.raises(ValueError, match="PARTITION_ID"):
        LadrunoMultiPartitionReader(paths)


def test_process_scope_skips_run_id_check(tmp_path: Path) -> None:
    """``process`` RUN_IDs are per-process by design: not cross-checked."""
    paths = _pair(
        tmp_path, run_id0="pid-1", run_id1="pid-2",
        run_scope0="process", run_scope1="process",
    )
    with LadrunoMultiPartitionReader(paths) as r:
        assert r.partitions("stage_0") == ["partition_0", "partition_1"]


def test_files_without_run_id_pass(tmp_path: Path) -> None:
    """Files written before fork WP-165 carry no RUN_ID: no check."""
    paths = _pair(
        tmp_path, run_id0=None, run_id1=None, run_scope0=None, run_scope1=None,
    )
    with LadrunoMultiPartitionReader(paths) as r:
        assert r.read_nodes("stage_0", "displacement_x").node_ids.size == 5


def test_partition_manifest_reports_run_identity(tmp_path: Path) -> None:
    p0, _ = _pair(tmp_path)
    with LadrunoReader(p0) as r:
        m = r.partition_manifest()
    assert m["NUM_PARTITIONS"] == 2 and m["PARTITION_ID"] == 0
    assert m["RUN_ID"] == "slurm-81234.0"
    assert m["RUN_ID_SCOPE"] == "launcher"


# ---------------------------------------------------------------------
# C. EMPTY_PARTITION
# ---------------------------------------------------------------------

@pytest.mark.parametrize("empty_part", [0, 1])
def test_empty_partition_stitches_as_empty(tmp_path: Path, empty_part: int) -> None:
    ids = np.array([10, 11, 12])
    paths = []
    for k in (0, 1):
        paths.append(_write_part(
            tmp_path / f"reg.part-{k}.ladruno",
            [] if k == empty_part else ids,
            part=k, num_parts=2, run_id="u-1", run_scope="user",
            empty=(k == empty_part),
        ))
    with LadrunoMultiPartitionReader(paths) as r:
        assert r.stages()[0].n_steps == _T
        np.testing.assert_allclose(r.time_vector("stage_0"), _TIME)
        slab = r.read_nodes("stage_0", "displacement_x")
        assert slab.node_ids.tolist() == ids.tolist()
        np.testing.assert_array_equal(slab.values, _disp(ids)[:, :, 0])
        fem = r.fem()
        assert fem is not None
        assert sorted(int(n) for n in fem.nodes.ids) == ids.tolist()
        assert r.opensees_model() is not None


def test_empty_partition_alone(tmp_path: Path) -> None:
    p = _write_part(tmp_path / "e.ladruno", [], empty=True)
    with LadrunoReader(p) as r:
        assert r.is_empty_partition("stage_0")
        assert r.fem() is None
        assert r.read_nodes("stage_0", "displacement_x").node_ids.size == 0


# ---------------------------------------------------------------------
# Stage pairing across parts (WP-165: rank-local stamps may differ)
# ---------------------------------------------------------------------

def _staged_pair(tmp_path: Path, stamps0, stamps1, *, empty1: bool = False):
    ids0, ids1 = np.array([1, 2, 3]), np.array([3, 4])
    return [
        _write_part(
            tmp_path / "st.part-0.ladruno", ids0, part=0, num_parts=2,
            stamps=stamps0,
        ),
        _write_part(
            tmp_path / "st.part-1.ladruno", [] if empty1 else ids1,
            part=1, num_parts=2, stamps=stamps1, empty=empty1,
        ),
    ]


def _check_stitched_stages(r, n_stages: int) -> None:
    ids = np.array([1, 2, 3, 4])
    for k in range(n_stages):
        slab = r.read_nodes(f"stage_{k}", "displacement_x")
        assert slab.node_ids.tolist() == ids.tolist()
        np.testing.assert_array_equal(slab.values, _disp(ids)[:, :, 0] + 1e6 * k)


def test_same_stage_names_pair_by_name_without_warning(tmp_path: Path) -> None:
    import warnings

    paths = _staged_pair(tmp_path, (1, 4), (1, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", StageOrderMatchWarning)
        with LadrunoMultiPartitionReader(paths) as r:
            assert [s.name for s in r.stages()] == [
                "MODEL_STAGE[1]", "MODEL_STAGE[4]",
            ]
            _check_stitched_stages(r, 2)


@pytest.mark.parametrize("stamps0, stamps1", [
    ((1, 4), (1, 7)),
    ((2, 10), (2, 11)),   # lexicographic order would put [10] before [2]
    ((2, 10), (3, 9)),
])
def test_different_stage_names_pair_by_order_with_warning(
    tmp_path: Path, stamps0, stamps1,
) -> None:
    paths = _staged_pair(tmp_path, stamps0, stamps1)
    with pytest.warns(StageOrderMatchWarning, match="matched by order") as rec:
        r = LadrunoMultiPartitionReader(paths)
    msg = str(rec[0].message)
    assert "st.part-0.ladruno" in msg and "st.part-1.ladruno" in msg
    with r:
        # Canonical names: the first non-empty part's (part 0 here).
        assert [s.name for s in r.stages()] == [
            f"MODEL_STAGE[{s}]" for s in stamps0
        ]
        _check_stitched_stages(r, 2)


def test_order_pairing_canonical_name_skips_empty_part(tmp_path: Path) -> None:
    """Part 0 empty in every stage: names come from part 1."""
    ids1 = np.array([5, 6])
    paths = [
        _write_part(tmp_path / "e.part-0.ladruno", [], part=0, num_parts=2,
                    stamps=(2, 10), empty=True),
        _write_part(tmp_path / "e.part-1.ladruno", ids1, part=1, num_parts=2,
                    stamps=(2, 11)),
    ]
    with pytest.warns(StageOrderMatchWarning):
        r = LadrunoMultiPartitionReader(paths)
    with r:
        assert [s.name for s in r.stages()] == [
            "MODEL_STAGE[2]", "MODEL_STAGE[11]",
        ]
        slab = r.read_nodes("stage_1", "displacement_x")
        assert slab.node_ids.tolist() == [5, 6]
        np.testing.assert_array_equal(slab.values, _disp(ids1)[:, :, 0] + 1e6)


def test_different_stage_counts_are_refused(tmp_path: Path) -> None:
    paths = _staged_pair(tmp_path, (1, 4), (1, 4, 9))
    with pytest.raises(ValueError, match="different stage counts") as exc:
        LadrunoMultiPartitionReader(paths)
    msg = str(exc.value)
    assert "st.part-0.ladruno" in msg and "MODEL_STAGE[9]" in msg
