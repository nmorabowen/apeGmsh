"""#2363: the repeated-id check scales, and native element-side reads
refuse a repeated element.

1. ``_repeated_rows`` (the check behind ``LadrunoRepeatedNodeId``,
   ``NativeRepeatedNodeId`` and the writer's ``NativeWriteRepeatedId``)
   scanned the whole id list once per repeated id: O(R·N), 26 s for 400k
   ids that all repeat. It now sorts once. Its result must stay exactly
   what it was (the oracle is the old function, copied below), and the
   400k case must finish well under a second.

2. The native ``read_gauss``, ``read_elements``, ``read_line_stations``,
   ``read_fibers`` and ``read_layers`` concatenated each group's
   ``_element_index`` with no repeat check, so one element in two places
   came back as two columns. The writer refuses a repeat within one group
   (#2914) and a second partition (#2918), but takes one element in two
   groups of one partition; hand-made or legacy files reach the other two
   shapes. Each read now refuses all three by name, as the MPCO reader
   does (``MPCOElementInTwoBuckets``).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers import NativeReader
from apeGmsh.results.readers import _native as native_reader
from apeGmsh.results.readers._ladruno import _repeated_rows
from apeGmsh.results.writers import NativeWriter
from apeGmsh.results.writers._native import NativeWriteRepeatedId

_TIME = np.array([0.0, 1.0])
_P0 = "/stages/stage_0/partitions/partition_0/elements"
_P1 = "/stages/stage_0/partitions/partition_1/elements"


# =====================================================================
# 1. _repeated_rows: same answer, O(N log N)
# =====================================================================

def _repeated_rows_before_2363(ids: np.ndarray) -> "dict[int, list[int]]":
    """The function as it was on main df17d674d: the oracle."""
    if ids.size < 2 or bool((ids[1:] > ids[:-1]).all()):
        return {}
    uniq, counts = np.unique(ids, return_counts=True)
    if uniq.size == ids.size:
        return {}
    return {
        int(n): np.flatnonzero(ids == n).tolist() for n in uniq[counts > 1]
    }


def _same(got: dict, want: dict) -> None:
    # Equal, in the same key order, with plain-int keys and rows: the
    # refusal messages print the dict, so these are byte-identical too.
    assert got == want
    assert list(got) == list(want)
    assert repr(got) == repr(want)
    for key, rows in got.items():
        assert type(key) is int
        assert all(type(r) is int for r in rows)


@pytest.mark.parametrize("ids", [
    [], [5], [7, 7], [1, 2, 1, 2], [3, 1, 2], [2, 1, 2, 3, 3],
    [0, 2**63 - 1, -2, 0], [2**63 - 1, -(2**63), 2**63 - 1],
    [4, 4, 4, 4], [9, 8, 7, 9, 8, 7, 9],
])
def test_repeated_rows_matches_the_old_function(ids: list[int]) -> None:
    arr = np.array(ids, dtype=np.int64)
    _same(_repeated_rows(arr), _repeated_rows_before_2363(arr))


def test_repeated_rows_matches_the_old_function_on_random_lists() -> None:
    rng = np.random.default_rng(2363)
    for _ in range(2000):
        n = int(rng.integers(0, 40))
        hi = int(rng.integers(1, 60))
        arr = rng.integers(-5, hi, size=n).astype(np.int64)
        _same(_repeated_rows(arr), _repeated_rows_before_2363(arr))


def _all_duplicated(n: int = 400_000) -> np.ndarray:
    ids = np.repeat(np.arange(n // 2, dtype=np.int64), 2)
    np.random.default_rng(0).shuffle(ids)
    return ids


def test_400k_ids_all_duplicated_finish_well_under_a_second() -> None:
    # The old function took ~26 s here (one scan of 400k ids per each of
    # the 200k repeated ids). The new one sorts once (~0.2 s on a laptop).
    ids = _all_duplicated()
    t0 = time.perf_counter()
    rows = _repeated_rows(ids)
    elapsed = time.perf_counter() - t0
    assert len(rows) == 200_000
    assert all(len(r) == 2 for r in rows.values())
    assert elapsed < 1.0, f"_repeated_rows took {elapsed:.2f} s"


def test_writer_refuses_400k_duplicated_node_ids_well_under_a_second(
    tmp_path: Path,
) -> None:
    # The writer's _refuse_repeated calls the same function.
    ids = _all_duplicated()
    with NativeWriter(tmp_path / "run.h5") as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=_TIME)
        t0 = time.perf_counter()
        with pytest.raises(NativeWriteRepeatedId) as info:
            w.write_nodes(sid, "partition_0", node_ids=ids, components={})
        elapsed = time.perf_counter() - t0
    assert len(info.value.repeated) == 200_000
    assert "and 199995 more" in str(info.value)
    assert elapsed < 1.0, f"the refusal took {elapsed:.2f} s"


# =====================================================================
# 2. Native element-side reads refuse a repeated element
# =====================================================================

def _v(*shape: int) -> np.ndarray:
    return np.arange(
        len(_TIME) * int(np.prod(shape)), dtype=np.float64,
    ).reshape(len(_TIME), *shape)


def _gauss(w, sid, gid, eidx, class_tag=1):
    w.write_gauss_group(
        sid, "partition_0", gid, class_tag=class_tag, int_rule=1,
        element_index=np.array(eidx), natural_coords=np.zeros((1, 2)),
        components={"stress_xx": _v(len(eidx), 1)},
    )


def _elements(w, sid, gid, eidx, class_tag=1):
    w.write_nodal_forces_group(
        sid, "partition_0", gid, class_tag=class_tag, frame="global",
        element_index=np.array(eidx),
        components={"nodal_resisting_force_x": _v(len(eidx), 2)},
    )


def _line_stations(w, sid, gid, eidx, class_tag=1):
    w.write_line_stations_group(
        sid, "partition_0", gid, class_tag=class_tag,
        element_index=np.array(eidx),
        station_natural_coord=np.array([-1.0, 1.0]),
        components={"axial_force": _v(len(eidx), 2)},
    )


def _fibers(w, sid, gid, eidx, class_tag=1):
    # Two fibers per element, at one section.
    e = np.repeat(np.array(eidx), 2)
    n = e.size
    w.write_fibers_group(
        sid, "partition_0", gid, section_tag=class_tag, section_class="Fiber",
        element_index=e, gp_index=np.zeros(n, dtype=np.int64),
        y=np.tile([-0.1, 0.1], len(eidx)), z=np.zeros(n), area=np.ones(n),
        material_tag=np.ones(n, dtype=np.int64),
        components={"fiber_stress": _v(n)},
    )


def _layers(w, sid, gid, eidx, class_tag=1):
    # Two layers per element, at one Gauss point.
    e = np.repeat(np.array(eidx), 2)
    n = e.size
    w.write_layers_group(
        sid, "partition_0", gid,
        element_index=e, gp_index=np.zeros(n, dtype=np.int64),
        layer_index=np.tile([0, 1], len(eidx)),
        sub_gp_index=np.zeros(n, dtype=np.int64),
        thickness=np.ones(n),
        local_axes_quaternion=np.tile([1.0, 0, 0, 0], (n, 1)),
        components={"stress_xx": _v(n)},
    )


def _read(level: str) -> Callable:
    return {
        "gauss": lambda r, **kw: r.read_gauss("stage_0", "stress_xx", **kw),
        "elements": lambda r, **kw: r.read_elements(
            "stage_0", "nodal_resisting_force_x", **kw),
        "line_stations": lambda r, **kw: r.read_line_stations(
            "stage_0", "axial_force", **kw),
        "fibers": lambda r, **kw: r.read_fibers(
            "stage_0", "fiber_stress", **kw),
        "layers": lambda r, **kw: r.read_layers("stage_0", "stress_xx", **kw),
    }[level]


def _eidx(level: str, slab) -> np.ndarray:
    return slab.element_ids if level == "elements" else slab.element_index


#: level -> (group writer, category group name, slab rows per element).
_LEVELS = {
    "gauss": (_gauss, "gauss_points", 1),
    "elements": (_elements, "nodal_forces", 1),
    "line_stations": (_line_stations, "line_stations", 2),
    "fibers": (_fibers, "fibers", 2),
    "layers": (_layers, "layers", 2),
}
#: The levels whose groups store one row per fiber or layer (two per
#: element here), not one per element.
_PER_ROW = {"fibers", "layers"}
#: The levels with one row per element, where a repeat within a group is
#: the element repeating.
_PER_ELEMENT = ["gauss", "elements", "line_stations"]


def _write(path: Path, write, groups: "list[tuple[str, list[int], int]]"):
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=_TIME)
        for gid, eidx, class_tag in groups:
            write(w, sid, gid, eidx, class_tag)
        w.end_stage()


def _set_index(path: Path, dataset: str, values: list[int]) -> None:
    with h5py.File(path, "r+") as f:
        f[dataset][...] = np.array(values, dtype=np.int64)


def _second_partition(path: Path, level: str, elements: list[int]) -> None:
    """Copy ``partition_0``'s group ``g0`` to a ``partition_1`` with raw
    h5py (the writer refuses a second partition, #2918), then set its
    ``elements``: a legacy multi-partition file."""
    cat = _LEVELS[level][1]
    rows = np.array(elements, dtype=np.int64)
    if level in _PER_ROW:
        rows = np.repeat(rows, 2)
    with h5py.File(path, "r+") as f:
        f.create_group(f"{_P1}/{cat}")
        f.copy(f[f"{_P0}/{cat}/g0"], f[f"{_P1}/{cat}"], name="g0")
        f[f"{_P1}/{cat}/g0/_element_index"][...] = rows


def _refused(path: Path, level: str, **kw):
    with NativeReader(path) as r:
        with pytest.raises(ValueError) as info:
            _read(level)(r, **kw)
    err = info.value
    assert isinstance(err, native_reader.NativeRepeatedElementId), err
    return err


@pytest.mark.parametrize("level", _PER_ELEMENT)
def test_an_element_twice_in_one_group_refuses(
    tmp_path: Path, level: str,
) -> None:
    write, cat, _ = _LEVELS[level]
    path = tmp_path / "run.h5"
    _write(path, write, [("g0", [5, 6], 1)])
    dataset = f"{_P0}/{cat}/g0/_element_index"
    _set_index(path, dataset, [5, 5])           # past the writer's check
    err = _refused(path, level)
    assert err.shape == "group"
    assert err.repeated == {5: [(dataset, 0), (dataset, 1)]}
    assert (err.stage_id, err.level, err.file) == ("stage_0", level, path)
    msg = str(err)
    assert "twice in one group" in msg and dataset in msg
    assert "'stage_0'" in msg and level in msg


@pytest.mark.parametrize("level", list(_LEVELS))
def test_an_element_in_two_groups_of_one_partition_refuses(
    tmp_path: Path, level: str,
) -> None:
    # The writer accepts this one: each group is checked on its own.
    write, cat, per = _LEVELS[level]
    path = tmp_path / "run.h5"
    _write(path, write, [("g0", [5, 6], 1), ("g1", [5, 7], 2)])
    err = _refused(path, level)
    assert err.shape == "groups"
    assert err.repeated == {5: [
        (f"{_P0}/{cat}/g0/_element_index", 0),
        (f"{_P0}/{cat}/g1/_element_index", 0),
    ]}
    msg = str(err)
    assert "in two groups of one partition" in msg
    assert f"{cat}/g0/_element_index" in msg
    assert f"{cat}/g1/_element_index" in msg
    # A filter that leaves element 5 out returns the rest, once each.
    with NativeReader(path) as r:
        slab = _read(level)(r, element_ids=np.array([6, 7]))
    assert sorted(set(_eidx(level, slab).tolist())) == [6, 7]
    assert _eidx(level, slab).size == 2 * per


@pytest.mark.parametrize("level", list(_LEVELS))
def test_an_element_in_two_partitions_refuses(
    tmp_path: Path, level: str,
) -> None:
    write, cat, per = _LEVELS[level]
    path = tmp_path / "run.h5"
    _write(path, write, [("g0", [5, 6], 1)])
    _second_partition(path, level, [5, 7])
    err = _refused(path, level)
    assert err.shape == "partitions"
    assert err.repeated == {5: [
        (f"{_P0}/{cat}/g0/_element_index", 0),
        (f"{_P1}/{cat}/g0/_element_index", 0),
    ]}
    msg = str(err)
    assert "in two partitions" in msg
    assert "partition_0" in msg and "partition_1" in msg


@pytest.mark.parametrize("level", list(_LEVELS))
def test_distinct_elements_over_groups_and_partitions_read(
    tmp_path: Path, level: str,
) -> None:
    # The negative control: a legacy multi-partition file and two groups
    # that hold different elements read as before.
    write, cat, per = _LEVELS[level]
    path = tmp_path / "run.h5"
    _write(path, write, [("g0", [5, 6], 1), ("g1", [9], 2)])
    _second_partition(path, level, [7, 8])
    with NativeReader(path) as r:
        slab = _read(level)(r)
    got = _eidx(level, slab)
    assert sorted(set(got.tolist())) == [5, 6, 7, 8, 9]
    assert got.size == 5 * per


def test_one_element_in_two_groups_with_different_components_reads(
    tmp_path: Path,
) -> None:
    # Only the groups a read returns are checked: element 5 has its
    # stress in one group and its strain in another.
    path = tmp_path / "run.h5"
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=_TIME)
        for gid, comp in (("g0", "stress_xx"), ("g1", "strain_xx")):
            w.write_gauss_group(
                sid, "partition_0", gid, class_tag=1, int_rule=1,
                element_index=np.array([5]), natural_coords=np.zeros((1, 2)),
                components={comp: _v(1, 1)},
            )
        w.end_stage()
    with NativeReader(path) as r:
        for comp in ("stress_xx", "strain_xx"):
            np.testing.assert_array_equal(
                r.read_gauss("stage_0", comp).element_index, [5])


def test_fibers_repeat_their_element_within_a_group_by_design(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.h5"
    _write(path, _fibers, [("g0", [2, 3], 1)])
    with NativeReader(path) as r:
        slab = r.read_fibers("stage_0", "fiber_stress")
    np.testing.assert_array_equal(slab.element_index, [2, 2, 3, 3])


def test_a_repeated_layer_row_within_a_group_refuses(tmp_path: Path) -> None:
    # Element 5's two layers both claim layer 0: the row key
    # (element, Gauss point, layer, sub-point) repeats.
    path = tmp_path / "run.h5"
    _write(path, _layers, [("g0", [5, 6], 1)])
    _set_index(path, f"{_P0}/layers/g0/_layer_index", [0, 0, 0, 1])
    err = _refused(path, "layers")
    assert err.shape == "group"
    assert err.repeated == {
        (5, 0, 0, 0): [(f"{_P0}/layers/g0", 0), (f"{_P0}/layers/g0", 1)],
    }
    assert "layer row id(s)" in str(err)


def test_message_shows_five_repeats_then_counts_the_rest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.h5"
    ids = list(range(1, 8))
    _write(path, _gauss, [("g0", ids, 1), ("g1", ids, 2)])
    err = _refused(path, "gauss")
    assert list(err.repeated) == ids
    assert "; and 2 more" in str(err)
