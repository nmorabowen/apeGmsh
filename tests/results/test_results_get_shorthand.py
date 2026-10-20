"""A composite ``get()`` refuses a name the stage does not record (#2343).

Every reader answers an unknown component with an EMPTY slab. Before
#2343 the composites passed that through, so a family shorthand
(``"displacement"`` for ``displacement_x/_y/_z``) or a typo read as "no
data" on a populated file. The composites now raise
:class:`ComponentNotRecorded` by name and list what the stage records.
A known name on an empty selection still reads as empty: that is a real
answer, not a missing one.

Oracles: the written values (native, MPCO) and the committed ``.ladruno``
fixture's own explicit reads, at ``rtol=1e-12``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results._composites import ComponentNotRecorded
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5, _stub_model_h5_path
from tests.test_results_mpco_multi_mock import _create_partition

LADRUNO = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"

_NODE_IDS = np.array([1, 2, 3], dtype=np.int64)
_TIME = np.array([0.0, 0.5, 1.0])


def _disp(axis: int) -> np.ndarray:
    """(T, N) displacement values, distinct per axis, step and node."""
    t = np.arange(_TIME.size, dtype=np.float64)[:, None]
    n = _NODE_IDS.astype(np.float64)[None, :]
    return 0.1 * (axis + 1) + t * 1.5 + n * 0.01


def _write_native(tmp_path: Path) -> Path:
    """Three stages: ``full`` records x/y/z, ``x_only`` records only x,
    and one mode stage records x/y."""
    path = tmp_path / "shorthand.h5"
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="full", kind="static", time=_TIME)
        w.write_nodes(sid, "partition_0", node_ids=_NODE_IDS, components={
            "displacement_x": _disp(0),
            "displacement_y": _disp(1),
            "displacement_z": _disp(2),
        })
        w.end_stage()
        sid = w.begin_stage(name="x_only", kind="static", time=_TIME)
        w.write_nodes(sid, "partition_0", node_ids=_NODE_IDS, components={
            "displacement_x": _disp(0),
        })
        w.end_stage()
        sid = w.begin_stage(
            name="mode_1", kind="mode", time=np.array([0.0]),
            eigenvalue=4.0, frequency_hz=1.0 / np.pi, period_s=np.pi,
            mode_index=1,
        )
        w.write_nodes(sid, "partition_0", node_ids=_NODE_IDS, components={
            "displacement_x": _disp(0)[:1],
            "displacement_y": _disp(1)[:1],
        })
        w.end_stage()
    return path


@pytest.fixture
def native(tmp_path: Path):
    path = _write_native(tmp_path)
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        yield r


# ---------------------------------------------------------------------
# Native
# ---------------------------------------------------------------------

def test_native_family_shorthand_refuses_by_name(native) -> None:
    with pytest.raises(ComponentNotRecorded) as exc:
        native.nodes.get(component="displacement", stage="full")
    msg = str(exc.value)
    assert "'displacement' is a family name" in msg
    assert "displacement_x, displacement_y, displacement_z" in msg


def test_native_refusal_is_a_value_error(native) -> None:
    # Callers that already catch ValueError (node_envelope's refusal,
    # the census) keep one except clause for every refusal by name.
    with pytest.raises(ValueError):
        native.nodes.get(component="displacment_x", stage="full")


def test_native_explicit_components_read_the_written_values(native) -> None:
    for axis, comp in enumerate(
        ("displacement_x", "displacement_y", "displacement_z"),
    ):
        slab = native.nodes.get(component=comp, stage="full")
        np.testing.assert_array_equal(slab.node_ids, _NODE_IDS)
        np.testing.assert_allclose(slab.values, _disp(axis), rtol=1e-12)


def test_native_listing_is_the_stage_being_read(native) -> None:
    # ``full`` records displacement_y; ``x_only`` does not. A union over
    # stages would hide the gap and hand back an empty slab.
    assert native.nodes.get(
        component="displacement_y", stage="full",
    ).values.shape == (3, 3)
    with pytest.raises(ComponentNotRecorded) as exc:
        native.nodes.get(component="displacement_y", stage="x_only")
    assert "stage" in str(exc.value)
    assert "['displacement_x']" in str(exc.value)


def test_native_mode_stage(native) -> None:
    mode = native.modes[0]
    np.testing.assert_allclose(
        mode.nodes.get(component="displacement_y").values,
        _disp(1)[:1], rtol=1e-12,
    )
    with pytest.raises(ComponentNotRecorded, match="family name"):
        mode.nodes.get(component="displacement")


def test_native_known_name_on_empty_selection_stays_empty(native) -> None:
    slab = native.nodes.get(
        component="displacement_x", ids=[999_999], stage="full",
    )
    assert slab.values.shape == (3, 0)
    assert slab.node_ids.size == 0


@pytest.mark.parametrize(
    "level", ["gauss", "line_stations", "fibers", "layers", "springs"],
)
def test_native_element_levels_refuse_an_unrecorded_name(
    native, level: str,
) -> None:
    composite = getattr(native.elements, level)
    with pytest.raises(ComponentNotRecorded, match=level):
        composite.get(component="stress", stage="full")


def test_native_per_element_node_level_refuses(native) -> None:
    with pytest.raises(ComponentNotRecorded, match="elements level"):
        native.elements.get(component="globalForce", stage="full")


# ---------------------------------------------------------------------
# .ladruno (committed fork fixtures)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["truss2d", "beam3d"])
def test_ladruno_family_shorthand_refuses_by_name(name: str) -> None:
    r = Results.from_ladruno(LADRUNO / f"{name}.ladruno")
    comps = [c for c in r.nodes.available_components()
             if c.startswith("displacement_")]
    assert comps, "fixture must record displacement components"
    with pytest.raises(ComponentNotRecorded) as exc:
        r.nodes.get(component="displacement")
    assert ", ".join(comps) in str(exc.value)
    # The explicit components still read, and are not empty.
    for comp in comps:
        slab = r.nodes.get(component=comp)
        assert slab.values.shape[1] > 0
        assert slab.component == comp


def test_ladruno_line_stations_refuse_an_unrecorded_name() -> None:
    r = Results.from_ladruno(LADRUNO / "beam3d.ladruno")
    avail = r.elements.line_stations.available_components()
    assert "bending_moment_y" in avail
    with pytest.raises(ComponentNotRecorded) as exc:
        r.elements.line_stations.get(component="bending_moment")
    assert "bending_moment_y" in str(exc.value)


def test_ladruno_spatial_query_refuses_through_get() -> None:
    # nearest_to / in_box / in_sphere / on_plane all read through get().
    r = Results.from_ladruno(LADRUNO / "truss2d.ladruno")
    with pytest.raises(ComponentNotRecorded):
        r.nodes.in_box(
            (-1e9, -1e9, -1e9), (1e9, 1e9, 1e9), component="displacement",
        )
    with pytest.raises(ComponentNotRecorded):
        r.nodes.nearest_to((0.0, 0.0, 0.0), component="displacement")


def test_ladruno_known_name_on_empty_selection_stays_empty() -> None:
    r = Results.from_ladruno(LADRUNO / "truss2d.ladruno")
    slab = r.nodes.get(component="displacement_x", ids=[999_999])
    assert slab.values.shape[1] == 0


# ---------------------------------------------------------------------
# MPCO (synthetic, via model_h5=): single file and two partitions
# ---------------------------------------------------------------------

def _disp_fn(k: int, nid: int) -> list[float]:
    return [k * 100.0 + nid, k * 10.0 + nid, k + nid * 0.5]


def _stress_fn(k: int, eid: int) -> list[float]:
    return [k + eid + ax * 0.25 for ax in range(6)]


def _mpco_partition(path: Path, node_ids, elem_id: int) -> None:
    _create_partition(
        path,
        node_ids=np.asarray(node_ids, dtype=np.int64),
        coords=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64,
        ),
        element_ids=np.array([elem_id], dtype=np.int64),
        element_connectivity=np.asarray([node_ids], dtype=np.int64),
        n_steps=3, dt=0.1,
        disp_value_fn=_disp_fn, stress_value_fn=_stress_fn,
    )


@pytest.fixture
def mpco_single(tmp_path: Path):
    path = tmp_path / "single.mpco"
    _mpco_partition(path, [1, 2, 3, 4], 10)
    with Results.from_mpco(str(path), model_h5=_stub_model_h5_path()) as r:
        yield r


@pytest.fixture
def mpco_two_parts(tmp_path: Path):
    p0 = tmp_path / "Recorder.part-0.mpco"
    p1 = tmp_path / "Recorder.part-1.mpco"
    _mpco_partition(p0, [1, 2, 3, 4], 10)
    _mpco_partition(p1, [4, 5, 6, 7], 20)
    with Results.from_mpco(str(p0), model_h5=_stub_model_h5_path()) as r:
        yield r


@pytest.mark.parametrize("which", ["mpco_single", "mpco_two_parts"])
def test_mpco_node_shorthand_refuses_and_components_read(
    which: str, request,
) -> None:
    r = request.getfixturevalue(which)
    with pytest.raises(ComponentNotRecorded) as exc:
        r.nodes.get(component="displacement")
    assert "displacement_x, displacement_y, displacement_z" in str(exc.value)
    for axis, comp in enumerate(
        ("displacement_x", "displacement_y", "displacement_z"),
    ):
        slab = r.nodes.get(component=comp)
        want = np.array([
            [_disp_fn(k, int(n))[axis] for n in slab.node_ids]
            for k in range(3)
        ])
        np.testing.assert_allclose(slab.values, want, rtol=1e-12)


@pytest.mark.parametrize("which", ["mpco_single", "mpco_two_parts"])
def test_mpco_gauss_shorthand_refuses_and_derived_still_reads(
    which: str, request,
) -> None:
    r = request.getfixturevalue(which)
    with pytest.raises(ComponentNotRecorded) as exc:
        r.elements.gauss.get(component="stress")
    assert "stress_xx" in str(exc.value)
    xx = r.elements.gauss.get(component="stress_xx")
    want = np.array([
        [_stress_fn(k, int(e))[0] for e in xx.element_index]
        for k in range(3)
    ])
    np.testing.assert_allclose(xx.values, want, rtol=1e-12)
    # Derived scalars resolve before the reader and are not refused.
    vm = r.elements.gauss.get(component="von_mises_stress", plane=None)
    s = np.array([
        [_stress_fn(k, int(e)) for e in vm.element_index] for k in range(3)
    ])                                      # (T, GP, 6): xx yy zz xy yz xz
    sxx, syy, szz, sxy, syz, sxz = np.moveaxis(s, -1, 0)
    closed = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
    )
    np.testing.assert_allclose(vm.values, closed, rtol=1e-12)


def test_mpco_filtered_refusal_adds_no_skip_warning(mpco_single) -> None:
    # #2305: a filtered read names only the skipped buckets it asked
    # for. The refusal's listing must not add warnings of its own.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ComponentNotRecorded):
            mpco_single.elements.gauss.get(component="stress", ids=[10])
    assert [str(w.message) for w in caught] == []


def test_mpco_known_name_on_empty_selection_stays_empty(mpco_single) -> None:
    slab = mpco_single.elements.gauss.get(
        component="stress_xx", ids=[999_999],
    )
    assert slab.values.shape[1] == 0
