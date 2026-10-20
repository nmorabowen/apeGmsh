"""#2221 item 1: a Gauss read joining 2-D- and 3-D-parent classes refuses
by name in all four readers.

A :class:`GaussSlab` carries one ``natural_coords`` array, ``(sum_GP,
dim)``. A read that joins a quad (``(xi, eta)``) with a tet (``(xi, eta,
zeta)``) has no faithful slab, so (maintainer ruling on #2221, option A)
every reader refuses it, naming one source per width and the file:

* MPCO single-file raised :class:`MPCOGaussDimensionMixed` already (#2165);
* MPCO partitioned padded the quad rows with NaN (``[xi, eta, NaN]``), so
  the answer depended on how the model was partitioned;
* ``.ladruno`` replaced every quad row with NaN, and the world coordinates
  and extrapolation downstream failed with no word;
* native failed with numpy's bare ``ValueError`` from ``np.concatenate``.

A read narrowed to one class answers in every reader, unpadded.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._gauss_refusals import GaussDimensionMixed
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._mpco import MPCOGaussDimensionMixed
from apeGmsh.results.readers._native import NativeReader
from apeGmsh.results.writers import NativeWriter
from tests.results.test_mpco_mixed_dim_gauss import (
    _QUAD,
    _TET,
    _gauss,
    _stress_file,
)

_FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"
_STAGE = "MODEL_STAGE[1]"
_TET_MODEL = "33001-BezierTet10[1000:1]"
_TET_BUCKET = "33001-BezierTet10[1000:1:0]"
_QUAD_BUCKET = "31-FourNodeQuad[201:0:0]"


# =====================================================================
# MPCO single file (already refused; stays green)
# =====================================================================

def test_mpco_single_still_refuses(tmp_path: Path) -> None:
    path = _stress_file(tmp_path / "mixed.mpco", [(_QUAD, [5]), (_TET, [10])])
    with pytest.raises(MPCOGaussDimensionMixed, match="FourNodeQuad"):
        _gauss(path, "stress_xx")


# =====================================================================
# MPCO partitioned
# =====================================================================

def _mpco_parts(tmp_path: Path) -> Path:
    """``Rec.part-0.mpco`` holds quad 5, ``Rec.part-1.mpco`` tet 10."""
    p0 = _stress_file(tmp_path / "Rec.part-0.mpco", [(_QUAD, [5])])
    _stress_file(tmp_path / "Rec.part-1.mpco", [(_TET, [10])])
    return p0


class TestMPCOPartitioned:
    def test_quad_part_beside_tet_part_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(MPCOGaussDimensionMixed) as info:
            _gauss(_mpco_parts(tmp_path), "stress_xx")
        msg = str(info.value)
        assert "2-D natural coordinates in Rec.part-0.mpco (element 5)" in msg
        assert "3-D natural coordinates in Rec.part-1.mpco (element 10)" in msg
        assert "element_ids" in msg

    @pytest.mark.parametrize("ids, dim, n", [([5], 2, 4), ([10], 3, 1)])
    def test_one_class_read_answers(
        self, tmp_path: Path, ids: list[int], dim: int, n: int,
    ) -> None:
        slab = _gauss(_mpco_parts(tmp_path), "stress_xx", ids=ids)
        assert slab.natural_coords.shape == (n, dim)
        assert not np.isnan(slab.natural_coords).any()
        np.testing.assert_array_equal(slab.element_index, ids * n)
        np.testing.assert_array_equal(slab.values, [[float(ids[0])] * n])


# =====================================================================
# .ladruno (single file and partitioned)
# =====================================================================

def _ladruno(
    path: Path, *, quad: bool = True, tet: bool = True,
    part: "tuple[int, int] | None" = None,
) -> Path:
    """``quad2d.ladruno`` (quad element 1, 4 GP) with, when ``tet``, the
    BezierTet10 stress bucket of ``bezier_tet10.ladruno`` added as element
    2 (4 GP); ``quad=False`` drops the quad's stress. ``part`` = (id, n)
    stamps the partition manifest."""
    shutil.copy(_FIX / "quad2d.ladruno", path)
    with h5py.File(_FIX / "bezier_tet10.ladruno", "r") as src, \
            h5py.File(path, "r+") as dst:
        dst["INFO"].attrs["SPATIAL_DIM"] = np.array([3], dtype=np.int32)
        if part is not None:
            dst["INFO"].attrs["PARTITIONED"] = np.int32(1)
            dst["INFO"].attrs["PARTITION_ID"] = np.int32(part[0])
            dst["INFO"].attrs["NUM_PARTITIONS"] = np.int32(part[1])
        on_el = dst[f"{_STAGE}/RESULTS/ON_ELEMENTS"]
        for tok in ("strain", "force"):
            del on_el[tok]
        if tet:
            src.copy(src[f"{_STAGE}/MODEL/ELEMENTS/{_TET_MODEL}"],
                     dst[f"{_STAGE}/MODEL/ELEMENTS"], _TET_MODEL)
            conn = dst[f"{_STAGE}/MODEL/ELEMENTS/{_TET_MODEL}/CONNECTIVITY"]
            c = conn[...]
            c[0, 0] = 2                                   # element id 2
            conn[...] = c
            src.copy(src[f"{_STAGE}/RESULTS/ON_ELEMENTS/stress/{_TET_BUCKET}"],
                     on_el["stress"], _TET_BUCKET)
            g = on_el[f"stress/{_TET_BUCKET}"]
            # Match quad2d's two steps.
            d = g["DATA"][...]
            del g["DATA"], g["STEP"], g["TIME"]
            g.create_dataset("DATA", data=np.concatenate([d, d], axis=0))
            q = on_el[f"stress/{_QUAD_BUCKET}"]
            g.create_dataset("STEP", data=q["STEP"][...])
            g.create_dataset("TIME", data=q["TIME"][...])
            g["ID"][...] = np.array([[2]])
        if not quad:
            del on_el[f"stress/{_QUAD_BUCKET}"]
    return path


def _lad_read(reader, ids=None):
    sid = reader.stages()[0].id
    return reader.read_gauss(
        sid, "stress_xx",
        element_ids=None if ids is None else np.asarray(ids),
    )


class TestLadrunoSingle:
    def test_quad_beside_tet_refuses(self, tmp_path: Path) -> None:
        path = _ladruno(tmp_path / "mixed.ladruno")
        with LadrunoReader(path) as r:
            with pytest.raises(GaussDimensionMixed) as info:
                _lad_read(r)
        msg = str(info.value)
        assert "mixed.ladruno" in msg
        assert f"2-D natural coordinates in FourNodeQuad (/{_STAGE}" in msg
        assert _QUAD_BUCKET in msg
        assert f"3-D natural coordinates in BezierTet10 (/{_STAGE}" in msg
        assert _TET_BUCKET in msg

    @pytest.mark.parametrize("ids, dim", [([1], 2), ([2], 3)])
    def test_one_class_read_answers(
        self, tmp_path: Path, ids: list[int], dim: int,
    ) -> None:
        path = _ladruno(tmp_path / "mixed.ladruno")
        with LadrunoReader(path) as r:
            slab = _lad_read(r, ids)
        assert slab.natural_coords.shape == (4, dim)
        assert not np.isnan(slab.natural_coords).any()
        np.testing.assert_array_equal(slab.element_index, ids * 4)


class TestLadrunoPartitioned:
    def _parts(self, tmp_path: Path) -> list[Path]:
        return [
            _ladruno(tmp_path / "run.part-0.ladruno", tet=False, part=(0, 2)),
            _ladruno(tmp_path / "run.part-1.ladruno", quad=False, part=(1, 2)),
        ]

    def test_quad_part_beside_tet_part_refuses(self, tmp_path: Path) -> None:
        with LadrunoMultiPartitionReader(self._parts(tmp_path)) as r:
            with pytest.raises(GaussDimensionMixed) as info:
                _lad_read(r)
        msg = str(info.value)
        assert "2-D natural coordinates in run.part-0.ladruno (element 1)" in msg
        assert "3-D natural coordinates in run.part-1.ladruno (element 2)" in msg

    @pytest.mark.parametrize("ids, dim", [([1], 2), ([2], 3)])
    def test_one_class_read_answers(
        self, tmp_path: Path, ids: list[int], dim: int,
    ) -> None:
        with LadrunoMultiPartitionReader(self._parts(tmp_path)) as r:
            slab = _lad_read(r, ids)
        assert slab.natural_coords.shape == (4, dim)
        assert not np.isnan(slab.natural_coords).any()
        np.testing.assert_array_equal(slab.element_index, ids * 4)


# =====================================================================
# native
# =====================================================================

_TRI_ID, _TET_ID = 11, 22


def _native(path: Path) -> Path:
    """A tri group (class_tag 33, one 2-D GP, value 1.0) and a tet group
    (class_tag 179, one 3-D GP, value 2.0) in one partition."""
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(name="s", kind="static", time=np.array([1.0]))
        w.write_gauss_group(
            sid, "partition_0", "tri", class_tag=33, int_rule=0,
            element_index=np.array([_TRI_ID]),
            natural_coords=np.array([[1 / 3, 1 / 3]]),
            components={"stress_xx": np.full((1, 1, 1), 1.0)},
        )
        w.write_gauss_group(
            sid, "partition_0", "tet", class_tag=179, int_rule=0,
            element_index=np.array([_TET_ID]),
            natural_coords=np.array([[0.25, 0.25, 0.25]]),
            components={"stress_xx": np.full((1, 1, 1), 2.0)},
        )
        w.end_stage()
    return path


def _native_read(path: Path, ids=None):
    with NativeReader(path) as r:
        return _lad_read(r, ids)


class TestNative:
    def test_tri_beside_tet_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(GaussDimensionMixed) as info:
            _native_read(_native(tmp_path / "mixed.h5"))
        msg = str(info.value)
        assert "native gauss read of 'stress_xx' in mixed.h5" in msg
        assert "2-D natural coordinates in class_tag 33" in msg
        assert "gauss_points/tri" in msg
        assert "3-D natural coordinates in class_tag 179" in msg
        assert "gauss_points/tet" in msg

    @pytest.mark.parametrize(
        "ids, dim, value", [([_TRI_ID], 2, 1.0), ([_TET_ID], 3, 2.0)],
    )
    def test_one_class_read_answers(
        self, tmp_path: Path, ids: list[int], dim: int, value: float,
    ) -> None:
        slab = _native_read(_native(tmp_path / "mixed.h5"), ids)
        assert slab.natural_coords.shape == (1, dim)
        np.testing.assert_array_equal(slab.element_index, ids)
        np.testing.assert_array_equal(slab.values, [[value]])


def test_mpco_error_is_a_value_error() -> None:
    """Both refusal classes stay ``ValueError``s, so a caller catching
    the old numpy ``ValueError`` still catches the named refusal."""
    assert issubclass(GaussDimensionMixed, ValueError)
    assert issubclass(MPCOGaussDimensionMixed, ValueError)
