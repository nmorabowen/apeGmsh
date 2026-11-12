"""LadrunoIMKBeam ``localForce`` line stations on MPCO and ``.ladruno`` (#2022).

Fork builds before 208523694 record a LadrunoIMKBeam's ``localForce``
untagged, so both formats name its columns ``C1..C12`` (2-D ``C1..C6``);
later builds tag them and read by name. Before #2022 both
readers passed such a bucket over: a line-station read of an IMK beam came
back ``(T, 0)`` with no warning, while an ElasticBeam3d with the same
values read its two end stations.

The column order is ``NODAL_FORCE_CATALOG[(cls, "local_force")]``, end i
then end j, live-verified against ``eleResponse(tag, "localForce")`` by
``test_mpco_frame_nodal_forces_live.py``; the stations take the one sign
table every route applies (``local_force_station_signs``). Two oracles:

* a hand table: a bucket whose column ``C<k>`` holds ``k`` reads, per
  component, ``[s_i * k_i, -s_i * k_j]`` with ``s_i`` the end-i sign
  (``N``, ``Vz``, ``T``, ``My``, ``Mz`` flip at end i; ``Vy`` keeps it);
* statics on #2107's cantilever (``test_results_line_station_source_rank``):
  the IMK bucket holding the elastic beam's end forces reads the same
  closed-form moment diagram, ``bending_moment_z`` 8600 -> 600.

Any other beam class whose ``localForce`` holds only ``C`` names has no
known column order and is refused by name; a force vector (a truss's, a
TrussSection's, a LadrunoEmbeddedNode's) is still no beam's stations and
stays passed over.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader
from tests.test_results_line_station_source_rank import (
    FORMATS,
    LAMBDA,
    MOMENTS,
    _assert_statics,
    _end_forces,
    _File,
    _listed,
    _read,
    _rows,
)

IMK_3D = "33003-LadrunoIMKBeam[1:0:0]"
IMK_2D = "33004-LadrunoIMKBeam2d[1:0:0]"
ELASTIC_3D = "5-ElasticBeam3d[1:0:0]"
UNKNOWN = "4242-UntaggedBeam3d[1:0:0]"
TRUSS = "12-Truss[1:0:0]"

COMPS_3D = ("axial_force", "shear_y", "shear_z", "torsion",
            "bending_moment_y", "bending_moment_z")
COMPS_2D = ("axial_force", "shear_y", "bending_moment_z")

#: ``C<k>`` holds ``k``: the two stations per component, by hand. End i
#: flips N, Vz, T, My, Mz and keeps Vy; end j takes the opposite sign.
HAND_3D = {"axial_force": (-1, 7), "shear_y": (2, -8), "shear_z": (-3, 9),
           "torsion": (-4, 10), "bending_moment_y": (-5, 11),
           "bending_moment_z": (-6, 12)}
HAND_2D = {"axial_force": (-1, 4), "shear_y": (2, -5),
           "bending_moment_z": (-3, 6)}

_TAGGED_3D = ",".join(f"{b}_{s}" for s in (1, 2)
                      for b in ("N", "Vy", "Vz", "T", "My", "Mz"))


def _c_names(n: int) -> str:
    return ",".join(f"C{k}" for k in range(1, n + 1))


def _counting(n: int) -> Any:
    return lambda _eid: [float(k) for k in range(1, n + 1)]


def _assert_hand(slab: Any, comp: str, eid: int,
                 hand: dict[str, tuple[int, int]]) -> None:
    rows = _rows(slab, eid)
    assert len(rows) == 2, (comp, eid, slab.element_index.tolist())
    np.testing.assert_allclose(slab.station_natural_coord[rows], [-1.0, 1.0],
                               rtol=0, atol=1e-12)
    want = np.outer(LAMBDA, hand[comp])
    np.testing.assert_allclose(slab.values[:, rows], want, rtol=1e-12,
                               err_msg=f"{comp} of element {eid}")


# ---------------------------------------------------------------------
# An IMK bucket decodes
# ---------------------------------------------------------------------

@pytest.mark.parametrize("fmt", FORMATS)
def test_imk_3d_bucket_reads_two_end_stations(tmp_path: Path, fmt: str) -> None:
    """The card's repro: main read ``(2, 0)``, no ids, no warning."""
    f = _File(fmt, tmp_path / f"imk.{fmt}")
    f.add("localForce", [11], key=IMK_3D, names=_c_names(12),
          row=_counting(12))
    with f.reader() as r:
        assert set(COMPS_3D) <= _listed(r)
        for comp in COMPS_3D:
            for ids in ([11], None):
                slab = _read(r, comp, ids)
                assert slab.element_index.tolist() == [11, 11]
                _assert_hand(slab, comp, 11, HAND_3D)


@pytest.mark.parametrize("comp", MOMENTS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_imk_3d_bucket_reads_cantilever_statics(
    tmp_path: Path, fmt: str, comp: str,
) -> None:
    """The elastic beam's end forces under the IMK's untagged names read
    #2107's closed-form moment diagram."""
    f = _File(fmt, tmp_path / f"imk.{fmt}")
    f.add("localForce", [11, 12], key=IMK_3D, names=_c_names(12),
          row=_end_forces)
    with f.reader() as r:
        slab = _read(r, comp, None)
        for eid in (11, 12):
            _assert_statics(slab, comp, eid, np.array([-1.0, 1.0]))


@pytest.mark.parametrize("fmt", FORMATS)
def test_imk_2d_bucket_reads_two_end_stations(tmp_path: Path, fmt: str) -> None:
    """``C1..C6`` of LadrunoIMKBeam2d: ``N, Vy, Mz`` per end. A component
    the 2-D order does not hold reads no stations and raises nothing."""
    f = _File(fmt, tmp_path / f"imk2d.{fmt}", ndm=2)
    f.add("localForce", [11], key=IMK_2D, names=_c_names(6),
          row=_counting(6))
    with f.reader() as r:
        assert set(COMPS_2D) <= _listed(r)
        assert not {"shear_z", "torsion", "bending_moment_y"} & _listed(r)
        for comp in COMPS_2D:
            _assert_hand(_read(r, comp, [11]), comp, 11, HAND_2D)
        for comp in ("shear_z", "torsion", "bending_moment_y"):
            assert _read(r, comp, [11]).element_index.size == 0


# ---------------------------------------------------------------------
# Shared namespace: the elastic-beam control is unchanged
# ---------------------------------------------------------------------

@pytest.mark.parametrize("fmt", FORMATS)
def test_imk_beside_elastic_beam_reads_the_same_stations(
    tmp_path: Path, fmt: str,
) -> None:
    """One file, one read: the tagged ElasticBeam3d (element 11) keeps its
    stations, and the IMK (element 12) holding the same values under
    ``C1..C12`` reads exactly the same ones."""
    f = _File(fmt, tmp_path / f"mixed.{fmt}")
    f.add("localForce", [11], key=ELASTIC_3D, names=_TAGGED_3D,
          row=_counting(12))
    f.add("localForce", [12], key=IMK_3D, names=_c_names(12),
          row=_counting(12))
    with f.reader() as r:
        for comp in COMPS_3D:
            slab = _read(r, comp, None)
            assert sorted(slab.element_index.tolist()) == [11, 11, 12, 12]
            assert slab.values.shape == (LAMBDA.size, 4)
            _assert_hand(slab, comp, 11, HAND_3D)
            _assert_hand(slab, comp, 12, HAND_3D)


@pytest.mark.parametrize("fmt", FORMATS)
def test_elastic_beam_control_alone(tmp_path: Path, fmt: str) -> None:
    f = _File(fmt, tmp_path / f"elastic.{fmt}")
    f.add("localForce", [11], key=ELASTIC_3D, names=_TAGGED_3D,
          row=_counting(12))
    with f.reader() as r:
        for comp in COMPS_3D:
            slab = _read(r, comp, [11])
            assert slab.values.shape == (LAMBDA.size, 2)
            assert slab.element_index.tolist() == [11, 11]
            _assert_hand(slab, comp, 11, HAND_3D)


# ---------------------------------------------------------------------
# A C-only bucket of any other class is refused by name
# ---------------------------------------------------------------------

@pytest.mark.parametrize("fmt", FORMATS)
def test_unknown_class_c_only_bucket_refuses_by_name(
    tmp_path: Path, fmt: str,
) -> None:
    """Main passed it over: ``(2, 0)``, no warning. Now a read of its
    element raises, naming the class and the bucket; a read of the IMK
    beside it goes on."""
    f = _File(fmt, tmp_path / f"unknown.{fmt}")
    f.add("localForce", [11], key=UNKNOWN, names=_c_names(12),
          row=_counting(12))
    f.add("localForce", [12], key=IMK_3D, names=_c_names(12),
          row=_counting(12))
    with f.reader() as r:
        for ids in ([11], None):
            with pytest.raises(ValueError,
                               match=r"UntaggedBeam3d.*untagged") as exc:
                _read(r, "bending_moment_z", ids)
            assert UNKNOWN in str(exc.value)
        _assert_hand(_read(r, "bending_moment_z", [12]),
                     "bending_moment_z", 12, HAND_3D)


@pytest.mark.parametrize("fmt", FORMATS)
def test_truss_c_only_bucket_stays_passed_over(
    tmp_path: Path, fmt: str,
) -> None:
    """A non-beam's ``C1..C4`` is a force vector, no beam's stations: no
    stations, no warning, no refusal (the pre-#2022 behaviour, kept)."""
    f = _File(fmt, tmp_path / f"truss.{fmt}")
    f.add("localForce", [11], key=TRUSS, names=_c_names(4),
          row=_counting(4))
    with f.reader() as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            slab = _read(r, "axial_force", None)
        assert slab.element_index.size == 0


#: Classes outside the catalog's non-beams whose untagged ``localForce`` is
#: a force vector (fork ``setResponse``): TrussSection ``Vector(numDOF)``,
#: LadrunoEmbeddedNode ``Vector(ndm)``, LadrunoEmbeddedRebar ``Vector(ndm)``
#: (older builds).
FORCE_VECTOR_BUCKETS = [
    ("13-TrussSection[1:0:0]", 6),
    ("33006-LadrunoEmbeddedNode[0:0:0]", 3),
    ("33005-LadrunoEmbeddedRebar[0:0:0]", 3),
]


@pytest.mark.parametrize(("key", "width"), FORCE_VECTOR_BUCKETS,
                         ids=[k.split("[")[0] for k, _ in FORCE_VECTOR_BUCKETS])
@pytest.mark.parametrize("fmt", FORMATS)
def test_force_vector_c_only_bucket_beside_a_beam_is_passed_over(
    tmp_path: Path, fmt: str, key: str, width: int,
) -> None:
    """The review's F1 repro: an unfiltered ``bending_moment_z`` read of a
    tagged ElasticBeam3d beside one of these buckets raised at 8ac43db0.
    The force vector is no beam's stations: the beam reads, nothing
    raises or warns."""
    f = _File(fmt, tmp_path / f"vector.{fmt}")
    f.add("localForce", [11], key=ELASTIC_3D, names=_TAGGED_3D,
          row=_counting(12))
    f.add("localForce", [21], key=key, names=_c_names(width),
          row=_counting(width))
    with f.reader() as r:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for ids in (None, [21]):
                slab = _read(r, "bending_moment_z", ids)
                assert slab.element_index.tolist() == (
                    [11, 11] if ids is None else [])
        _assert_hand(_read(r, "bending_moment_z", None),
                     "bending_moment_z", 11, HAND_3D)


# ---------------------------------------------------------------------
# Degenerate IMK buckets
# ---------------------------------------------------------------------

@pytest.mark.parametrize(("key", "names", "ndm", "match"), [
    (IMK_3D, _c_names(6), 3, r"6 untagged columns.*12 wide"),
    (IMK_2D, _c_names(12), 2, r"12 untagged columns.*6 wide"),
    (IMK_3D, _c_names(7), 3, r"7 untagged columns.*12 wide"),
    (IMK_3D, "C2,C1," + ",".join(f"C{k}" for k in range(3, 13)), 3,
     r"not C1\.\.C12 in order"),
])
@pytest.mark.parametrize("fmt", FORMATS)
def test_imk_bucket_of_a_foreign_width_refuses(
    tmp_path: Path, fmt: str, key: str, names: str, ndm: int, match: str,
) -> None:
    n = len(names.split(","))
    f = _File(fmt, tmp_path / f"width.{fmt}", ndm=ndm)
    f.add("localForce", [11], key=key, names=names, row=_counting(n))
    with f.reader() as r:
        with pytest.raises(ValueError, match=match) as exc:
            _read(r, "axial_force", [11])
        assert key in str(exc.value)


def _add_empty(f: _File, key: str, names: str) -> None:
    """A ``localForce`` bucket that recorded steps but holds no element."""
    width = len(names.split(","))
    with h5py.File(f.path, "r+") as h:
        grp = h["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS"].require_group(
            "localForce").create_group(key)
        if f.fmt == "ladruno":
            f._ladruno_column_map(grp, [names], level=0)
            f._ladruno_data(grp, [], np.zeros((LAMBDA.size, 0, width)))
            return
        grp.create_group("META").create_dataset(
            "COMPONENTS", data=np.array([f"0.{names}".encode()]))
        grp.create_dataset("ID", data=np.zeros((0, 1), dtype=np.int32))
        dg = grp.create_group("DATA")
        for t in range(LAMBDA.size):
            ds = dg.create_dataset(f"STEP_{t}", data=np.zeros((0, width)))
            ds.attrs["STEP"] = t
            ds.attrs["TIME"] = float(t + 1)


@pytest.mark.parametrize("fmt", FORMATS)
def test_imk_bucket_with_no_rows_reads_empty(tmp_path: Path, fmt: str) -> None:
    f = _File(fmt, tmp_path / f"empty.{fmt}")
    _add_empty(f, IMK_3D, _c_names(12))
    f.add("localForce", [11], key=ELASTIC_3D, names=_TAGGED_3D,
          row=_counting(12))
    with f.reader() as r:
        slab = _read(r, "axial_force", None)
        assert slab.element_index.tolist() == [11, 11]
        _assert_hand(slab, "axial_force", 11, HAND_3D)


# ---------------------------------------------------------------------
# Partitioned: the IMK beam lives in one part
# ---------------------------------------------------------------------

@pytest.mark.parametrize("fmt", FORMATS)
def test_partition_set_with_imk_in_one_part(tmp_path: Path, fmt: str) -> None:
    part0 = _File(fmt, tmp_path / f"run.part-0.{fmt}")
    part0.add("localForce", [11], key=ELASTIC_3D, names=_TAGGED_3D,
              row=_counting(12))
    part1 = _File(fmt, tmp_path / f"run.part-1.{fmt}")
    part1.add("localForce", [12], key=IMK_3D, names=_c_names(12),
              row=_counting(12))
    multi = (LadrunoMultiPartitionReader if fmt == "ladruno"
             else MPCOMultiPartitionReader)
    with multi([part0.path, part1.path]) as r:
        for comp in COMPS_3D:
            for ids in ([12], None):
                slab = _read(r, comp, ids)
                _assert_hand(slab, comp, 12, HAND_3D)
            assert sorted(slab.element_index.tolist()) == [11, 11, 12, 12]
            _assert_hand(slab, comp, 11, HAND_3D)
