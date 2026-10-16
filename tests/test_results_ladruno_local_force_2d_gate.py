"""``.ladruno`` reads a 2-D ``localForce`` spelling only behind a gate (#2013).

A 2-D beam tags its ``localForce`` ``N_1,V_1,M_1,N_2,V_2,M_2`` (elastic
beams) or ``N1,V1,M1,N2,V2,M2`` (DispBeamColumn2d, LadrunoDispBeamColumn2d).
``V``/``M`` read as ``shear_y``/``bending_moment_z`` only when the bucket's
DATA holds exactly the tagged columns, it spells no 3-D base beside them,
the file is 2-D, and the class is no ``...3d`` class: the gate #2010 put
on MPCO.

ComponentElement3d and ForceBeamColumnCBDI3d tag a 12-value ``localForce``
with the six 2-D names (fork ``ComponentElement3d.cpp`` ``P(12)``,
``ForceBeamColumnCBDI3d.cpp`` ``theVector(12)``). The fork's recorder keeps
the declared 6 columns and fills each row: NaN at the pin, zeros on builds
before WP-163 (probed on 79e06236: ``returned 12 values but the bucket
declares 6 columns; that row is zero-filled``). Before this gate the alias
read those filler columns as ``shear_y`` and ``bending_moment_z`` with no
word. Now each such bucket lists nothing, warns naming its class, and a
read of its elements raises naming its class.

The oracle is constructive: each synthetic column holds a distinct value,
and a station reads ``sign * end value`` under the one sign table
(axial -1, shear_y +1, bending_moment_z -1 at end i; the opposite at end j).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._protocol import ResultLevel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ladruno"
STAGE = "MODEL_STAGE[1]"
FILE_2D = "fiberbeam.ladruno"   # INFO/SPATIAL_DIM = 2
FILE_3D = "beam3d.ladruno"      # INFO/SPATIAL_DIM = 3

NAMES_UNDERSCORE = "N_1,V_1,M_1,N_2,V_2,M_2"
NAMES_BARE = "N1,V1,M1,N2,V2,M2"

#: canonical -> (end-i column, end-j column, end-i sign)
_COLUMNS = {
    "axial_force": (0, 3, -1.0),
    "shear_y": (1, 4, 1.0),
    "bending_moment_z": (2, 5, -1.0),
}


def _raw(e: int, col: int, t: int) -> float:
    return 1000.0 * e + 10.0 * (col + 1) + 0.5 * (t + 1)


def _add_bucket(
    path: Path, key: str, names: str, ids: list[int], *,
    multiplicity: int = 1, fill: float | None = None,
) -> None:
    """Add ``ON_ELEMENTS/localForce/<key>``: ``names`` repeated
    ``multiplicity`` times, each cell ``_raw`` (or ``fill``)."""
    width = len(names.split(",")) * multiplicity
    with h5py.File(path, "r+") as f:
        on_e = f[f"{STAGE}/RESULTS/ON_ELEMENTS"]
        some = next(iter(next(iter(on_e.values())).values()))
        n_steps = int(some["DATA"].shape[0])
        lf = (on_e["localForce"] if "localForce" in on_e
              else on_e.create_group("localForce"))
        grp = lf.create_group(key)
        data = np.empty((n_steps, len(ids), width))
        for t in range(n_steps):
            for i, e in enumerate(ids):
                for c in range(width):
                    data[t, i, c] = _raw(e, c, t) if fill is None else fill
        cm = grp.create_group("COLUMN_MAP")
        for name, value in (("LEVELS", 0), ("GAUSS_ID", -1),
                            ("MULTIPLICITY", multiplicity),
                            ("SECTION_TAG", -1), ("FIBER_ID", -1)):
            cm.create_dataset(name, data=np.array([[value]], dtype=np.int32))
        cm.create_dataset("NUM_COMP", data=np.array(
            [[len(names.split(","))]], dtype=np.int32))
        cm.attrs["COMP_NAMES"] = np.bytes_(names)
        grp.create_dataset("DATA", data=data)
        grp.create_dataset("ID", data=np.array(ids, dtype=np.int32)[:, None])
        grp.create_dataset("STEP", data=np.arange(n_steps, dtype=np.int32))
        grp.create_dataset("TIME", data=np.asarray(some["TIME"][...]))


def _copy(tmp_path: Path, source: str) -> Path:
    dst = tmp_path / source
    shutil.copyfile(FIXTURES / source, dst)
    return dst


def _stations(path: Path, comp: str, ids: list[int]) -> np.ndarray:
    with LadrunoReader(path) as r:
        slab = r.read_line_stations(
            "stage_0", comp, element_ids=np.array(ids),
        )
    assert slab.element_index.tolist() == [e for e in ids for _ in (1, 2)]
    np.testing.assert_allclose(slab.station_natural_coord,
                               np.tile([-1.0, 1.0], len(ids)))
    return slab.values


# ---------------------------------------------------------------------
# A 2-D beam in a 2-D file reads, in either spelling
# ---------------------------------------------------------------------

@pytest.mark.parametrize("key, names", [
    ("62-DispBeamColumn2d[1000:1:0]", NAMES_BARE),
    ("33013-LadrunoDispBeamColumn2d[0:0:0]", NAMES_BARE),
    ("5-ElasticBeam2d[1:0:0]", NAMES_UNDERSCORE),
])
def test_2d_beam_reads_its_stations(
    tmp_path: Path, key: str, names: str,
) -> None:
    path = _copy(tmp_path, FILE_2D)
    ids = [901, 902]
    _add_bucket(path, key, names, ids)
    with LadrunoReader(path) as r:
        comps = set(r.available_components(
            "stage_0", ResultLevel.LINE_STATIONS))
    assert set(_COLUMNS) <= comps
    for comp, (ci, cj, sign) in _COLUMNS.items():
        values = _stations(path, comp, ids)
        n_t = values.shape[0]
        want = np.array([[v for e in ids for v in (
            sign * _raw(e, ci, t), -sign * _raw(e, cj, t))]
            for t in range(n_t)])
        np.testing.assert_allclose(values, want, rtol=0, atol=0,
                                   err_msg=f"{key} {comp}")


# ---------------------------------------------------------------------
# A 3-D class under the 2-D names refuses, naming the class
# ---------------------------------------------------------------------

#: The pinned fork's ``.ladruno`` bucket for a 12-value localForce under 6
#: tags: the declared 6 columns, each row filled (NaN at the pin, zeros
#: before WP-163). ``None`` fill = the card's 12-wide layout.
_REFUSED_3D = [
    ("195195-ComponentElement3d[0:0:0]", 1, float("nan")),
    ("195195-ComponentElement3d[0:0:0]", 1, 0.0),
    ("195195-ComponentElement3d[0:0:0]", 2, None),
    ("78-ForceBeamColumnCBDI3d[0:0:0]", 1, float("nan")),
    ("78-ForceBeamColumnCBDI3d[0:0:0]", 1, 0.0),
]


@pytest.mark.parametrize("key, mult, fill", _REFUSED_3D)
def test_3d_class_under_2d_names_refuses_by_class(
    tmp_path: Path, key: str, mult: int, fill: float | None,
) -> None:
    cls = key.split("-", 1)[1].split("[", 1)[0]
    path = _copy(tmp_path, FILE_3D)
    _add_bucket(path, key, NAMES_UNDERSCORE, [77], multiplicity=mult,
                fill=fill)
    with LadrunoReader(path) as r:
        with pytest.warns(UserWarning, match=cls):
            r.available_components("stage_0", ResultLevel.LINE_STATIONS)
        for comp in ("shear_y", "bending_moment_z", "axial_force"):
            for ids in (np.array([77]), None):
                with pytest.raises(ValueError, match=cls):
                    r.read_line_stations("stage_0", comp, element_ids=ids)
        # The fixture's own ElasticBeam3d (element 1) still reads.
        slab = r.read_line_stations("stage_0", "shear_y",
                                    element_ids=np.array([1]))
        assert slab.element_index.tolist() == [1, 1]
        assert np.isfinite(slab.values).all()


def test_listing_leaves_a_refused_bucket_out(tmp_path: Path) -> None:
    """In a 2-D file, ComponentElement3d's bucket alone must list nothing:
    its 2-D names read as no component (the fixture's section stations
    list ``bending_moment_z`` and ``axial_force``, never ``shear_y``)."""
    path = _copy(tmp_path, FILE_2D)
    with LadrunoReader(path) as r:
        before = set(r.available_components(
            "stage_0", ResultLevel.LINE_STATIONS))
    assert "shear_y" not in before
    _add_bucket(path, "195195-ComponentElement3d[0:0:0]", NAMES_UNDERSCORE,
                [77], fill=float("nan"))
    with LadrunoReader(path) as r:
        with pytest.warns(UserWarning, match="ComponentElement3d"):
            after = set(r.available_components(
                "stage_0", ResultLevel.LINE_STATIONS))
        assert after == before
        with pytest.raises(ValueError, match="ComponentElement3d"):
            r.read_line_stations("stage_0", "shear_y",
                                 element_ids=np.array([77]))


# ---------------------------------------------------------------------
# Each gate on its own, on a 2-D class
# ---------------------------------------------------------------------

def test_width_gate_refuses_12_columns_under_6_names(tmp_path: Path) -> None:
    path = _copy(tmp_path, FILE_2D)
    _add_bucket(path, "5-ElasticBeam2d[1:0:0]", NAMES_UNDERSCORE, [901],
                multiplicity=2)
    with LadrunoReader(path) as r:
        with pytest.raises(ValueError, match=r"ElasticBeam2d.*12 columns"):
            r.read_line_stations("stage_0", "shear_y",
                                 element_ids=np.array([901]))


def test_dimension_gate_refuses_2d_names_in_a_3d_file(tmp_path: Path) -> None:
    path = _copy(tmp_path, FILE_3D)
    _add_bucket(path, "62-DispBeamColumn2d[1000:1:0]", NAMES_BARE, [901])
    with LadrunoReader(path) as r:
        with pytest.raises(ValueError,
                           match=r"DispBeamColumn2d.*spatial dimension is 3"):
            r.read_line_stations("stage_0", "bending_moment_z",
                                 element_ids=np.array([901]))


def test_mixed_2d_and_3d_names_refuse(tmp_path: Path) -> None:
    path = _copy(tmp_path, FILE_2D)
    _add_bucket(path, "5-ElasticBeam2d[1:0:0]",
                "N_1,V_1,Mz_1,N_2,V_2,Mz_2", [901])
    with LadrunoReader(path) as r:
        with pytest.raises(ValueError, match=r"ElasticBeam2d.*mixes"):
            r.read_line_stations("stage_0", "shear_y",
                                 element_ids=np.array([901]))


def test_a_refused_bucket_blocks_only_its_own_elements(tmp_path: Path) -> None:
    """A read of another bucket's elements passes the refused one over."""
    path = _copy(tmp_path, FILE_2D)
    _add_bucket(path, "62-DispBeamColumn2d[1000:1:0]", NAMES_BARE, [901])
    _add_bucket(path, "195195-ComponentElement3d[0:0:0]", NAMES_UNDERSCORE,
                [77], fill=float("nan"))
    values = _stations(path, "shear_y", [901])
    np.testing.assert_allclose(values[:, 0], [_raw(901, 1, t)
                                              for t in range(len(values))])
