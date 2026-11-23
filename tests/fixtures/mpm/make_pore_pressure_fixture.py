"""Build the ``PORE_PRESSURE`` fixture of #3121 from the fork's fixture.

Fork release ``ladruno-v20261123b`` (``3eb195bbc``, WP-232l, OpenSees#1116, P12)
adds one particle output field to Contract 2 (232a §2.6): ``PORE_PRESSURE``,
``COMPONENTS = "p"`` in 2-D and 3-D, ``DIMENSION = "F/L^2"``, ``DATA_TYPE = 0``,
the excess pore pressure of a two-phase particle, compression positive (0 on a
single-phase particle); additive, so ``FORMAT_MINOR_VERSION`` stays 2. The fork
commits no ``.ladruno`` file with that field, so this one is
``particles_v1m2.ladruno`` (the fork's fixture, vendored here) with a
``RESULTS/ON_PARTICLES/PORE_PRESSURE`` group added in the writer's encoding
(``LadrunoMPMIO.cpp``, the stage-open field table at the pin):

- ``DISPLAY_NAME = "Pore pressure (compression positive)"``, ``COMPONENTS =
  "p"``, ``DIMENSION = "F/L^2"``, ``PARTITION_REDUCTION = "NONE"``: each a
  1-element array of a fixed-length, null-terminated ASCII string;
- ``TYPE = 0``, ``DATA_TYPE = 0``: each a 1-element ``int32`` array;
- ``DESCRIPTION = ""``: the writer's empty string, which the contract's
  encoding stores as an absent attribute (232a §0; the vendored file has no
  ``DESCRIPTION`` on any field);
- ``DATA``: ``float64 [S, 1]``, chunked, deflated and shuffled as the file's
  other one-column fields (``VOLUME``).

The values are a closed form of each row's position,
``p = GAMMA_W * (Y_TOP - y)``, so a test recomputes them from ``POSITION`` and
the pressure grows with depth as a hydrostatic column's does. They are not a
two-phase run's output: the file pins what a reader *accepts*, not what a
two-phase run computes (the live test,
``tests/opensees/integration_ladruno/test_mpm_pore_pressure_readers_live.py``,
runs one and checks this encoding against the fork's).

Run from the repository root: ``python tests/fixtures/mpm/make_pore_pressure_fixture.py``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "particles_v1m2.ladruno"
TARGET = HERE / "particles_pore_pressure_v1m2.ladruno"
ON_P = "MODEL_STAGE[1]/RESULTS/ON_PARTICLES"
#: The closed form of the stored values, ``p = GAMMA_W * (Y_TOP - y)``.
GAMMA_W = 9.81
Y_TOP = 2.0
#: The writer's string attributes of the field (``LadrunoMPMIO.cpp`` at 3eb195bbc).
TEXT = {
    "DISPLAY_NAME": "Pore pressure (compression positive)",
    "COMPONENTS": "p",
    "DIMENSION": "F/L^2",
    "PARTITION_REDUCTION": "NONE",
}
INTS = {"TYPE": 0, "DATA_TYPE": 0}


def write_text(obj, name: str, value: str) -> None:
    """``name`` as the fork writes a string: a 1-element fixed NULLTERM ASCII array."""
    raw = value.encode("ascii")
    tid = h5py.h5t.C_S1.copy()
    tid.set_size(len(raw))
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    tid.set_cset(h5py.h5t.CSET_ASCII)
    space = h5py.h5s.create_simple((1,))
    attr = h5py.h5a.create(obj.id, name.encode("ascii"), tid, space)
    attr.write(np.array([raw], dtype=f"S{len(raw)}"), mtype=tid)


def pressure(position: np.ndarray) -> np.ndarray:
    """The stored ``[S, 1]`` values for the ``[S, ndm]`` positions."""
    return (GAMMA_W * (Y_TOP - position[:, 1])).reshape(-1, 1)


def add_pore_pressure(path: Path) -> None:
    """Add ``ON_PARTICLES/PORE_PRESSURE`` to the copy at ``path``, in the writer's encoding."""
    with h5py.File(path, "r+") as f:
        onp = f[ON_P]
        if "PORE_PRESSURE" in onp:
            raise ValueError(f"{path}: PORE_PRESSURE is already there")
        like = onp["VOLUME"]["DATA"]
        g = onp.create_group("PORE_PRESSURE")
        for k, v in TEXT.items():
            write_text(g, k, v)
        for k, v in INTS.items():
            g.attrs.create(k, np.array([v], dtype="<i4"))
        g.create_dataset(
            "DATA", data=pressure(onp["POSITION"]["DATA"][...]).astype("<f8"),
            maxshape=like.maxshape, chunks=like.chunks,
            compression=like.compression, compression_opts=like.compression_opts,
            shuffle=like.shuffle, track_times=False)


def build(path: Path) -> None:
    shutil.copyfile(SOURCE, path)
    add_pore_pressure(path)


if __name__ == "__main__":
    build(TARGET)
    print(TARGET.name)
