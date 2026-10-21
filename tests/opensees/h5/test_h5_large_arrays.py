"""A long ``*args`` array archives and replays exactly (#2404, K #2418).

``ops.timeSeries.Path(values=<16384 floats>)`` ran its deck, but the
``model.h5`` write failed with ``OSError: object header message is too
large``: the writer stored the values as one HDF5 attribute, and an
attribute's object-header message is capped at 64 KiB (about 8,180
float64 slots, or 4,090 variable-length string slots, measured on
h5py 3.15 / HDF5 1.14.6).

Oracles:

* **Bridge equality.** ``OpenSeesModel.from_h5(p).build("tcl")`` equals
  the deck the bridge itself emits, byte for byte, at a small control
  size and at 16,384 and 200,000 values.
* **Hash round-trip.** ``model_hash`` of the archive equals the one of
  ``from_h5(p).to_h5(q)``, so the identity of the large value survives a
  rewrite.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.lineage import compute_model_hash
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem

#: A control below every limit, the size #2404 hit, and a size well past
#: anything the attribute form could ever hold.
SIZES = (16, 16_384, 200_000)


def _values(n: int) -> tuple[float, ...]:
    # Non-integral, distinct values: a slot that lost its float-ness or
    # its position shows in the deck.
    return tuple(0.001 * i + 0.0005 for i in range(n))


def _frame_with_path(n: int) -> apeSees:
    ops = apeSees(cast("object", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.0125, E=2900.25, Iz=1.5e-4, Iy=2.5e-4, G=1100.75, J=3.5e-4,
    )
    ops.timeSeries.Path(values=_values(n), dt=0.01)
    return ops


def _bridge_deck(ops: apeSees, tmp_path: Path) -> str:
    p = tmp_path / "bridge.tcl"
    ops.tcl(str(p), progress=False)
    return p.read_text(encoding="utf-8")


def _hash(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return compute_model_hash("", f["opensees"])


@pytest.mark.parametrize("n", SIZES)
def test_path_series_archives_and_replays_exactly(
    n: int, tmp_path: Path,
) -> None:
    ops = _frame_with_path(n)
    bridge = _bridge_deck(ops, tmp_path)
    src = tmp_path / "model.h5"
    ops.h5(str(src))

    replay = OpenSeesModel.from_h5(str(src)).build("tcl")

    assert isinstance(replay, str)
    assert replay == bridge


@pytest.mark.parametrize("n", SIZES)
def test_path_series_model_hash_round_trips(n: int, tmp_path: Path) -> None:
    src = tmp_path / "model.h5"
    _frame_with_path(n).h5(str(src))
    out = tmp_path / "rewrite.h5"

    OpenSeesModel.from_h5(str(src)).to_h5(str(out))

    assert _hash(out) == _hash(src)
