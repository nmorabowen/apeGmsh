"""A ``.part-N.ladruno`` set whose parts disagree on the spatial dimension
is refused by name (#2267, the Python side of the viewer's #2245).

``LadrunoMultiPartitionReader._validate_part_set`` resolves each part's
dimension with ``_file_spatial_dim``, the one rule every ``.ladruno`` read
uses: ``INFO/SPATIAL_DIM``, else the one ``MODEL/NODES/COORDINATES``
width, else unknown. Parts that disagree are refused, and the message names
each part, its value and where the value came from. An unknown dimension
beside a known one is a disagreement too, as in the viewer: the fork writes
``INFO/SPATIAL_DIM`` into every part it creates.

The oracle is the committed pair ``truss2d.part-{0,1}.ladruno``: both say
``SPATIAL_DIM = 2`` over 2-wide coordinates, and the set opens. Each case
edits a scratch copy of part 1. Before this check, every refused case below
opened in silence with part 0's dimension.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno_element_io import SpatialDimInvalid
from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "ladruno"
STAGE = "MODEL_STAGE[1]"
PARTS = ("truss2d.part-0.ladruno", "truss2d.part-1.ladruno")

Edit = Callable[[h5py.File], None]


def _set_dim(value) -> Edit:
    def edit(f: h5py.File) -> None:
        f["INFO"].attrs["SPATIAL_DIM"] = np.array([value], dtype=np.int32)
    return edit


def _strip_dim(f: h5py.File) -> None:
    del f["INFO"].attrs["SPATIAL_DIM"]


def _coords_width(width: int) -> Edit:
    """Rewrite the stage's coordinates ``width`` wide (zero-padded)."""
    def edit(f: h5py.File) -> None:
        nodes = f[f"{STAGE}/MODEL/NODES"]
        xy = nodes["COORDINATES"][()]
        del nodes["COORDINATES"]
        out = np.zeros((xy.shape[0], width), dtype=xy.dtype)
        out[:, :xy.shape[1]] = xy
        nodes.create_dataset("COORDINATES", data=out)
    return edit


def _drop_model(f: h5py.File) -> None:
    del f[f"{STAGE}/MODEL"]


def _scratch_set(tmp_path: Path, edits: dict[int, list[Edit]]) -> list[Path]:
    """Copy the committed pair, applying ``edits[k]`` to part ``k``."""
    out = []
    for k, name in enumerate(PARTS):
        dst = tmp_path / name
        shutil.copyfile(FIXTURES / name, dst)
        if edits.get(k):
            with h5py.File(dst, "r+") as f:
                for edit in edits[k]:
                    edit(f)
        out.append(dst)
    return out


def _opens(paths: list[Path]) -> None:
    with LadrunoMultiPartitionReader(paths) as m:
        assert m.stages()


def test_the_committed_pair_agrees_and_opens(tmp_path: Path) -> None:
    for name in PARTS:
        with h5py.File(FIXTURES / name, "r") as f:
            assert np.asarray(f["INFO"].attrs["SPATIAL_DIM"]).tolist() == [2]
            assert f[f"{STAGE}/MODEL/NODES/COORDINATES"].shape[1] == 2
    _opens(_scratch_set(tmp_path, {}))


def test_part_1_rewritten_as_3_against_part_0s_2_is_refused_by_name(
        tmp_path: Path) -> None:
    paths = _scratch_set(tmp_path, {1: [_set_dim(3)]})
    said = (r"truss2d\.part-0\.ladruno=2 \(INFO@SPATIAL_DIM\), "
            r"truss2d\.part-1\.ladruno=3 \(INFO@SPATIAL_DIM\)")
    with pytest.raises(ValueError, match="disagree on the spatial dimension: "
                       + said):
        LadrunoMultiPartitionReader(paths)
    # The public path refuses it from either part.
    for p in paths:
        with pytest.raises(ValueError, match=said):
            Results.from_ladruno(p)


def test_an_inferred_width_of_3_against_part_0s_2_names_its_source(
        tmp_path: Path) -> None:
    paths = _scratch_set(tmp_path, {1: [_strip_dim, _coords_width(3)]})
    with pytest.raises(
            ValueError,
            match=(r"truss2d\.part-0\.ladruno=2 \(INFO@SPATIAL_DIM\), "
                   r"truss2d\.part-1\.ladruno=3 \(the coordinate width\)")):
        LadrunoMultiPartitionReader(paths)


def test_an_inferred_width_that_agrees_opens(tmp_path: Path) -> None:
    _opens(_scratch_set(tmp_path, {1: [_strip_dim]}))


def test_unknown_beside_known_is_refused(tmp_path: Path) -> None:
    # The viewer's #2245 rule: the set is never read as part 0's 2.
    paths = _scratch_set(tmp_path, {1: [_strip_dim, _drop_model]})
    with pytest.raises(
            ValueError,
            match=(r"truss2d\.part-0\.ladruno=2 \(INFO@SPATIAL_DIM\), "
                   r"truss2d\.part-1\.ladruno=unknown \(no INFO@SPATIAL_DIM "
                   r"and no single coordinate width\)")):
        LadrunoMultiPartitionReader(paths)


def test_every_part_unknown_opens(tmp_path: Path) -> None:
    # No part knows the dimension, so the parts do not disagree; the
    # dimension-dependent reads refuse as a single file's do (#2081).
    _opens(_scratch_set(tmp_path, {
        0: [_strip_dim, _drop_model], 1: [_strip_dim, _drop_model]}))


def test_an_invalid_attribute_on_a_part_refuses_naming_it(
        tmp_path: Path) -> None:
    paths = _scratch_set(tmp_path, {1: [_set_dim(7)]})
    with pytest.raises(SpatialDimInvalid,
                       match=r"truss2d\.part-1\.ladruno: INFO/SPATIAL_DIM is"):
        LadrunoMultiPartitionReader(paths)
