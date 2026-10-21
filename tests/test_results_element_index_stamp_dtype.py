"""#1811: ``element_index_space`` is a scalar vlen UTF-8 string, exactly "fem".

The writers emit that form, and ``NativeReader`` verifies only that form.
Any other on-disk form of the text ``fem`` (fixed-length, bytes, a vlen
ASCII string, an array) reads as unverified: another HDF5 reader may strip
a fixed-length string's padding, or decode ASCII the same way, and compare
it equal to ``"fem"``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._native import NativeReader
from apeGmsh.results.schema import _native

from tests.test_results_element_index_stamp import (  # noqa: F401 (fixture)
    _node_record,
    _transcode,
    portal,
)

ATTR = _native.ATTR_ELEMENT_INDEX_SPACE


def _assert_vlen_utf8(path: Path) -> None:
    with h5py.File(path, "r") as f:
        attr = f.attrs.get_id(ATTR)
        info = h5py.check_string_dtype(attr.dtype)
        assert info is not None, attr.dtype
        assert info.length is None, "fixed-length stamp"
        assert info.encoding == "utf-8"
        assert attr.shape == ()
        assert f.attrs[ATTR] == "fem"


def _stamped_file(tmp_path: Path, portal: dict) -> Path:
    """A transcoder-written native file carrying the writer's own stamp."""
    return _transcode(tmp_path, portal, _node_record(), build="sequential")


# ---------------------------------------------------------------------------
# Writers emit vlen UTF-8
# ---------------------------------------------------------------------------


def test_transcoder_stamp_is_vlen_utf8(
    tmp_path: Path, portal: dict,  # noqa: F811 (the imported fixture)
) -> None:
    _assert_vlen_utf8(_stamped_file(tmp_path, portal))


# ---------------------------------------------------------------------------
# The reader verifies only vlen UTF-8
# ---------------------------------------------------------------------------


def _restamp(path: Path, value: Any, dtype: Any) -> None:
    with h5py.File(path, "r+") as f:
        del f.attrs[ATTR]
        f.attrs.create(ATTR, value, dtype=dtype)


def _verified(path: Path) -> bool:
    with NativeReader(path) as r:
        assert r.stages(), "the file must still read"
        return r.element_index_verified


def test_vlen_utf8_fem_is_verified(
    tmp_path: Path, portal: dict,  # noqa: F811 (the imported fixture)
) -> None:
    path = _stamped_file(tmp_path, portal)
    _restamp(path, "fem", h5py.string_dtype("utf-8"))
    assert _verified(path)


@pytest.mark.parametrize("value, dtype", [
    (b"fem", h5py.string_dtype("utf-8", 3)),      # fixed-length UTF-8
    (b"fem", h5py.string_dtype("ascii", 3)),      # fixed-length ASCII
    (b"fem\x00\x00", h5py.string_dtype("ascii", 5)),  # NUL-padded
    (np.bytes_(b"fem"), None),                    # numpy bytes -> fixed |S3
    (b"fem", h5py.string_dtype("ascii")),         # vlen BYTES: verified on main
    ("fem", h5py.string_dtype("ascii")),          # vlen ASCII from a str
    (["fem"], h5py.string_dtype("utf-8")),        # vlen UTF-8, but an array
    (1, np.int64),                                # not a string
    (1.0, np.float64),                            # not a string
], ids=["fixed_utf8", "fixed_ascii", "nul_padded", "numpy_bytes",
        "vlen_bytes", "vlen_ascii", "array", "int", "float"])
def test_other_forms_read_unverified(
    tmp_path: Path, portal: dict,  # noqa: F811 (the imported fixture)
    value: Any, dtype: Any,
) -> None:
    path = _stamped_file(tmp_path, portal)
    _restamp(path, value, dtype)
    assert not _verified(path)
