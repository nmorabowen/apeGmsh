"""``INFO/SPATIAL_DIM`` refuses what the viewer refuses (#2244).

Python read the attribute through ``float()``, which keeps the real part
of a complex ``2+5j`` (with only a ``ComplexWarning``) and reads any
Unicode decimal digit (``'２'``, ``'٢'``, ``'𝟐'``) as its value. The
viewer's ``apeGmshViewer/src/reader/ladrunoInfo.ts`` (#2205, #2265)
refuses both. Three oracles:

- the viewer's own table (``apeGmshViewer/test/results/ladruno.test.ts``,
  "#2205: INFO@SPATIAL_DIM ..."), row for row, through a real file;
- the Unicode database: every non-ASCII decimal digit is refused;
- ``float()`` itself for whitespace: over every code point ``c``, ``c+'2'``
  and ``'2'+c`` read as 2 exactly when ``float()`` reads them as 2 and
  ``c`` is no non-ASCII digit, so the skipped set is ``float()``'s own.
"""
from __future__ import annotations

import sys
import unicodedata
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno_element_io import (
    SpatialDimInvalid,
    _file_spatial_dim,
    _spatial_dim_attr,
    _spatial_dim_number,
)


def _dim_of(tmp_path: Path, value, *, coords_width: int = 2):
    """``_file_spatial_dim`` of a file whose ``INFO`` carries ``value``."""
    path = tmp_path / "dim.h5"
    with h5py.File(path, "w") as h:
        info = h.create_group("INFO")
        if value is not None:
            info.attrs["SPATIAL_DIM"] = value
        h.create_dataset("MODEL_STAGE[1]/MODEL/NODES/COORDINATES",
                         data=np.zeros((3, coords_width)))
    with h5py.File(path, "r") as h, warnings.catch_warnings():
        warnings.simplefilter("error")  # a ComplexWarning is no answer
        return _file_spatial_dim(h)


def _reads(s: object) -> int | None:
    """The rule on one value: its dimension, or ``None`` when refused."""
    try:
        return _spatial_dim_attr(np.asarray(s), "f.ladruno")
    except SpatialDimInvalid:
        return None


# -- the viewer's table, row for row ----------------------------------------

_VIEWER_REFUSED = [
    ("7", np.array([7], np.int32)),
    ("0", np.array([0], np.int32)),
    ("-1", np.array([-1], np.int32)),
    ("2.9", np.array([2.9])),
    ("'abc'", "abc"),
    ("[3, 2]", np.array([3, 2], np.int32)),
    ("U+FEFF then 2", "﻿2"),
    ("U+001C then 2", "\x1c2"),
    ("fullwidth 2", "２"),
]

_VIEWER_READ = [
    ("int32 [1]", np.array([1], np.int32), 1),
    ("int32 [2]", np.array([2], np.int32), 2),
    ("int32 scalar 3", np.int32(3), 3),
    ("int64 [3]", np.array([3], np.int64), 3),
    ("float64 [2.0]", np.array([2.0]), 2),
    ("'2'", "2", 2),
    ("' 2 '", " 2 ", 2),
    ("U+0085 around 2", "\x852\x85", 2),
    ("U+3000 then 2", "　2", 2),
]


@pytest.mark.parametrize("value", [v for _, v in _VIEWER_REFUSED],
                         ids=[c for c, _ in _VIEWER_REFUSED])
def test_viewer_refused_rows_are_refused(tmp_path: Path, value) -> None:
    with pytest.raises(SpatialDimInvalid, match="INFO/SPATIAL_DIM is "):
        _dim_of(tmp_path, value)


@pytest.mark.parametrize("value, dim", [(v, d) for _, v, d in _VIEWER_READ],
                         ids=[c for c, _, _ in _VIEWER_READ])
def test_viewer_read_rows_read(tmp_path: Path, value, dim) -> None:
    assert _dim_of(tmp_path, value, coords_width=3 if dim != 3 else 2) == dim


@pytest.mark.parametrize("value", [None, np.zeros(0, np.int32)],
                         ids=["absent", "no value"])
def test_viewer_unknown_rows_infer(tmp_path: Path, value) -> None:
    assert _dim_of(tmp_path, value, coords_width=2) == 2


# -- the new refusals ---------------------------------------------------------

@pytest.mark.parametrize("value", [
    np.complex128(2 + 5j), np.complex128(2 + 0j), np.complex64(3 + 0j),
    np.array([2 + 0j]),
], ids=["2+5j", "2+0j", "complex64 3+0j", "[2+0j]"])
def test_complex_is_refused_by_name(tmp_path: Path, value) -> None:
    """A complex is refused, never read as its real part (#2244)."""
    with pytest.raises(SpatialDimInvalid) as info:
        _dim_of(tmp_path, value)
    assert "dim.h5: INFO/SPATIAL_DIM is " in str(info.value)
    assert "j" in str(info.value)  # the value is shown, imaginary part too


def _non_ascii_digits() -> list[str]:
    return [chr(c) for c in range(0x80, sys.maxunicode + 1)
            if unicodedata.decimal(chr(c), None) in (1, 2, 3)]


def test_every_non_ascii_digit_is_refused() -> None:
    """Every Unicode decimal digit 1, 2 or 3 outside ASCII (#2244).

    ``float()`` reads each as its value; the rule refuses each, alone and
    beside an ASCII digit.
    """
    digits = _non_ascii_digits()
    assert len(digits) >= 3 * 67  # every digit script, at Unicode 14 or later
    for d in digits:
        assert float(d) in (1.0, 2.0, 3.0), repr(d)  # what float() did
        assert _reads(d) is None, f"U+{ord(d):04X}"
        assert _reads(f" {d} ") is None, f"U+{ord(d):04X}"
        assert _reads(f"0{d}") is None, f"U+{ord(d):04X}"


def test_non_ascii_digit_through_a_file_is_refused(tmp_path: Path) -> None:
    for d in ("２", "٢", "\U0001d7d0", "२"):
        with pytest.raises(SpatialDimInvalid):
            _dim_of(tmp_path, d)


def _float_reads_two(s: str) -> bool:
    try:
        return float(s) == 2.0
    except ValueError:
        return False


def test_skipped_whitespace_is_floats_own() -> None:
    """Over every code point, the skipped set is ``float()``'s (#2205)."""
    for code in range(sys.maxunicode + 1):
        c = chr(code)
        if not c.isascii() and unicodedata.decimal(c, None) is not None:
            continue  # refused on purpose: test_every_non_ascii_digit_...
        for s in (c + "2", "2" + c):
            assert (_spatial_dim_number(s) == 2.0) == _float_reads_two(s), \
                f"U+{code:04X} in {s!r}"


@pytest.mark.parametrize("value, dim", [
    (np.bytes_(b"2"), 2), (np.bytes_(b" 3\n"), 3), (np.bytes_(b"2.0"), 2),
    (np.bytes_(b"\xa02"), None), (np.bytes_(b"\xef\xbc\x92"), None),
], ids=["b'2'", "b' 3\\n'", "b'2.0'", "b'\\xa02'", "utf-8 fullwidth 2"])
def test_fixed_length_bytes(tmp_path: Path, value, dim) -> None:
    """A fixed-length string reads as its ASCII text; other bytes refuse."""
    if dim is None:
        with pytest.raises(SpatialDimInvalid):
            _dim_of(tmp_path, value)
    else:
        assert _dim_of(tmp_path, value, coords_width=1) == dim


@pytest.mark.parametrize("value, dim", [
    (np.bool_(True), 1), (np.bool_(False), None), (np.float32(3.0), 3),
    (np.uint8(2), 2), (np.float64("nan"), None), (np.float64("inf"), None),
    ("+2", 2), ("2.", 2), (".2e1", 2), ("2e0_0", 2), ("1_0", None),
    ("2_", None), ("_2", None), ("0x2", None), ("inf", None), ("2\n", 2),
])
def test_float_literal_edges(value, dim) -> None:
    """Python's own literal rules, kept where the viewer keeps them."""
    assert _reads(value) == dim
