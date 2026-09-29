"""Canonical text dump of an HDF5 file: one line per group, dataset and attr.

Usage::

    python scripts/h5_canonical_dump.py model.h5

Each line names an object by path and pins its dtype, shape and the sha1
of a canonical byte encoding of its value.  Two files dump to the same
text iff they hold the same tree, the same dtypes and shapes and the same
values (floats compared at ``FLOAT_SIG_DIGITS`` significant digits),
whatever HDF5 chunking, object order on disk, byte order or
creation time they have.  The golden emit corpus
(``tests/opensees/golden/``) compares these dumps; see its README for why
each mask exists.

Line grammar (sorted depth-first by name; a group's attrs precede its
children)::

    G <path>
    L <path> -> <soft-link target>
    A <path>@<name> dtype=<label> shape=<tuple> sha1=<hex | masked>
    D <path> dtype=<label> shape=<tuple> sha1=<hex>

Only stdlib, numpy and h5py; importable (``dump``) as well as runnable.
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Attribute values that legitimately differ between two writes of the same
# model.  Keyed ``<group path>@<attr name>``; the value is the reason.  An
# attr is masked only at the exact path listed: a wall-clock field that
# appears anywhere else is a determinism bug the dump must expose.
MASKS: dict[str, str] = {
    "/meta@created_iso": "wall-clock write time",
    "/meta@apeGmsh_version": (
        "release identity of the installed distribution, not emit output"
    ),
    "/meta/lineage@model_hash": (
        "derived digest: blake2b over the raw float bytes under /opensees "
        "(lineage.compute_model_hash), so a last-ulp libm difference "
        "changes it; every dataset it summarises is pinned line by line"
    ),
}


# Floating-point values hash through their text at FLOAT_SIG_DIGITS
# significant digits, with |x| < FLOAT_ZERO_FLOOR hashed as 0.  Values
# computed through libm (the orientation vecxz: sin/cos/sqrt) differ in the
# last ulp between platforms (#1258); 12 digits absorb that while a relative
# change of 1e-11 or more still changes the sha1.  The dtype label still
# pins the stored width (<f4 vs <f8).
FLOAT_SIG_DIGITS = 12
FLOAT_ZERO_FLOOR = 1e-15


def _float_text(x: float) -> str:
    if not math.isfinite(x):
        return repr(x)
    if abs(x) < FLOAT_ZERO_FLOOR:
        x = 0.0  # also folds -0.0
    return format(x, f".{FLOAT_SIG_DIGITS - 1}e")


def dtype_label(dt: np.dtype) -> str:
    """A platform-independent name for an HDF5-backed numpy dtype."""
    info = h5py.check_string_dtype(dt)
    if info is not None:
        length = "vlen" if info.length is None else str(info.length)
        return f"str[{info.encoding},{length}]"
    base = h5py.check_vlen_dtype(dt)
    if base is not None:
        return f"vlen[{dtype_label(np.dtype(base))}]"
    if dt.fields is not None:
        fields = ",".join(
            f"{name}:{dtype_label(dt.fields[name][0])}"
            for name in dt.names or ()
        )
        return "{" + fields + "}"
    if dt.subdtype is not None:
        sub, shape = dt.subdtype
        return f"{dtype_label(sub)}{list(shape)}"
    return dt.newbyteorder("<").str


def _frame(payload: bytes) -> bytes:
    """Length-prefix one chunk so concatenations cannot collide."""
    return struct.pack("<Q", len(payload)) + payload


def canonical_bytes(value: Any) -> bytes:
    """A byte encoding of ``value`` independent of platform and byte order."""
    if isinstance(value, h5py.Empty):
        return b"E"
    if isinstance(value, bytes):
        return b"b" + _frame(value)
    if isinstance(value, str):
        return b"s" + _frame(value.encode("utf-8"))
    arr = np.asarray(value)
    dt = arr.dtype
    head = b"a" + _frame(repr(arr.shape).encode("ascii"))
    if dt.fields is not None:
        return head + b"".join(
            _frame(name.encode("utf-8")) + canonical_bytes(arr[name])
            for name in dt.names or ()
        )
    if dt.kind == "O":
        return head + b"".join(
            _frame(canonical_bytes(item)) for item in arr.ravel(order="C")
        )
    if dt.kind == "U":
        return head + b"".join(
            b"s" + _frame(str(item).encode("utf-8"))
            for item in arr.ravel(order="C")
        )
    if dt.kind == "f":
        return head + b"f" + _frame(" ".join(
            _float_text(float(x)) for x in arr.ravel(order="C")
        ).encode("ascii"))
    if dt.kind == "c":
        return head + b"c" + _frame(" ".join(
            f"{_float_text(float(z.real))},{_float_text(float(z.imag))}"
            for z in arr.ravel(order="C")
        ).encode("ascii"))
    if dt.kind in "SV" or dt.kind == "b":
        return head + _frame(np.ascontiguousarray(arr).tobytes())
    little = np.ascontiguousarray(arr, dtype=dt.newbyteorder("<"))
    return head + _frame(little.tobytes())


def _sha1(value: Any) -> str:
    return hashlib.sha1(canonical_bytes(value)).hexdigest()


def _attr_lines(obj: h5py.HLObject, path: str) -> list[str]:
    lines = []
    for name in sorted(obj.attrs):
        key = f"{path}@{name}"
        aid = obj.attrs.get_id(name)
        label = dtype_label(aid.dtype)
        if key in MASKS:
            digest = "masked"
        else:
            digest = _sha1(obj.attrs[name])
        lines.append(
            f"A {key} dtype={label} shape={aid.shape} "
            f"sha1={digest}"
        )
    return lines


def _walk(group: h5py.Group, path: str, out: list[str]) -> None:
    out.append(f"G {path}")
    out.extend(_attr_lines(group, path))
    for name in sorted(group):
        child_path = f"{path.rstrip('/')}/{name}"
        link = group.get(name, getlink=True)
        if isinstance(link, h5py.SoftLink):
            out.append(f"L {child_path} -> {link.path}")
            continue
        if isinstance(link, h5py.ExternalLink):
            out.append(f"L {child_path} -> {link.filename}:{link.path}")
            continue
        child = group[name]
        if isinstance(child, h5py.Group):
            _walk(child, child_path, out)
        elif isinstance(child, h5py.Dataset):
            shape = child.shape
            value = h5py.Empty(child.dtype) if shape is None else child[()]
            out.append(
                f"D {child_path} dtype={dtype_label(child.dtype)} "
                f"shape={shape} sha1={_sha1(value)}"
            )
            out.extend(_attr_lines(child, child_path))
        else:  # pragma: no cover - committed types / other HDF5 objects
            raise TypeError(
                f"h5_canonical_dump: unsupported HDF5 object at "
                f"{child_path}: {type(child).__name__}"
            )


def dump(path: "str | Path") -> str:
    """Return the canonical dump of the HDF5 file at ``path``."""
    out: list[str] = []
    with h5py.File(path, "r") as f:
        _walk(f, "/", out)
    return "\n".join(out) + "\n"


def main(argv: "list[str] | None" = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python scripts/h5_canonical_dump.py FILE.h5",
              file=sys.stderr)
        return 2
    sys.stdout.write(dump(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
