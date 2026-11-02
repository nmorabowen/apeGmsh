"""``results_hash``: the ``rf1`` fingerprint of a ``.ladruno`` (ADR 0130 A6).

``rf1`` is not a file digest. It is SHA-256 over identity attributes, the index
of every result and each stage's last slab, so it costs milliseconds and does
not move when a file is stamped: ``INFO/PROVENANCE`` is left out. The limit is
stated in A6: a rewrite of earlier steps under an identical last step is not
seen. The same limit covers every value outside the sample: a rewrite of
``ON_DOMAIN``, ``ON_REGIONS`` or ``ENVELOPES`` values that keeps each dataset's
shape and, where it has one, ``TIME`` and ``STEP`` is not seen.

The encoding, which the app's TypeScript (slice VU3a) must equal byte for byte.
``line(s)`` hashes ``s + "\\n"`` as UTF-8; arrays hash as little-endian bytes.

1. ``line("rf1")``, once.
2. For each part in partition order: ``line("<K>=<v>")`` for ``GENERATOR``,
   ``FORMAT_VERSION``, ``RUN_ID`` from ``INFO``'s attributes, ``-`` for one that
   is absent. A value is the attribute's first element, without trailing NULs.
3. For each ``MODEL_STAGE[i]`` by ``i``: ``line("MODEL_STAGE[i]|<KIND>")``. Then
   ``ON_NODES`` results by name, then ``ON_ELEMENTS`` results by name and their
   buckets by name. One without ``DATA`` is skipped.

   * A ``DATA`` dataset: ``line("<path>|<shape, comma-joined>|<kind><itemsize>")``,
     where ``path`` is ``ON_NODES/<r>`` or ``ON_ELEMENTS/<r>/<b>`` and ``kind`` is
     ``f`` or ``i``; then the bytes of ``TIME``, then of ``STEP``.
   * A ``DATA`` group (modes): ``STEP_<s>`` by ``s`` and ``MODE_<m>`` by ``m``:
     ``line("<path>/STEP_<s>/MODE_<m>|<shape>")``, then ``LAMBDA`` as ``f8``.

   Then every other child of ``RESULTS`` by name, descending by name: a group
   holding ``DATA`` is hashed as above, with no sample; any other group is
   descended into; a dataset outside such a group (``ENVELOPES/ON_NODES/<r>/MAX``)
   gives ``line("<path>|<shape>|<kind><itemsize>")``.

4. The stage's sample, once, after its results: the last slab (``DATA[T-1]``) of
   the first ``DATA`` dataset with ``T > 0``, or the last mode of the first modes
   result, whichever ``ON_NODES`` reaches first. With no such nodal result, the
   last slab of the first element bucket with ``T > 0``. None at all: nothing.

The element fallback of step 4 is the amendment's addition to the measured
algorithm; a stage with no nodal result would otherwise hash no field value.

A native ``results.h5`` uses ``native:<lineage hash>``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from ._errors import FingerprintRefused

__all__ = ["results_fingerprint"]

_STAGE = "MODEL_STAGE["
#: The families the sample comes from, in the order they are read.
_SAMPLED = ("ON_NODES", "ON_ELEMENTS")


def results_fingerprint(path: "str | Path") -> str:
    """``rf1:<hex>`` of a ``.ladruno`` or its part set, or ``native:<hash>``.

    ``path`` is one ``.ladruno``, any part of a partitioned set (every part is
    hashed, in partition order), or a native ``results.h5``. Anything else
    refuses with :class:`FingerprintRefused`: no hash is ever guessed.
    """
    from ..readers._ladruno_multi import discover_partition_files

    p = Path(path)
    if not p.is_file():
        raise FingerprintRefused(f"no such results file: {p}")
    try:
        parts = discover_partition_files(p)
    except ValueError as exc:
        raise FingerprintRefused(str(exc)) from exc

    h = hashlib.sha256()
    _line(h, "rf1")
    for part in parts:
        try:
            with h5py.File(part, "r") as f:
                if "INFO" not in f:
                    if len(parts) != 1:
                        raise FingerprintRefused(f"{part} has no INFO group")
                    return _native_hash(f, part)
                _hash_part(h, f, part)
        except FingerprintRefused:
            raise
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise FingerprintRefused(f"{part} cannot be fingerprinted: {exc}") from exc
    return "rf1:" + h.hexdigest()


def _native_hash(f: h5py.File, path: Path) -> str:
    if "meta" in f and "lineage" in f["meta"]:
        lineage = f["meta/lineage"].attrs
        if "results_hash" in lineage:
            text = _one(lineage["results_hash"])
            if text:
                return f"native:{text}"
    raise FingerprintRefused(
        f"{path} is neither a recorder results file (no INFO group) nor a "
        "native results file with /meta/lineage@results_hash"
    )


def _line(h: Any, text: str) -> None:
    h.update((text + "\n").encode("utf-8"))


def _one(value: Any) -> str:
    a = np.asarray(value).ravel()
    x = a[0] if a.size else ""
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return str(x).rstrip("\x00")


def _le(a: np.ndarray) -> bytes:
    return np.asarray(a).astype(a.dtype.newbyteorder("<")).tobytes()


def _index(name: str, prefix_len: int, where: str) -> int:
    try:
        return int(name[prefix_len:].rstrip("]"))
    except ValueError as exc:
        raise FingerprintRefused(f"{where}: {name!r} has no integer index") from exc


def _hash_part(h: Any, f: h5py.File, path: Path) -> None:
    info = f["INFO"]
    for key in ("GENERATOR", "FORMAT_VERSION", "RUN_ID"):
        _line(h, f"{key}={_one(info.attrs[key]) if key in info.attrs else '-'}")
    stages = [k for k in f if k.startswith(_STAGE)]
    for stage in sorted(stages, key=lambda n: _index(n, len(_STAGE), str(path))):
        group = f[stage]
        where = f"{path}:{stage}"
        if "KIND" not in group.attrs:
            raise FingerprintRefused(f"{where} has no KIND")
        _line(h, f"{stage}|{_one(group.attrs['KIND'])}")
        if "RESULTS" in group:
            _hash_stage_results(h, group["RESULTS"], where)


def _hash_stage_results(h: Any, results: h5py.Group, where: str) -> None:
    sample: Optional[np.ndarray] = None
    for family in _SAMPLED:
        if family not in results:
            continue
        for name in sorted(results[family]):
            result = results[family][name]
            if family == "ON_NODES":
                holders = [(f"{family}/{name}", result)]
            else:
                holders = [(f"{family}/{name}/{b}", result[b]) for b in sorted(result)]
            for path, holder in holders:
                if not isinstance(holder, h5py.Group):
                    raise FingerprintRefused(f"{where}/{path} is not a result group")
                if "DATA" not in holder:
                    continue
                data = holder["DATA"]
                here = f"{where}/{path}"
                # Only the sample's slab is read: reading every result's last
                # slab would cost what the fingerprint exists to avoid.
                want = sample is None
                if isinstance(data, h5py.Dataset):
                    found = _hash_dataset(h, holder, data, path, here, want)
                else:
                    if family != "ON_NODES":
                        raise FingerprintRefused(f"{here}/DATA is a group on an element bucket")
                    found = _hash_modes(h, data, path, here, want)
                if want:
                    sample = found
    _hash_index(h, results, "", where, skip=_SAMPLED)
    if sample is not None:
        h.update(_le(sample))


def _hash_index(
    h: Any, group: h5py.Group, prefix: str, where: str, skip: tuple[str, ...] = (),
) -> None:
    """The index of everything the sample does not come from, with no values."""
    for name in sorted(group):
        if name in skip:
            continue
        child = group[name]
        path = f"{prefix}{name}"
        here = f"{where}/{path}"
        if isinstance(child, h5py.Dataset):
            _line(h, f"{path}|{_shape(child)}|{_kind(child, here)}")
        elif "DATA" not in child:
            _hash_index(h, child, f"{path}/", where)
        elif isinstance(child["DATA"], h5py.Dataset):
            _hash_dataset(h, child, child["DATA"], path, here, False)
        else:
            _hash_modes(h, child["DATA"], path, here, False)


def _shape(ds: h5py.Dataset) -> str:
    return ",".join(map(str, ds.shape))


def _kind(ds: h5py.Dataset, where: str) -> str:
    if ds.dtype.kind not in "fi":
        raise FingerprintRefused(f"{where} has dtype {ds.dtype}; rf1 knows f and i")
    return f"{ds.dtype.kind}{ds.dtype.itemsize}"


def _hash_dataset(
    h: Any, holder: h5py.Group, data: h5py.Dataset, path: str, where: str, want: bool,
) -> Optional[np.ndarray]:
    for child in ("TIME", "STEP"):
        if child not in holder:
            raise FingerprintRefused(f"{where} has no {child}")
    _line(h, f"{path}|{_shape(data)}|{_kind(data, f'{where}/DATA')}")
    h.update(_le(holder["TIME"][...]))
    h.update(_le(holder["STEP"][...]))
    return data[data.shape[0] - 1] if want and data.shape[0] > 0 else None


def _hash_modes(
    h: Any, data: h5py.Group, path: str, where: str, want: bool,
) -> Optional[np.ndarray]:
    last: Optional[h5py.Dataset] = None
    for step in sorted(data, key=lambda n: _index(n, len("STEP_"), where)):
        for mode in sorted(data[step], key=lambda n: _index(n, len("MODE_"), where)):
            ds = data[step][mode]
            if "LAMBDA" not in ds.attrs:
                raise FingerprintRefused(f"{where}/{step}/{mode} has no LAMBDA")
            value = np.asarray(ds.attrs["LAMBDA"], dtype="<f8").ravel()
            if value.size != 1:
                raise FingerprintRefused(f"{where}/{step}/{mode}@LAMBDA holds {value.size} values")
            _line(h, f"{path}/{step}/{mode}|{_shape(ds)}")
            h.update(value.tobytes())
            last = ds
    return last[...] if want and last is not None else None
