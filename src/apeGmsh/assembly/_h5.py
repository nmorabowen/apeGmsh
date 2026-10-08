"""The ``/assembly`` zone of an assembly archive (ADR 0117 D5).

An assembly archive is a plain ``model.h5`` (``apeSees.h5``) plus one
optional root zone that re-lists what was assembled::

    /assembly                @name
      /instances   label · source_path · source_fem_hash ·
                   source_opensees_hash · translate(3) · rotate(4) ·
                   fem_id_base · fem_id_span · partition_rank
      /ties        name · kind · master · slave · params · n_records

with ``/meta@assembly_schema_version`` as its own version key (ADR 0112
D2). Every table is a group of equal-length column datasets, the layout of
``/provenance``. The flat zones stay authoritative: no reader of the model
reads this zone, and nothing here is hashed (``fem_hash`` reads the
neutral zone, ``model_hash`` ``/opensees``).

Encodings, each exact both ways:

* ``rotate`` is ``(ax, ay, az, theta)``; a declared rotation never has a
  zero axis (``check_rotate`` refuses one), so the all-zero row is "no
  rotation";
* ``partition_rank`` is ``-1`` for none (a rank is ``>= 0``);
* a tie ``name`` is ``""`` when unnamed (``check_label`` refuses ``""``);
* ``params`` is the tie's options as canonical JSON (sorted keys).

:func:`write_assembly_zone` validates every row and builds every column
before it opens the file, replaces an existing zone rather than appending
to it, and removes its own partial group if HDF5 fails part-way.
:func:`read_assembly_zone` refuses a file without the zone, a stamp
outside the reader's range, a missing column, ragged columns and an
unknown tie kind.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._v1 import AssemblyError

__all__ = [
    "TIE_KINDS",
    "TIE_PARAMS",
    "AssemblyZone",
    "InstanceRow",
    "TieRow",
    "read_assembly_zone",
    "validate_rows",
    "write_assembly_zone",
]

#: The ``/assembly`` root group.
ZONE_GROUP = "assembly"

#: The ``params`` JSON keys of each kind a ``/assembly/ties`` row records
#: (ADR 0117 D3). ``tie`` is AS3's; AS4 adds the couplings and ``node``, the
#: assembly-owned reference node (``master`` and ``slave`` are ``""``, the
#: name is the node's, ``n_records`` is 1). A kind outside this table, or a
#: row whose params carry other keys, is refused on write; an unknown kind
#: is refused on read.
TIE_PARAMS: dict[str, frozenset[str]] = {
    "tie": frozenset({"dofs", "enforce", "method", "tolerance"}),
    "equal_dof": frozenset({"dofs", "tolerance"}),
    "rigid_link": frozenset({"link_type", "master_point"}),
    "rigid_diaphragm": frozenset({
        "constrained_dofs", "master_point", "plane_normal", "plane_tolerance"}),
    "embedded": frozenset({"stiffness", "tolerance"}),
    "kinematic_coupling": frozenset({"dofs"}),
    "distributing_coupling": frozenset({"weighting"}),
    "node": frozenset({"coords"}),
}

#: Tie kinds the zone records; a kind outside this set is refused on
#: write and on read.
TIE_KINDS: frozenset[str] = frozenset(TIE_PARAMS)

_INSTANCE_STR = ("label", "source_path", "source_fem_hash", "source_opensees_hash")
_INSTANCE_INT = ("fem_id_base", "fem_id_span", "partition_rank")
_INSTANCE_VEC = {"translate": 3, "rotate": 4}
_TIE_STR = ("name", "kind", "master", "slave", "params")
_TIE_INT = ("n_records",)


@dataclass(frozen=True)
class InstanceRow:
    """One row of ``/assembly/instances``."""

    label: str
    source_path: str
    source_fem_hash: str
    source_opensees_hash: str
    translate: tuple[float, float, float]
    #: ``(ax, ay, az, theta)``; all zero when the instance is not rotated.
    rotate: tuple[float, float, float, float]
    fem_id_base: int
    fem_id_span: int
    #: ``-1`` when the instance has no rank hint.
    partition_rank: int


@dataclass(frozen=True)
class TieRow:
    """One row of ``/assembly/ties``."""

    #: ``""`` for an unnamed tie.
    name: str
    kind: str
    master: str
    slave: str
    #: The tie's options as canonical JSON (sorted keys).
    params: str
    n_records: int


@dataclass(frozen=True)
class AssemblyZone:
    """What :func:`read_assembly_zone` returns."""

    name: str
    version: str
    instances: tuple[InstanceRow, ...]
    ties: tuple[TieRow, ...]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_assembly_zone(
    path: "str | Path",
    name: str,
    instances: Sequence[InstanceRow],
    ties: Sequence[TieRow],
) -> None:
    """Write ``/assembly`` into the existing ``model.h5`` at ``path``.

    Replaces a zone already there. Raises :class:`AssemblyError` for an
    invalid row and ``MalformedH5Error`` for a file without ``/meta``,
    in both cases before the file changes.
    """
    import h5py

    from apeGmsh.opensees._internal.schema_version import (
        ASSEMBLY_KEY,
        ASSEMBLY_SCHEMA_VERSION,
    )
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    columns = _columns(name, instances, ties)
    with h5py.File(str(path), "r+") as f:
        if "meta" not in f:
            raise MalformedH5Error(
                f"{path}: no /meta group; /assembly is written into a "
                f"model.h5 that apeSees.h5 wrote."
            )
        meta = f["meta"]
        if ZONE_GROUP in f:
            del f[ZONE_GROUP]
        if ASSEMBLY_KEY in meta.attrs:
            del meta.attrs[ASSEMBLY_KEY]
        try:
            _write_columns(f, name, columns)
            meta.attrs[ASSEMBLY_KEY] = ASSEMBLY_SCHEMA_VERSION
        except BaseException:
            if ZONE_GROUP in f:
                del f[ZONE_GROUP]
            if ASSEMBLY_KEY in meta.attrs:
                del meta.attrs[ASSEMBLY_KEY]
            raise


def validate_rows(
    name: str, instances: Sequence[InstanceRow], ties: Sequence[TieRow],
) -> None:
    """Raise :class:`AssemblyError` for any row :func:`write_assembly_zone`
    would refuse, without touching a file. ``Assembly.h5`` calls it before
    ``apeSees.h5`` overwrites the target."""
    _columns(name, instances, ties)


def _columns(
    name: str, instances: Sequence[InstanceRow], ties: Sequence[TieRow],
) -> dict[str, dict[str, np.ndarray]]:
    """Validate every row and return the column arrays of both tables."""
    if not isinstance(name, str) or not name:
        raise AssemblyError(f"/assembly: the assembly name must be a non-empty string, got {name!r}.")
    labels = [r.label for r in instances]
    if not labels:
        raise AssemblyError("/assembly: an assembly archive has at least one instance.")
    if len(set(labels)) != len(labels):
        raise AssemblyError(f"/assembly: instance labels repeat: {labels}.")
    for r in instances:
        for col in _INSTANCE_STR:
            if not isinstance(getattr(r, col), str):
                raise AssemblyError(f"/assembly instance {r.label!r}: {col} must be a string.")
        if not r.label or not r.source_path or not r.source_fem_hash or not r.source_opensees_hash:
            raise AssemblyError(
                f"/assembly instance {r.label!r}: label, source_path and both "
                f"source hashes must be non-empty."
            )
        for col, n in _INSTANCE_VEC.items():
            v = getattr(r, col)
            if len(v) != n or not all(math.isfinite(float(x)) for x in v):
                raise AssemblyError(
                    f"/assembly instance {r.label!r}: {col} must be {n} finite numbers, got {v!r}."
                )
        if r.fem_id_base < 1 or r.fem_id_span < 1 or r.partition_rank < -1:
            raise AssemblyError(
                f"/assembly instance {r.label!r}: fem_id_base and fem_id_span must "
                f"be >= 1 and partition_rank >= -1, got {r.fem_id_base}, "
                f"{r.fem_id_span}, {r.partition_rank}."
            )
    tie_names = [t.name for t in ties if t.name]
    if len(set(tie_names)) != len(tie_names):
        raise AssemblyError(f"/assembly: tie names repeat: {tie_names}.")
    for t in ties:
        for col in _TIE_STR:
            if not isinstance(getattr(t, col), str):
                raise AssemblyError(f"/assembly tie {t.name!r}: {col} must be a string.")
        if t.kind not in TIE_KINDS:
            raise AssemblyError(
                f"/assembly tie {t.name!r}: kind {t.kind!r} is not one of {sorted(TIE_KINDS)}."
            )
        try:
            params = json.loads(t.params)
        except ValueError as exc:
            raise AssemblyError(f"/assembly tie {t.name!r}: params is not JSON: {exc}") from exc
        if not isinstance(params, dict):
            raise AssemblyError(f"/assembly tie {t.name!r}: params must be a JSON object.")
        if set(params) != TIE_PARAMS[t.kind]:
            raise AssemblyError(
                f"/assembly tie {t.name!r}: {t.kind} params carry "
                f"{sorted(params)}, expected {sorted(TIE_PARAMS[t.kind])}."
            )
        if t.n_records < 1:
            raise AssemblyError(
                f"/assembly tie {t.name!r}: n_records={t.n_records}; a tie resolves "
                f"at least one record (ADR 0117 INV-7)."
            )
    inst: dict[str, np.ndarray] = {
        col: np.array([getattr(r, col) for r in instances], dtype=object)
        for col in _INSTANCE_STR
    }
    inst.update({
        col: np.array([getattr(r, col) for r in instances], dtype=np.int64)
        for col in _INSTANCE_INT
    })
    inst.update({
        col: np.array([getattr(r, col) for r in instances], dtype=np.float64).reshape(-1, n)
        for col, n in _INSTANCE_VEC.items()
    })
    tie: dict[str, np.ndarray] = {
        col: np.array([getattr(t, col) for t in ties], dtype=object).reshape(-1)
        for col in _TIE_STR
    }
    tie.update({
        col: np.array([getattr(t, col) for t in ties], dtype=np.int64).reshape(-1)
        for col in _TIE_INT
    })
    return {"instances": inst, "ties": tie}


def _write_columns(
    f: Any, name: str, columns: dict[str, dict[str, np.ndarray]],
) -> None:
    import h5py

    str_dt = h5py.string_dtype(encoding="utf-8")
    grp = f.create_group(ZONE_GROUP)
    grp.attrs["name"] = name
    for table, cols in columns.items():
        sub = grp.create_group(table)
        for col in sorted(cols):
            data = cols[col]
            if data.dtype == object:
                sub.create_dataset(col, data=data, dtype=str_dt)
            else:
                sub.create_dataset(col, data=data)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_assembly_zone(path: "str | Path") -> AssemblyZone:
    """Read ``/assembly`` from ``path``.

    Raises :class:`AssemblyError` when the file has no ``/assembly`` zone
    (a plain ``model.h5``), ``SchemaVersionError`` for a stamp outside the
    reader's range, and ``MalformedH5Error`` for a zone without its stamp,
    a missing column, ragged columns or an unknown tie kind.
    """
    import h5py

    from apeGmsh.opensees._internal.schema_version import (
        ASSEMBLY,
        read_zone_version,
        reader_version,
        validate_zone_version,
    )
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    with h5py.File(str(path), "r") as f:
        if ZONE_GROUP not in f:
            raise AssemblyError(
                f"{path}: no /assembly zone. The file is a plain model.h5; "
                f"open it with FEMData.from_h5 or OpenSeesModel.from_h5."
            )
        if "meta" not in f:
            raise MalformedH5Error(f"{path}: /assembly is present but /meta is missing.")
        version = read_zone_version(f["meta"].attrs, ASSEMBLY)
        if version is None:
            raise MalformedH5Error(
                f"{path}: /assembly is present but /meta carries no "
                f"assembly_schema_version."
            )
        validate_zone_version(version, reader_version(ASSEMBLY), zone=ASSEMBLY)
        grp = f[ZONE_GROUP]
        if "name" not in grp.attrs:
            raise MalformedH5Error(f"{path}: /assembly@name is missing.")
        name = _str(grp.attrs["name"])
        inst = _read_table(
            grp, "instances", (*_INSTANCE_STR, *_INSTANCE_INT, *_INSTANCE_VEC), path)
        tie = _read_table(grp, "ties", (*_TIE_STR, *_TIE_INT), path)

    instances = tuple(
        InstanceRow(
            label=_str(inst["label"][i]),
            source_path=_str(inst["source_path"][i]),
            source_fem_hash=_str(inst["source_fem_hash"][i]),
            source_opensees_hash=_str(inst["source_opensees_hash"][i]),
            translate=_vec3(inst["translate"][i]),
            rotate=_vec4(inst["rotate"][i]),
            fem_id_base=int(inst["fem_id_base"][i]),
            fem_id_span=int(inst["fem_id_span"][i]),
            partition_rank=int(inst["partition_rank"][i]),
        )
        for i in range(len(inst["label"]))
    )
    ties = tuple(
        TieRow(
            name=_str(tie["name"][i]),
            kind=_str(tie["kind"][i]),
            master=_str(tie["master"][i]),
            slave=_str(tie["slave"][i]),
            params=_str(tie["params"][i]),
            n_records=int(tie["n_records"][i]),
        )
        for i in range(len(tie["name"]))
    )
    for t in ties:
        if t.kind not in TIE_KINDS:
            raise MalformedH5Error(
                f"{path}: /assembly/ties row {t.name!r} has kind {t.kind!r}, "
                f"which this reader does not know ({sorted(TIE_KINDS)})."
            )
    return AssemblyZone(name=name, version=str(version), instances=instances, ties=ties)


def _read_table(
    grp: Any, table: str, cols: Sequence[str], path: "str | Path",
) -> dict[str, np.ndarray]:
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    if table not in grp:
        raise MalformedH5Error(f"{path}: /assembly/{table} is missing.")
    sub = grp[table]
    out: dict[str, np.ndarray] = {}
    for col in cols:
        if col not in sub:
            raise MalformedH5Error(f"{path}: /assembly/{table}/{col} is missing.")
        out[col] = np.asarray(sub[col][()])
    for col, n in _INSTANCE_VEC.items():
        if col in out and (out[col].ndim != 2 or out[col].shape[1] != n):
            raise MalformedH5Error(
                f"{path}: /assembly/{table}/{col} has shape {out[col].shape}, "
                f"expected (n, {n})."
            )
    lengths = {col: len(v) for col, v in out.items()}
    if len(set(lengths.values())) > 1:
        raise MalformedH5Error(
            f"{path}: /assembly/{table} columns differ in length: {lengths}."
        )
    return out


def _str(raw: object) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _vec3(row: np.ndarray) -> tuple[float, float, float]:
    a, b, c = (float(v) for v in row)
    return (a, b, c)


def _vec4(row: np.ndarray) -> tuple[float, float, float, float]:
    a, b, c, d = (float(v) for v in row)
    return (a, b, c, d)
