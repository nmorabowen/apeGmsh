"""Write a ``<stem>.user.h5``: atomic, and never past a zone it does not know.

Every write builds ``<stem>.user.h5.partial`` in the same folder, reopens and
verifies it, and only then replaces the file, so a killed writer leaves the
previous file whole (ADR 0130 D2, A4).

Rules this module holds:

* ``/meta`` hashes come from the model archive's ``/meta/lineage``, byte-equal.
* A zone it does not write is copied verbatim, whatever its version. A zone it
  does write refuses when the file's version is outside its range (A4).
* ``combinations`` adds a generation; earlier ones stay as the backup (A4).
* A hash change never silently re-labels what was derived under the old hash.
  Those entries are rewritten in the same call, or ``drop_stale=True`` drops
  them; otherwise the write refuses.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

from ... import __version__
from ...opensees._internal.lineage import read_stored_lineage
from ...opensees._internal.schema_version import (
    COMBINATIONS,
    COMBINATIONS_KEY,
    COMBINATIONS_SCHEMA_VERSION,
    USER_META_KEY,
    USER_META_SCHEMA_VERSION,
    USER_RESULTS,
    USER_RESULTS_KEY,
    USER_RESULTS_SCHEMA_VERSION,
    read_zone_version,
    reader_version,
    validate_zone_version,
)
from ._descriptor import Combination, resolve, write_generation
from ._errors import UserWriteRefused
from ._fields import EntryRefused, UserResult, write_entry
from ._fingerprint import results_fingerprint
from ._names import name_problem, text_or_empty
from ._reader import UserFile, read_model_combinations

__all__ = ["model_hashes", "write_user_file"]

SUFFIX = ".user.h5"
#: The zones that go stale with the model hashes (A6), and their /meta keys.
_MODEL_ZONES = {
    "modes": "modes_schema_version",
    "spectra": "spectra_schema_version",
    "entities": "entities_schema_version",
}
_GENERATION = re.compile(r"g(\d+)\Z")
_ENTRY = re.compile(r"r(\d+)\Z")
_REPLACE_TRIES = 10
_OPEN_TRIES = 100
_REPLACE_WAIT_S = 0.1


def model_hashes(model_h5: "str | Path") -> tuple[str, Optional[str]]:
    """``(fem_hash, model_hash)`` from ``/meta/lineage``, exactly as stored.

    ``fem_hash`` is required. ``model_hash`` is ``None`` for a model that has
    only ``fem_hash`` (ADR 0130 A1).
    """
    path = Path(model_h5)
    if not path.is_file():
        raise UserWriteRefused(f"no such model archive: {path}")
    try:
        with h5py.File(path, "r") as f:
            if "meta" not in f:
                raise UserWriteRefused(f"{path} has no /meta")
            fem_hash, model_hash, _ = read_stored_lineage(f["meta"])
    except (KeyError, OSError, TypeError) as exc:
        raise UserWriteRefused(f"{path} is not a readable model archive: {exc}") from exc
    if not fem_hash:
        raise UserWriteRefused(f"{path}: /meta/lineage has no fem_hash to pair with")
    return fem_hash, model_hash or None


def write_user_file(
    path: "str | Path",
    model_h5: "str | Path",
    *,
    results_file: "Optional[str | Path]" = None,
    combinations: Optional[Sequence[Combination]] = None,
    results: Sequence[UserResult] = (),
    units: Optional[str] = None,
    writer: Optional[str] = None,
    drop_stale: bool = False,
) -> Path:
    """Create or update ``path`` atomically and return it.

    ``model_h5`` supplies the pairing hashes. ``results_file`` is the paired
    ``.ladruno`` (or native file); its fingerprint becomes ``@results_hash``,
    and leaving it out keeps the file's current one. ``combinations`` replaces
    the table by adding a generation, and ``None`` leaves the zone untouched.
    ``results`` adds entries, or replaces the entry of the same name.
    """
    target = Path(path)
    if not target.name.endswith(SUFFIX) or target.name == SUFFIX:
        raise UserWriteRefused(f"{target}: a user file is named <stem>{SUFFIX}")
    fem_hash, model_hash = model_hashes(model_h5)
    previous = UserFile(target) if target.exists() else None
    results_hash = (
        results_fingerprint(results_file) if results_file is not None
        else (previous.results_hash if previous is not None else None)
    )
    names = _check_results(results, results_hash)
    if combinations is not None:
        _check_combinations(combinations, model_h5)

    model_moved = results_moved = False
    if previous is not None:
        # model_hash appearing or disappearing counts: what was derived under
        # the old identity would otherwise be re-labelled by this write. With
        # nothing derived, _derived_to_drop finds nothing and the write stands.
        model_moved = fem_hash != previous.fem_hash or model_hash != previous.model_hash
        results_moved = results_hash != previous.results_hash
        # /results must be understood to judge it, even when none of it is rewritten.
        _check_known_versions(
            target, combinations is not None,
            bool(results) or drop_stale or model_moved or results_moved,
        )
    dropped = _derived_to_drop(target, model_moved, results_moved, names, drop_stale)

    partial = target.with_name(target.name + ".partial")
    out = _open_partial(partial)  # outside the try: a failed open is another writer's file
    try:
        with out:
            _write_all(out, previous, target, fem_hash, model_hash, results_hash,
                       combinations, results, units, writer, dropped)
        _verify(partial, model_h5, fem_hash, model_hash, results_hash, combinations, names)
        if previous is not None and UserFile(target).revision != previous.revision:
            raise UserWriteRefused(
                f"{target} changed while this write ran (revision "
                f"{previous.revision} moved); nothing was replaced. Write again.")
        _fsync(partial)
        _replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target


# ---------------------------------------------------------------------------
# Checks before anything is written
# ---------------------------------------------------------------------------


def _check_results(results: Sequence[UserResult], results_hash: Optional[str]) -> set[str]:
    names: set[str] = set()
    for r in results:
        problem = name_problem(r.name)
        if problem is not None:
            raise UserWriteRefused(f"user result name {r.name!r}: {problem}")
        if r.name in names:
            raise UserWriteRefused(f"user result {r.name!r} is given twice")
        names.add(r.name)
        if r.basis == "results" and results_hash is None:
            raise UserWriteRefused(
                f"user result {r.name!r} has @basis=results, and this file has no "
                "results_hash to pair it with: pass results_file=.")
    return names


def _check_combinations(rows: Sequence[Combination], model_h5: "str | Path") -> None:
    table = resolve(rows, read_model_combinations(model_h5))
    # A bad model.h5 row that touches none of these is that file's refusal to
    # report at read; a clash refuses the row here as well, so it is caught.
    lines = [f"{r.name!r}: {r.reason}: {r.detail}" for r in table.refused if r.origin == "user"]
    if lines:
        raise UserWriteRefused("combinations refused: " + "; ".join(lines))


def _check_known_versions(target: Path, combinations: bool, results: bool) -> None:
    """No writer touches a zone newer than it knows (A4)."""
    with h5py.File(target, "r") as f:
        attrs = f["meta"].attrs
        for wanted, zone in ((combinations, COMBINATIONS), (results, USER_RESULTS)):
            version = read_zone_version(attrs, zone) if wanted else None
            if version is not None:
                validate_zone_version(version, reader_version(zone), zone=zone)


def _derived_to_drop(
    target: Path,
    model_moved: bool,
    results_moved: bool,
    rewritten: set[str],
    drop_stale: bool,
) -> set[str]:
    """Links (``results/r3``, ``modes``) a hash change leaves behind.

    Refuses unless ``drop_stale``: keeping them under the new hash would draw
    old values as current.
    """
    if not (model_moved or results_moved):
        return set()
    losing: set[str] = set()
    with h5py.File(target, "r") as f:
        if model_moved:
            losing |= {root for root in _MODEL_ZONES if root in f}
        if "results" in f and isinstance(f["results"], h5py.Group):
            for link in f["results"]:
                group = f["results"][link]
                if not _ENTRY.match(link) or not isinstance(group, h5py.Group):
                    continue
                basis = text_or_empty(group.attrs["basis"]) if "basis" in group.attrs else ""
                name = text_or_empty(group.attrs["name"]) if "name" in group.attrs else ""
                if name in rewritten:
                    continue
                if (basis == "results" and results_moved) or (basis == "model" and model_moved):
                    losing.add(f"results/{link}")
    if losing and not drop_stale:
        raise UserWriteRefused(
            f"{target}: the pairing hashes changed, and these would become stale: "
            f"{sorted(losing)}. Rewrite them in this call, or pass drop_stale=True.")
    return losing


# ---------------------------------------------------------------------------
# The write
# ---------------------------------------------------------------------------


def _write_all(
    out: h5py.File,
    previous: Optional[UserFile],
    target: Path,
    fem_hash: str,
    model_hash: Optional[str],
    results_hash: Optional[str],
    combinations: Optional[Sequence[Combination]],
    results: Sequence[UserResult],
    units: Optional[str],
    writer: Optional[str],
    dropped: set[str],
) -> None:
    meta = out.create_group("meta")
    if previous is not None:
        with h5py.File(target, "r") as src:
            for key, value in src["meta"].attrs.items():
                meta.attrs[key] = value
            for root in src:
                if root != "meta" and root not in dropped:
                    src.copy(src[root], out, root)
            for link in dropped:
                if link.startswith("results/"):
                    del out[link]
    for root, key in _MODEL_ZONES.items():
        if root in dropped and key in meta.attrs:
            del meta.attrs[key]

    meta.attrs[USER_META_KEY] = USER_META_SCHEMA_VERSION
    meta.attrs["fem_hash"] = fem_hash
    _set_or_clear(meta, "model_hash", model_hash)
    _set_or_clear(meta, "results_hash", results_hash)
    meta.attrs["writer"] = writer if writer is not None else f"apeGmsh {__version__}"
    if units is not None:
        meta.attrs["units"] = units
    elif "units" not in meta.attrs:
        meta.attrs["units"] = ""
    meta.attrs["revision"] = np.int32(1 if previous is None else previous.revision + 1)

    if combinations is not None:
        _add_generation(out, combinations)
        meta.attrs[COMBINATIONS_KEY] = COMBINATIONS_SCHEMA_VERSION
    if results:
        _put_results(out, results)
        meta.attrs[USER_RESULTS_KEY] = USER_RESULTS_SCHEMA_VERSION


def _set_or_clear(group: h5py.Group, key: str, value: Optional[str]) -> None:
    if value is not None:
        group.attrs[key] = value
    elif key in group.attrs:
        del group.attrs[key]


def _add_generation(out: h5py.File, rows: Sequence[Combination]) -> None:
    zone = out.require_group("combinations")
    held = [int(m.group(1)) for m in map(_GENERATION.match, zone) if m]
    n = max(held) + 1 if held else 0
    write_generation(zone.create_group(f"g{n}"), rows)
    zone.attrs["current"] = np.int32(n)


def _put_results(out: h5py.File, results: Sequence[UserResult]) -> None:
    zone = out.require_group("results")
    by_name: dict[str, str] = {}
    numbers: list[int] = []
    for link in zone:
        m = _ENTRY.match(link)
        if m is None:
            continue
        numbers.append(int(m.group(1)))
        group = zone[link]
        if isinstance(group, h5py.Group) and "name" in group.attrs:
            by_name[text_or_empty(group.attrs["name"])] = link
    next_k = max(numbers) + 1 if numbers else 0
    for r in results:
        link = by_name.get(r.name)
        if link is None:
            link = f"r{next_k}"
            next_k += 1
        else:
            del zone[link]
        try:
            write_entry(zone.create_group(link), r)
        except EntryRefused as exc:
            raise UserWriteRefused(f"user result {r.name!r}: {exc}") from exc


def _verify(
    partial: Path,
    model_h5: "str | Path",
    fem_hash: str,
    model_hash: Optional[str],
    results_hash: Optional[str],
    combinations: Optional[Sequence[Combination]],
    names: set[str],
) -> None:
    """Read the partial back through the reader before it may replace anything."""
    back = UserFile(partial)
    if (back.fem_hash, back.model_hash, back.results_hash) != (fem_hash, model_hash, results_hash):
        raise UserWriteRefused(f"{partial}: the hashes did not read back")
    if combinations is not None and (
        back.combinations(model_h5=model_h5).rows != tuple(combinations)
    ):
        raise UserWriteRefused(f"{partial}: the combinations did not read back")
    if names:
        index = back.results()
        missing = names - {e.name for e in index.entries}
        if missing:
            raise UserWriteRefused(f"{partial}: did not read back: {sorted(missing)}")


def _fsync(path: Path) -> None:
    with open(path, "r+b") as f:  # Windows refuses fsync on a read-only handle
        os.fsync(f.fileno())


def _open_partial(partial: Path) -> h5py.File:
    """Create the partial, retried. Windows releases a dead process's file locks
    some time after it exits, and how long depends on the load (``LockFileEx``),
    so a killed writer's lock can outlive it. A live writer's lock outlasts this."""
    for attempt in range(_OPEN_TRIES):
        try:
            return h5py.File(partial, "w")
        except OSError as exc:
            if attempt == _OPEN_TRIES - 1:
                raise UserWriteRefused(
                    f"{partial} is held by another writer; nothing was replaced: {exc}"
                ) from exc
            time.sleep(_REPLACE_WAIT_S)
    raise AssertionError("unreachable")


def _replace(partial: Path, target: Path) -> None:
    """``os.replace``, retried: on Windows a reader's handle blocks it for a moment."""
    for attempt in range(_REPLACE_TRIES):
        try:
            os.replace(partial, target)
            return
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(_REPLACE_WAIT_S)
