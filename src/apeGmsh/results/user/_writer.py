"""Write a ``<stem>.user.h5``: atomic, and never past a zone it does not know.

Every write builds its own ``<stem>.user.h5.<host>-<pid>-<random>.partial`` in the
same folder, created exclusively, then reopens and verifies it, and only then replaces
the file, so a killed writer leaves the previous file whole (ADR 0130 D2, A4).
The name is per writer because truncating a shared one destroys the other
writer's data: on Windows ``h5py.File(name, "w")`` truncates before it fails on
the lock. ``/meta@revision`` is compared immediately before the replace, so the
second of two writers refuses. A window remains between that comparison and the
replace, which only a lock on the target could close.

Rules this module holds:

* ``/meta`` hashes come from the model archive's ``/meta/lineage``, byte-equal.
* A zone it does not write is copied verbatim, whatever its version. A zone it
  does write refuses when the file's version is outside its range (A4).
* ``combinations`` adds a generation; earlier ones stay as the backup (A4).
* ``modes``, ``spectra`` and ``entities`` replace their whole zone; ``None`` leaves
  it as it is.
* An ``ON_ENTITIES`` result names a row the file's ``/entities`` holds, or the
  write refuses: checked whenever either side is written.
* A hash change never silently re-labels what was derived under the old hash.
  Those entries are rewritten in the same call, or ``drop_stale=True`` drops
  them; otherwise the write refuses.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import re
import secrets
import socket
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
    ENTITIES,
    ENTITIES_KEY,
    ENTITIES_SCHEMA_VERSION,
    MODES,
    MODES_KEY,
    MODES_SCHEMA_VERSION,
    SPECTRA,
    SPECTRA_KEY,
    SPECTRA_SCHEMA_VERSION,
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
from ._entities import Entities, entities_problem, write_entities
from ._errors import UserWriteRefused
from ._fields import EntryRefused, UserResult, write_entry
from ._fingerprint import results_fingerprint
from ._modes import ModeTable, table_problem, write_modes
from ._names import name_problem, text_or_empty
from ._reader import UserFile, read_model_combinations
from ._spectra import SpectrumCase, case_problem, same_bits, write_spectra

__all__ = ["model_hashes", "write_user_file"]

SUFFIX = ".user.h5"
#: The zones that go stale with the model hashes (A6), and their /meta keys.
_MODEL_ZONES = {
    "modes": MODES_KEY,
    "spectra": SPECTRA_KEY,
    "entities": ENTITIES_KEY,
}
_GENERATION = re.compile(r"g(\d+)\Z")
_ENTRY = re.compile(r"r(\d+)\Z")
_REPLACE_TRIES = 10
_REPLACE_WAIT_S = 0.1
#: A process's exit code while it runs, and ``OpenProcess``'s rights and refusal.
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_MAX_DWORD = 0xFFFFFFFF
#: The host token when the machine's name cannot be told; it never sweeps.
_UNKNOWN_HOST = "unknownhost"


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
    modes: Optional[ModeTable] = None,
    spectra: Optional[Sequence[SpectrumCase]] = None,
    entities: Optional[Entities] = None,
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
    ``modes`` (a :class:`~apeGmsh.results.user.ModeTable`) and ``spectra``
    (one :class:`~apeGmsh.results.user.SpectrumCase` per case) replace that
    zone whole, and ``None`` leaves it untouched; so does ``entities`` (an
    :class:`~apeGmsh.results.user.Entities`). A ``target="entities"`` result
    must name rows of the ``/entities`` the file holds after this write.
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
    _check_modes_and_spectra(modes, spectra)
    if entities is not None:
        if not isinstance(entities, Entities):
            raise UserWriteRefused(
                f"entities must be an Entities, got {type(entities).__name__}")
        problem = entities_problem(entities)
        if problem is not None:
            raise UserWriteRefused(f"entities refused: {problem}")
    rewritten = {root for root, given in (
        ("modes", modes), ("spectra", spectra), ("entities", entities)) if given is not None}

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
            rewritten,
        )
    dropped = _derived_to_drop(
        target, model_moved, results_moved, names, drop_stale, rewritten)

    _sweep_dead_partials(target)
    partial, out = _create_partial(target)
    try:
        with out:
            _write_all(out, previous, target, fem_hash, model_hash, results_hash,
                       combinations, results, modes, spectra, entities, units, writer,
                       dropped)
        _verify(partial, model_h5, fem_hash, model_hash, results_hash, combinations, results,
                modes, spectra, entities)
        _fsync(partial)
        _refuse_if_moved(target, previous)
        _replace(partial, target)
    except BaseException:
        with contextlib.suppress(OSError):  # the first failure is the one to report
            partial.unlink(missing_ok=True)
        raise
    return target


def _refuse_if_moved(target: Path, previous: Optional[UserFile]) -> None:
    """The second of two writers refuses here, rather than overwriting (A4)."""
    if previous is None:
        if target.exists():
            raise UserWriteRefused(
                f"{target} was created while this write ran; nothing was replaced. "
                "Write again.")
    elif UserFile(target).revision != previous.revision:
        raise UserWriteRefused(
            f"{target} changed while this write ran (revision "
            f"{previous.revision} moved); nothing was replaced. Write again.")


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


def _check_modes_and_spectra(
    modes: Optional[ModeTable], spectra: Optional[Sequence[SpectrumCase]],
) -> None:
    if modes is not None:
        problem = table_problem(modes)
        if problem is not None:
            raise UserWriteRefused(f"modes table refused: {problem}")
    if spectra is not None:
        seen: set[str] = set()
        for case in spectra:
            problem = case_problem(case)
            if problem is not None:
                raise UserWriteRefused(f"spectrum case {case.name!r} refused: {problem}")
            if case.name in seen:
                raise UserWriteRefused(f"spectrum case {case.name!r} is given twice")
            seen.add(case.name)


def _check_known_versions(
    target: Path, combinations: bool, results: bool, rewritten: "set[str] | None" = None,
) -> None:
    """No writer touches a zone newer than it knows (A4), or one with no version:
    a group with no key cannot be judged, and copying it on would hide that."""
    zones = rewritten or set()
    with h5py.File(target, "r") as f:
        attrs = f["meta"].attrs
        for root, key in (("combinations", COMBINATIONS_KEY), ("results", USER_RESULTS_KEY),
                          ("modes", MODES_KEY), ("spectra", SPECTRA_KEY),
                          ("entities", ENTITIES_KEY)):
            if root in f and key not in attrs:
                raise UserWriteRefused(
                    f"{target}: /{root} has no {key} in /meta; nothing was replaced.")
        for wanted, zone in ((combinations, COMBINATIONS), (results, USER_RESULTS),
                             ("modes" in zones, MODES), ("spectra" in zones, SPECTRA),
                             ("entities" in zones, ENTITIES)):
            version = read_zone_version(attrs, zone) if wanted else None
            if version is not None:
                validate_zone_version(version, reader_version(zone), zone=zone)


def _derived_to_drop(
    target: Path,
    model_moved: bool,
    results_moved: bool,
    rewritten: set[str],
    drop_stale: bool,
    zones: "set[str] | None" = None,
) -> set[str]:
    """Links (``results/r3``, ``modes``) a hash change leaves behind.

    ``rewritten`` names the user results this call writes again, ``zones`` the
    model zones it replaces (``modes``, ``spectra``, ``entities``): neither is left
    behind. ``/entities`` leaves with every ``ON_ENTITIES`` result, whatever its
    basis: a result whose rows are gone cannot stand.

    Refuses unless ``drop_stale``: keeping them under the new hash would draw
    old values as current.
    """
    if not (model_moved or results_moved):
        return set()
    losing: set[str] = set()
    with h5py.File(target, "r") as f:
        if model_moved:
            losing |= {root for root in _MODEL_ZONES
                       if root in f and root not in (zones or set())}
        entities_leave = "entities" in losing
        if "results" in f and isinstance(f["results"], h5py.Group):
            for link in f["results"]:
                group = f["results"][link]
                if not _ENTRY.match(link) or not isinstance(group, h5py.Group):
                    continue
                basis = text_or_empty(group.attrs["basis"]) if "basis" in group.attrs else ""
                name = text_or_empty(group.attrs["name"]) if "name" in group.attrs else ""
                bound = "target" in group.attrs and (
                    text_or_empty(group.attrs["target"]) == "entities")
                if name in rewritten:
                    continue
                if ((basis == "results" and results_moved) or (basis == "model" and model_moved)
                        or (bound and entities_leave)):
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
    modes: Optional[ModeTable],
    spectra: Optional[Sequence[SpectrumCase]],
    entities: Optional[Entities],
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
    if modes is not None:
        if "modes" in out:
            del out["modes"]
        write_modes(out, modes)
        meta.attrs[MODES_KEY] = MODES_SCHEMA_VERSION
    if spectra is not None:
        if "spectra" in out:
            del out["spectra"]
        try:
            write_spectra(out, spectra)
        except EntryRefused as exc:
            raise UserWriteRefused(f"spectra refused: {exc}") from exc
        meta.attrs[SPECTRA_KEY] = SPECTRA_SCHEMA_VERSION
    if entities is not None:
        if "entities" in out:
            del out["entities"]
        write_entities(out, entities)
        meta.attrs[ENTITIES_KEY] = ENTITIES_SCHEMA_VERSION


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
    results: Sequence[UserResult],
    modes: Optional[ModeTable] = None,
    spectra: Optional[Sequence[SpectrumCase]] = None,
    entities: Optional[Entities] = None,
) -> None:
    """Read the partial back through the reader before it may replace anything,
    and compare the ``DATA`` this call wrote bit for bit: a partial another process
    truncated can still read as a well-formed file holding zeros."""
    back = UserFile(partial)
    if (back.fem_hash, back.model_hash, back.results_hash) != (fem_hash, model_hash, results_hash):
        raise UserWriteRefused(f"{partial}: the hashes did not read back")
    if combinations is not None and (
        back.combinations(model_h5=model_h5).rows != tuple(combinations)
    ):
        raise UserWriteRefused(f"{partial}: the combinations did not read back")
    if entities is not None or any(r.target == "entities" for r in results):
        _check_bindings(back)
    if results:
        index = back.results()
        missing = {r.name for r in results} - {e.name for e in index.entries}
        if missing:
            raise UserWriteRefused(f"{partial}: did not read back: {sorted(missing)}")
        for r in results:
            if not _same_data(r, back.read_result(r.name)):
                raise UserWriteRefused(f"{partial}: the DATA of {r.name!r} did not read back")
    if modes is not None and not modes.same_bits(back.modes()):
        raise UserWriteRefused(f"{partial}: the modes table did not read back")
    if spectra is not None:
        read = back.spectra()
        if read.refused or len(read.cases) != len(spectra) or not all(
                same_bits(a, b) for a, b in zip(spectra, read.cases)):
            raise UserWriteRefused(f"{partial}: the spectrum cases did not read back")
    if entities is not None and not entities.same_bits(back.entities()):
        raise UserWriteRefused(f"{partial}: the entities did not read back")


def _check_bindings(back: UserFile) -> None:
    """Every ``ON_ENTITIES`` result names a row the file's ``/entities`` holds.

    Run on the partial, so it judges what the file holds after this write: a
    result written now, and one an ``entities`` rewrite would leave dangling.
    """
    for ref in back.results().refused:
        if ref.reason == "unresolved":
            raise UserWriteRefused(f"{back.path}: {ref.detail}; nothing was replaced.")


def _bits(values: object) -> bytes:
    return np.asarray(values, dtype="f8").tobytes()


def _same_data(wrote: UserResult, read: UserResult) -> bool:
    if wrote.target != "elements":
        return _bits(wrote.data) == _bits(read.data)
    theirs = {b.name: b for b in read.buckets}
    return all(
        b.name in theirs and _bits(b.data) == _bits(theirs[b.name].data)
        for b in wrote.buckets
    )


def _fsync(path: Path) -> None:
    with open(path, "r+b") as f:  # Windows refuses fsync on a read-only handle
        os.fsync(f.fileno())


def _create_partial(target: Path) -> tuple[Path, h5py.File]:
    """This writer's own partial, created exclusively: no other writer's is touched."""
    name = f"{target.name}.{_host_token()}-{os.getpid()}-{secrets.token_hex(4)}.partial"
    partial = target.with_name(name)
    try:
        return partial, h5py.File(partial, "x")
    except OSError as exc:
        raise UserWriteRefused(f"{partial} cannot be created; nothing was replaced: {exc}") from exc


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` runs. Unsure reads as alive, so a sweep leaves the file.

    Never ``os.kill(pid, 0)`` on Windows: there any signal value but the two
    console events terminates the process.
    """
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:  # EPERM: it runs under another user
            return True
        return True
    if pid > _MAX_DWORD:
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not handle:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _host_token() -> str:
    """This machine's name, safe in a file name and free of the ``-`` separator.

    :data:`_UNKNOWN_HOST` when it cannot be told: a partial then still gets a
    name, but no sweep runs, because no host could prove the partial is its own.
    """
    try:
        host = socket.gethostname()
    except OSError:
        return _UNKNOWN_HOST
    token = re.sub(r"[^a-z0-9]+", "_", host.lower()).strip("_")[:63]
    return token or _UNKNOWN_HOST


def _sweep_dead_partials(target: Path) -> None:
    """Remove this host's partials whose writer has died.

    A set lives in a synced folder, so another machine's live writer has a PID
    this host cannot see, and a deletion here would sync to it: only partials
    naming this host are ever removed. Failure here never fails the write: a
    dead process's lock outlives it for a moment, and the next write tries again.
    """
    host = _host_token()
    if host == _UNKNOWN_HOST:
        return
    pattern = re.compile(
        re.escape(target.name) + r"\.([a-z0-9_]+)-(\d+)-[0-9a-f]+\.partial\Z")
    with contextlib.suppress(OSError):
        for entry in target.parent.iterdir():
            found = pattern.match(entry.name)
            if found is None or found.group(1) != host:
                continue
            if not _pid_alive(int(found.group(2))):
                with contextlib.suppress(OSError):
                    entry.unlink()


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
