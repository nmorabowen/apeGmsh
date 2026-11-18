"""Open a ``.ladruno`` by the fork's reader rule, steps 1-3 (M2-S7 #2991).

The rule is the fork's contract ``Ladruno_implementation/232a_mpm_contracts.md``
§2.5 ("The reader rule: opening a particle file"), merged on the fork's
``ladruno`` branch at ``a1d4c7260``; its reference is
``ladruno_format.open_ladruno`` / ``repair_copy``. The writer runs HDF5 SWMR
from the end of each stage's layout to close, so a killed run leaves one of
three files:

1. **Plain read-only open.** A cleanly closed file (superblock consistency
   flags 0x00) opens this way, and so can a file killed inside a layout,
   because sec2 writes the flags lazily.
2. **SWMR-read open** (flags 0x05, WRITE_ACCESS | SWMR_WRITE): the plain open
   refuses with "file is already open for write" or "truncated file". Never
   ``h5clear``: the stored end-of-allocation lags the data, and only an
   SWMR-read open skips that check.
3. **Repair a copy** (flags 0x01, WRITE_ACCESS only: killed while a layout was
   written, which neither open accepts). The copy, never the user's file,
   gets flags 0x00, an end-of-allocation of max(stored EOA, file size) with
   zero padding, and a new lookup3 checksum over superblock bytes 0-43 at
   byte 44; the copy then opens plainly.

A lock error ("unable to lock file", or a raw read the OS refuses) means a
live writer holds the file: :class:`LadrunoFileBusyError`, and nothing is
copied or repaired. The recorder turns HDF5's own locking off and holds a
guard lock instead (schema §7b.2, "A file a live run is writing"), so a
locking reader meets that error for the whole life of the run.

This module does the open and reports which step opened the file
(:attr:`OpenedLadruno.step`). What a step means for the file's stages is the
caller's rule: the particle reader's is in ``_ladruno_particles.py``; the
recorder file's is :func:`stage_report` here (M2-S8 #3003), schema §7b.2 at
the fork's ``ladruno-v20261116`` (``6ee968060``), whose reference is
``ladruno_format.stage_report`` / ``committed_steps``.
"""
from __future__ import annotations

import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    import h5py


#: The steps of the reader rule, as :attr:`OpenedLadruno.step` reports them.
STEP_PLAIN = 1
STEP_SWMR = 2
STEP_REPAIRED = 3

#: Superblock consistency flags (HDF5 version-2/3 superblock, byte 11).
FLAGS_WRITE_ACCESS = 0x01
FLAGS_SWMR_KILLED = 0x05

#: A version-2/3 superblock with 8-byte offsets: the signature, the version at
#: byte 8, the offset size at byte 9, the flags at byte 11, the end-of-file
#: address at byte 28, and lookup3 over bytes 0..43 stored at byte 44.
_SB_SIG = b"\x89HDF\r\n\x1a\n"
_SB_LEN = 48
_SB_FLAGS = 11
_SB_EOA = slice(28, 36)
_SB_SUM_AT = 44


class LadrunoFileBusyError(OSError):
    """A live writer holds the ``.ladruno`` (a lock error, "unable to lock
    file"). The file was not copied or repaired; open it again later."""


#: How to follow a run that is still writing (schema §7b.2, "To read a live
#: file"): the SWMR tail read, with HDF5 file locking off, which never blocks
#: the writer. Named in every :class:`LadrunoFileBusyError`.
SWMR_TAIL_READ = 'h5py.File(path, "r", swmr=True, locking=False)'


class LadrunoStageCorrupt(ValueError):
    """A ``MODEL_STAGE`` does not open or read where the stage rule of schema
    §7b.2 allows none to fail: an earlier stage than the last, any stage of a
    file opened SWMR-read (killed with SWMR armed, after its last layout), or
    any stage of a file whose superblock the SWMR writer did not write."""


class LadrunoStageSkippedWarning(UserWarning):
    """The last stage of a killed run was torn while it was laid out (schema
    §7b.2): it does not open or read, holds no committed step, and is not
    listed."""


# ---------------------------------------------------------------------
# The superblock
# ---------------------------------------------------------------------

def lookup3(data: bytes, initval: int = 0) -> int:
    """HDF5's metadata checksum, ``H5_checksum_lookup3`` (Bob Jenkins' hashlittle)."""
    m32 = 0xFFFFFFFF

    def rot(x: int, k: int) -> int:
        return ((x << k) | (x >> (32 - k))) & m32

    n = len(data)
    a = b = c = (0xDEADBEEF + n + initval) & m32
    k = 0
    while n > 12:
        a = (a + int.from_bytes(data[k:k + 4], "little")) & m32
        b = (b + int.from_bytes(data[k + 4:k + 8], "little")) & m32
        c = (c + int.from_bytes(data[k + 8:k + 12], "little")) & m32
        a = (a - c) & m32
        a ^= rot(c, 4)
        c = (c + b) & m32
        b = (b - a) & m32
        b ^= rot(a, 6)
        a = (a + c) & m32
        c = (c - b) & m32
        c ^= rot(b, 8)
        b = (b + a) & m32
        a = (a - c) & m32
        a ^= rot(c, 16)
        c = (c + b) & m32
        b = (b - a) & m32
        b ^= rot(a, 19)
        a = (a + c) & m32
        c = (c - b) & m32
        c ^= rot(b, 4)
        b = (b + a) & m32
        n -= 12
        k += 12
    if n == 0:
        return c
    tail = bytes(data[k:k + n]) + b"\0" * (12 - n)
    a = (a + int.from_bytes(tail[0:4], "little")) & m32
    b = (b + int.from_bytes(tail[4:8], "little")) & m32
    c = (c + int.from_bytes(tail[8:12], "little")) & m32
    c ^= b
    c = (c - rot(b, 14)) & m32
    a ^= c
    a = (a - rot(c, 11)) & m32
    b ^= a
    b = (b - rot(a, 25)) & m32
    c ^= b
    c = (c - rot(b, 16)) & m32
    a ^= c
    a = (a - rot(c, 4)) & m32
    b ^= a
    b = (b - rot(a, 14)) & m32
    c ^= b
    c = (c - rot(b, 24)) & m32
    return c


def _is_v23(sb: bytes) -> bool:
    return (len(sb) >= _SB_LEN and sb[:8] == _SB_SIG and sb[8] in (2, 3)
            and sb[9] == 8)


def _read_superblock(path: Path) -> bytes:
    """The first 48 bytes; a refused read (a live writer's lock) is busy."""
    try:
        with open(path, "rb") as fh:
            return fh.read(_SB_LEN)
    except PermissionError as e:
        raise _busy_error(path, e) from e


def superblock_flags(path: "str | Path") -> Optional[int]:
    """The consistency flags of a version-2/3 superblock (0x00 after a clean
    close), or None for any other superblock (a version-0 file has none)."""
    sb = _read_superblock(Path(path))
    return sb[_SB_FLAGS] if _is_v23(sb) else None


def repair_copy(src: "str | Path", dst: "str | Path") -> None:
    """Reader rule step 3 on a COPY; ``src`` is never written.

    Copy ``src`` to ``dst``; set the consistency flags to 0x00; set the stored
    end-of-allocation to max(stored EOA, file size), padding the copy with
    zeros up to it; re-checksum the superblock. Refuses a file whose
    superblock is not version 2/3 with 8-byte offsets, or whose stored
    checksum does not match (it is not the superblock the rule describes).
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("repair_copy repairs a copy: dst must differ from src")
    sb = _read_superblock(src)
    if not _is_v23(sb):
        raise ValueError(
            f"{src}: not a version-2/3 HDF5 superblock with 8-byte offsets, "
            "so the reader rule's repair does not apply")
    if lookup3(sb[:_SB_SUM_AT]) != int.from_bytes(sb[_SB_SUM_AT:_SB_LEN], "little"):
        raise ValueError(f"{src}: the superblock checksum does not match")
    shutil.copyfile(src, dst)
    size = dst.stat().st_size
    eoa = max(int.from_bytes(sb[_SB_EOA], "little"), size)
    fixed = bytearray(sb)
    fixed[_SB_FLAGS] = 0x00
    fixed[_SB_EOA] = eoa.to_bytes(8, "little")
    fixed[_SB_SUM_AT:_SB_LEN] = lookup3(bytes(fixed[:_SB_SUM_AT])).to_bytes(4, "little")
    with open(dst, "r+b") as fh:
        fh.write(fixed)
        if size < eoa:
            fh.seek(size)
            fh.write(b"\0" * (eoa - size))


# ---------------------------------------------------------------------
# The open
# ---------------------------------------------------------------------

@dataclass
class OpenedLadruno:
    """A ``.ladruno`` opened by the reader rule.

    ``step`` is the step that opened it (``STEP_PLAIN``, ``STEP_SWMR`` or
    ``STEP_REPAIRED``); ``flags`` the user's file's superblock flags at the
    open (None for a version-0/1 superblock); ``copy`` the repaired copy that
    step 3 opened, removed by :meth:`close`.
    """

    h5: "h5py.File"
    path: Path
    step: int
    flags: Optional[int]
    copy: Optional[Path] = None
    _scratch: Optional[Path] = field(default=None, repr=False)

    def close(self) -> None:
        try:
            self.h5.close()
        finally:
            if self._scratch is not None:
                shutil.rmtree(self._scratch, ignore_errors=True)
                self._scratch = None

    def __enter__(self) -> "OpenedLadruno":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _busy(e: BaseException) -> bool:
    return isinstance(e, PermissionError) or "unable to lock file" in str(e)


def _busy_error(path: Path, e: BaseException) -> LadrunoFileBusyError:
    return LadrunoFileBusyError(
        f"{path}: a live writer holds the file ({str(e).splitlines()[0]}); "
        "open it again when the run has stopped. To follow the run while it "
        f"writes, use the SWMR tail read, {SWMR_TAIL_READ}, and re-open it to "
        "refresh (fork schema section 7b.2). Nothing was copied or repaired.")


def open_ladruno(path: "str | Path", *,
                 scratch_dir: "Optional[str | Path]" = None) -> OpenedLadruno:
    """Open ``path`` read-only by the reader rule (steps 1-3 above).

    Step 3's copy goes in a new directory under ``scratch_dir`` (default: the
    system temp directory) and is removed when the result closes. Raises
    :class:`LadrunoFileBusyError` on a lock error; any other refusal re-raises
    the plain open's error.
    """
    import h5py

    path = Path(path)
    flags = superblock_flags(path)   # a raw read a live writer's lock refuses is busy
    try:
        return OpenedLadruno(h5py.File(path, "r"), path, STEP_PLAIN, flags)
    except OSError as e:
        if _busy(e):
            raise _busy_error(path, e) from e
        if "already open for write" not in str(e) and "truncated file" not in str(e):
            raise
        plain = e
    try:
        return OpenedLadruno(h5py.File(path, "r", swmr=True), path, STEP_SWMR, flags)
    except OSError as e:
        if _busy(e):
            raise _busy_error(path, e) from e
        if "not already open for SWMR writing" not in str(e) or flags != FLAGS_WRITE_ACCESS:
            raise plain from e
    scratch = Path(tempfile.mkdtemp(prefix="apegmsh-ladruno-repair-",
                                    dir=None if scratch_dir is None else str(scratch_dir)))
    try:
        copy = scratch / path.name
        repair_copy(path, copy)
        h5 = h5py.File(copy, "r")
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return OpenedLadruno(h5, path, STEP_REPAIRED, flags, copy, scratch)


# ---------------------------------------------------------------------
# The recorder file's stage rule and crash rule (schema §7b.2)
# ---------------------------------------------------------------------

_STAGE_PREFIX = "MODEL_STAGE["
#: The ``RESULTS`` families whose result groups stream (schema §7.0).
_STREAMING_FAMILIES = ("ON_NODES", "ON_ELEMENTS", "ON_DOMAIN", "ON_REGIONS")


def stage_names(h5: "h5py.File") -> "list[str]":
    """The ``MODEL_STAGE[<k>]`` groups of ``h5``, by their numeric id."""
    names = [k for k in h5 if k.startswith(_STAGE_PREFIX)]
    return sorted(names, key=lambda n: (int(n[len(_STAGE_PREFIX):].rstrip("]")), n))


def streaming_results(stage: "h5py.Group") -> "Iterator[h5py.Group]":
    """Every streaming result group of a stage: a group holding a ``DATA``
    dataset beside ``TIME`` and ``STEP``, under ``RESULTS/ON_NODES``,
    ``ON_ELEMENTS`` (``<display>/<bucket>``), ``ON_DOMAIN`` or ``ON_REGIONS``.
    A modes result (its ``DATA`` is a group) is not one. The fork's
    ``ladruno_format._streaming_results``."""
    import h5py

    if "RESULTS" not in stage:
        return
    res = stage["RESULTS"]
    for fam in _STREAMING_FAMILIES:
        if fam not in res:
            continue
        stack = [res[fam]]
        while stack:
            g = stack.pop()
            for name in g:
                item = g[name]
                if not isinstance(item, h5py.Group):
                    continue
                cls = item.get("DATA", getclass=True)
                if cls is h5py.Dataset and "STEP" in item and "TIME" in item:
                    yield item
                elif cls is None:
                    stack.append(item)


def committed_steps(stage: "h5py.Group") -> Optional[int]:
    """T' of a stage (schema §7b.2, the crash rule): the minimum over its
    streaming results of min(len ``DATA``, len ``TIME``, len ``STEP``); None
    for a stage with no streaming result. The fork's
    ``ladruno_format.committed_steps``. It applies to a KILLED file only
    (superblock flags other than 0x00); a cleanly closed file is read as it
    is."""
    n: Optional[int] = None
    for g in streaming_results(stage):
        m = min(g["DATA"].shape[0], g["TIME"].shape[0], g["STEP"].shape[0])
        n = m if n is None else min(n, m)
    return n


@dataclass(frozen=True)
class LadrunoStages:
    """What the stage rule made of a recorder file's ``MODEL_STAGE`` groups.

    ``names`` are the stages that read, in id order; ``committed`` maps each
    to the rows a reader may use: T' (:func:`committed_steps`) on a killed
    file, None (no cut) on a cleanly closed one or a stage with no streaming
    result. ``skipped`` names the torn last stage, if any, with the reason.
    """

    names: "tuple[str, ...]"
    committed: "dict[str, Optional[int]]"
    killed: bool
    skipped: "tuple[tuple[str, str], ...]" = ()


def stage_report(opened: OpenedLadruno) -> LadrunoStages:
    """The stage rule of schema §7b.2 for a recorder file opened by
    :func:`open_ladruno` (the fork's ``ladruno_format.stage_report``).

    Each stage is read as the reference reads it: its group opens and every
    streaming result's ``DATA``/``TIME``/``STEP`` lengths read
    (:func:`committed_steps`). A stage that fails was torn while it was laid
    out only if it is the LAST stage and the file was opened plainly (step 1)
    or through a repaired copy (step 3, flags 0x01): it is skipped, with a
    :class:`LadrunoStageSkippedWarning`. Everywhere else a stage that fails is
    :class:`LadrunoStageCorrupt`: an earlier stage; any stage of a file opened
    SWMR-read (step 2, flags 0x05: killed with SWMR armed, after its last
    layout); and any stage of a file whose superblock is not version 2/3, which
    the SWMR writer never wrote.

    ``killed`` is True when the user's file's superblock flags differ from
    0x00; only then are the stages cut to T'.
    """
    h5, path = opened.h5, opened.path
    killed = opened.flags not in (None, 0x00)
    names = stage_names(h5)
    readable: "list[str]" = []
    committed: "dict[str, Optional[int]]" = {}
    skipped: "list[tuple[str, str]]" = []
    for i, name in enumerate(names):
        try:
            n = committed_steps(h5[name])
        except Exception as e:  # noqa: BLE001 - any failure to read is the rule's input
            why = (str(e).splitlines() or [type(e).__name__])[0]
            if opened.step == STEP_SWMR:
                raise LadrunoStageCorrupt(
                    f"{path.name} {name} does not read in a file killed with "
                    "SWMR armed (superblock flags 0x05, opened SWMR-read), "
                    "where every stage must read: corruption (fork schema "
                    f"section 7b.2): {why}") from e
            if i < len(names) - 1:
                raise LadrunoStageCorrupt(
                    f"{path.name} {name} does not read and is not the file's "
                    f"last stage, so it is corruption (fork schema section "
                    f"7b.2): {why}") from e
            if opened.flags is None:
                raise LadrunoStageCorrupt(
                    f"{path.name} {name} does not read, and the file's "
                    "superblock is not one the SWMR recorder writes, so the "
                    f"torn-last-stage rule does not apply: {why}") from e
            reason = (
                f"{path.name} {name} was torn while it was laid out (the last "
                "stage of a killed run) and holds no committed step; it is "
                f"skipped (fork schema section 7b.2): {why}")
            warnings.warn(reason, LadrunoStageSkippedWarning, stacklevel=3)
            skipped.append((name, reason))
            continue
        readable.append(name)
        committed[name] = n if killed else None
    return LadrunoStages(tuple(readable), committed, killed, tuple(skipped))
