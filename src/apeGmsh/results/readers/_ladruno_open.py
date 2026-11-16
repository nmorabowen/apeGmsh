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
copied or repaired.

This module does the open only and reports which step opened the file
(:attr:`OpenedLadruno.step`). What a step means for the file's stages is the
caller's rule: the particle reader's is in ``_ladruno_particles.py``.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

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
        raise LadrunoFileBusyError(
            f"{path}: the file cannot be read (a live writer holds it); "
            "open it again later") from e


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
        "open it again when the run has stopped. Nothing was copied or "
        "repaired.")


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
