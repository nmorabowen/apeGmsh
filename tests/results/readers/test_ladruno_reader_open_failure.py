"""A ``LadrunoReader`` whose construction fails closes its file (#2113).

``LadrunoReader.__init__`` opens the HDF5 file, then checks that it is a
``.ladruno`` (``INFO/GENERATOR`` and ``FORMAT_VERSION``) and sets up its
caches. When that check or that setup fails, or is interrupted, the reader:

* re-raises the original error;
* closes the file first, so it stays unlocked while the traceback lives
  (a REPL or notebook keeps ``sys.last_value``, whose frames hold the
  half-built reader);
* raises nothing from ``__del__`` once the half-built reader is collected.

The oracles are the h5py file lock and ``sys.unraisablehook``:

* h5py refuses a ``"r+"`` open of a file this process still has open, and
  Windows also refuses to delete it, so a clean reopen and delete, done
  while the error is still held and with no ``gc.collect()`` before it,
  proves the handle was closed;
* an exception raised in ``__del__`` goes to ``sys.unraisablehook``, which
  these tests capture.

On origin/main 30bd01c5, every failure case below fails its reopen: the
file was opened before ``_validate_identity`` and nothing closed it on the
way out.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers import _ladruno
from apeGmsh.results.readers._ladruno import LadrunoReader
from tests.results.readers.test_ladruno_wp165 import _write_part

pytestmark = pytest.mark.filterwarnings("error")


def _assert_released(path: Path) -> None:
    """``path`` is closed in this process: it reopens for writing and deletes."""
    h5py.File(path, "r+").close()
    path.unlink()


def _assert_locked(path: Path) -> None:
    with pytest.raises(OSError):
        h5py.File(path, "r+")


@pytest.fixture
def unraisable(monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Collect every exception that reaches ``sys.unraisablehook``."""
    seen: list[BaseException] = []
    monkeypatch.setattr(
        sys, "unraisablehook", lambda u: seen.append(u.exc_value),
    )
    return seen


def _valid(tmp_path: Path) -> Path:
    return _write_part(tmp_path / "run.ladruno", [1, 2])


def _no_info(f: h5py.File) -> None:
    f.create_group("MODEL_STAGE[1]")


def _info_without_generator(f: h5py.File) -> None:
    f.create_group("INFO").attrs["FORMAT_VERSION"] = 1


def _foreign_generator(f: h5py.File) -> None:
    info = f.create_group("INFO")
    info.attrs["GENERATOR"] = np.bytes_(b"STKO")
    info.attrs["FORMAT_VERSION"] = 1


def _unsupported_version(f: h5py.File) -> None:
    info = f.create_group("INFO")
    info.attrs["GENERATOR"] = np.bytes_(b"Ladruno")
    info.attrs["FORMAT_VERSION"] = 999


_NOT_LADRUNO: list[tuple[Callable[[h5py.File], None], str]] = [
    (_info_without_generator, "missing INFO/GENERATOR"),
    (_no_info, "missing INFO/GENERATOR"),
    (_foreign_generator, "INFO/GENERATOR is 'STKO'"),
    (_unsupported_version, "INFO/FORMAT_VERSION=999 is not supported"),
]


def test_oracle_an_open_reader_holds_the_lock(tmp_path: Path) -> None:
    """Positive control: the reopen check bites while a reader is open."""
    path = _valid(tmp_path)
    reader = LadrunoReader(path)
    _assert_locked(path)
    reader.close()
    _assert_released(path)


@pytest.mark.parametrize(
    "write, message", _NOT_LADRUNO,
    ids=["info_without_generator", "no_info", "foreign_generator",
         "unsupported_version"],
)
def test_identity_failure_surfaces_and_releases_the_file(
    tmp_path: Path, write, message, unraisable,
) -> None:
    path = tmp_path / "not_ladruno.h5"
    with h5py.File(path, "w") as f:
        write(f)

    with pytest.raises(ValueError, match=message) as info:
        LadrunoReader(path)

    # The identity error itself, with nothing chained on top by the cleanup.
    assert info.type is ValueError
    assert info.value.__context__ is None
    # Closed while the error and its traceback are still held.
    _assert_released(path)

    del info
    gc.collect()
    assert unraisable == []


def test_interrupted_validation_releases_the_file(
    tmp_path: Path, unraisable,
) -> None:
    """Ctrl-C during the identity check, on an otherwise valid file."""
    path = _valid(tmp_path)

    def interrupt(self) -> None:
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LadrunoReader, "_validate_identity", interrupt)
        with pytest.raises(KeyboardInterrupt) as info:
            LadrunoReader(path)

    assert info.type is KeyboardInterrupt
    _assert_released(path)
    del info
    gc.collect()
    assert unraisable == []


def test_failed_setup_after_validation_releases_the_file(
    tmp_path: Path, unraisable,
) -> None:
    """A valid file whose post-validation setup raises."""
    path = _valid(tmp_path)

    def refuse(name: str) -> None:
        raise RuntimeError("setup failed (test)")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_ladruno._mnodal, "RotationMask", refuse)
        with pytest.raises(RuntimeError, match=r"setup failed \(test\)") as info:
            LadrunoReader(path)

    assert info.value.__context__ is None
    _assert_released(path)
    del info
    gc.collect()
    assert unraisable == []


def test_unopenable_file_raises_cleanly(tmp_path: Path, unraisable) -> None:
    """The open itself fails: ``__del__`` on the reader with no handle is quiet."""
    path = tmp_path / "corrupt.ladruno"
    path.write_bytes(b"this is not an HDF5 file" * 8)

    with pytest.raises(OSError, match="file signature not found") as info:
        LadrunoReader(path)

    assert info.type is OSError
    del info
    gc.collect()
    assert unraisable == []
