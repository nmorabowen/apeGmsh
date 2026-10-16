"""A multi-partition reader whose construction fails releases its parts (#2106).

``LadrunoMultiPartitionReader`` and ``MPCOMultiPartitionReader`` open one
child reader per part file, then read every part to check that they agree.
When a part fails to open, or that check fails or is interrupted, the facade:

* re-raises the original error;
* closes every part it already opened, so the files stay unlocked while
  the traceback lives (a REPL or notebook keeps ``sys.last_value``);
* raises nothing from ``__del__`` once the half-built facade is collected.

The oracles are the h5py file lock and ``sys.unraisablehook``:

* h5py refuses a ``"r+"`` open of a file this process still has open, and
  Windows also refuses to delete it, so a clean reopen and delete proves the
  handle was closed;
* an exception raised in ``__del__`` goes to ``sys.unraisablehook``, which
  these tests capture.

On origin/main 4a1a341c, the ``__del__`` and interrupted-validation cases
fail: the facade's ``__del__`` raised ``AttributeError`` (``_readers`` was
bound only after every part opened), and the cleanup caught ``Exception``,
so a ``KeyboardInterrupt`` left every part open.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers._ladruno_multi import LadrunoMultiPartitionReader
from apeGmsh.results.readers._mpco_multi import MPCOMultiPartitionReader
from tests.results.readers.test_ladruno_wp165 import _write_part
from tests.test_results_mpco_multi_mock import _create_partition

pytestmark = pytest.mark.filterwarnings("error")

_N_PARTS = 3


def _ladruno_parts(tmp_path: Path) -> list[Path]:
    return [
        _write_part(
            tmp_path / f"run.part-{k}.ladruno", [3 * k + 1, 3 * k + 2],
            part=k, num_parts=_N_PARTS,
        )
        for k in range(_N_PARTS)
    ]


def _mpco_parts(tmp_path: Path) -> list[Path]:
    out = []
    for k in range(_N_PARTS):
        path = tmp_path / f"Recorder.part-{k}.mpco"
        ids = np.array([4 * k + 1, 4 * k + 2, 4 * k + 3, 4 * k + 4])
        _create_partition(
            path,
            node_ids=ids,
            coords=np.eye(4, 3) + k,
            element_ids=np.array([10 + k]),
            element_connectivity=ids.reshape(1, 4),
            n_steps=2, dt=0.1,
            disp_value_fn=lambda s, nid: [s * 100.0 + nid, 0.0, 0.0],
            stress_value_fn=lambda s, eid: [s + eid + 0.001 * a for a in range(6)],
        )
        out.append(path)
    return out


_FACADES: list[tuple[type, Callable[[Path], list[Path]]]] = [
    (LadrunoMultiPartitionReader, _ladruno_parts),
    (MPCOMultiPartitionReader, _mpco_parts),
]
_IDS = ["ladruno", "mpco"]


def _assert_released(path: Path) -> None:
    """``path`` is closed in this process: it reopens for writing and deletes."""
    h5py.File(path, "r+").close()
    path.unlink()


def _raised_while_opening(exc: BaseException) -> list[object]:
    """The ``path`` argument of every frame ``exc`` passed through.

    The caller's own frame is skipped: reading ``f_locals`` on a running
    frame snapshots its locals into a dict that outlives ``del``, which
    would keep the facade alive past the test's ``gc.collect()``.
    """
    caller = sys._getframe(1)
    out = []
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame is not caller:
            out.append(tb.tb_frame.f_locals.get("path"))
        tb = tb.tb_next
    return out


@pytest.fixture
def unraisable(monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Collect every exception that reaches ``sys.unraisablehook``."""
    seen: list[BaseException] = []
    monkeypatch.setattr(
        sys, "unraisablehook", lambda u: seen.append(u.exc_value),
    )
    return seen


@pytest.mark.parametrize("facade, make_parts", _FACADES, ids=_IDS)
def test_corrupt_middle_part_surfaces_and_releases_part_0(
    tmp_path: Path, facade, make_parts, unraisable,
) -> None:
    parts = make_parts(tmp_path)
    parts[1].write_bytes(b"this is a corrupt part file" * 8)

    with pytest.raises(OSError, match="file signature not found") as info:
        facade(parts)

    # The original open error, raised by the child reader of the corrupt
    # part, with nothing chained on top of it by the cleanup.
    assert info.type is OSError
    assert info.value.__context__ is None
    assert parts[1] in _raised_while_opening(info.value)
    # Part 0 is closed while the error (and its traceback) is still held.
    _assert_released(parts[0])

    del info
    gc.collect()
    assert unraisable == []


@pytest.mark.parametrize("facade, make_parts", _FACADES, ids=_IDS)
def test_failed_validation_releases_every_part(
    tmp_path: Path, facade, make_parts, unraisable,
) -> None:
    parts = make_parts(tmp_path)

    def refuse(self) -> None:
        raise ValueError("parts disagree (test)")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(facade, "_validate_consistency", refuse)
        with pytest.raises(ValueError, match=r"parts disagree \(test\)") as info:
            facade(parts)

    for p in parts:
        _assert_released(p)
    del info
    gc.collect()
    assert unraisable == []


@pytest.mark.parametrize("facade, make_parts", _FACADES, ids=_IDS)
def test_interrupted_validation_releases_every_part(
    tmp_path: Path, facade, make_parts, unraisable,
) -> None:
    """Ctrl-C during the consistency read, which reads every part."""
    parts = make_parts(tmp_path)

    def interrupt(self) -> None:
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(facade, "_validate_consistency", interrupt)
        with pytest.raises(KeyboardInterrupt) as info:
            facade(parts)

    # The traceback holds the half-built facade; its parts are closed anyway.
    assert info.value is not None
    for p in parts:
        _assert_released(p)
    del info
    gc.collect()
    assert unraisable == []


@pytest.mark.parametrize("facade", [f for f, _ in _FACADES], ids=_IDS)
def test_empty_path_list_raises_cleanly(facade, unraisable) -> None:
    with pytest.raises(ValueError, match="requires at least one path") as info:
        facade([])
    del info
    gc.collect()
    assert unraisable == []
