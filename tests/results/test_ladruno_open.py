"""M2-S7 (#2991): the shared ``.ladruno`` open, the reader rule's steps 1-3.

The rule is the fork's contract 232a §2.5 at ``a1d4c7260`` (vendored with its
reference code in ``tests/fixtures/mpm/fork_a1d4c7260/``). The oracles:

- the fork's own ``lookup3`` / ``set_superblock`` (and HDF5's checksum in a
  real version-3 superblock) for the superblock arithmetic;
- the fork's killed-run image ``particles_killed_swmr_replay.h5`` (made with
  the fork's ``hdf5_crash_replay.py`` at ``f0b9a1df6``; see the fixture
  README), whose committed outputs the fork's ``committed_outputs`` counts and
  whose rows the replay writes in closed form;
- ``sha256`` of the user's file before and after every open: the repair
  touches only its copy.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results.readers import _ladruno_open as lo
from apeGmsh.results.readers._ladruno_open import (
    STEP_PLAIN,
    STEP_REPAIRED,
    STEP_SWMR,
    LadrunoFileBusyError,
    lookup3,
    open_ladruno,
    repair_copy,
    superblock_flags,
)

MPM = Path(__file__).resolve().parents[1] / "fixtures" / "mpm"
ORACLE = MPM / "fork_a1d4c7260"
KILLED = MPM / "particles_killed_swmr_replay.h5"
V3 = MPM / "particles_torn_last_stage_v1m2.ladruno"   # closed, version-3 superblock
V0 = MPM / "particles_v1m2.ladruno"                   # closed, version-0 superblock
ON_P = "MODEL_STAGE[0]/RESULTS/ON_PARTICLES"

#: The replay's committed rows in closed form (hdf5_crash_replay.py ``rows`` / ``IDS``).
REPLAY_IDS = np.array([3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610], dtype=np.int64)
REPLAY_FIELDS = (("POSITION", 2), ("VELOCITY", 2), ("STRESS", 3), ("VOLUME", 1))
REPLAY_COMMITTED = 4


def replay_rows(k: int, nc: int) -> np.ndarray:
    i = np.arange(REPLAY_IDS.size)[:, None]
    c = np.arange(nc)[None, :]
    return (k * 1000.0 + i * 10.0 + c + 0.5).astype(np.float64)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def fork():
    """The fork's ``ladruno_format`` and ``mpm_contracts`` at a1d4c7260, with
    ``sys.path`` / ``sys.modules`` restored."""
    names = ("mpm_contracts", "ladruno_format", "ladruno_basis")
    saved_path = list(sys.path)
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    sys.path.insert(0, str(ORACLE))
    try:
        yield importlib.import_module("mpm_contracts"), importlib.import_module("ladruno_format")
    finally:
        sys.path[:] = saved_path
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


@pytest.fixture
def flagged(tmp_path, fork):
    """A copy of the closed version-3 fixture with its flags (and EOA) set by
    the fork's own ``set_superblock``."""
    _, lf = fork

    def make(flags: int, eoa_extra: int = 0) -> Path:
        p = tmp_path / f"flags{flags:02x}_{eoa_extra}.ladruno"
        shutil.copyfile(V3, p)
        eoa = p.stat().st_size + eoa_extra if eoa_extra else None
        lf.set_superblock(p, flags=flags, eoa=eoa)
        return p
    return make


def _committed_rows(f: h5py.File, n: int) -> dict:
    op = f[ON_P]
    offs = op["OFFSETS"][: n + 1]
    end = int(offs[-1])
    out = {"OFFSETS": offs, "STEP": op["STEP"][:n], "TIME": op["TIME"][:n],
           "ID": op["ID"][:end]}
    for name, _ in REPLAY_FIELDS:
        out[name] = op[name]["DATA"][:end]
    return out


# --- the superblock arithmetic ---------------------------------------------------------

def test_lookup3_is_the_forks(fork) -> None:
    _, lf = fork
    rng = np.random.default_rng(2991)
    for n in list(range(0, 40)) + [44, 100, 257]:
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        assert lookup3(data) == lf.lookup3(data)


@pytest.mark.parametrize("path", [V3, KILLED], ids=lambda p: p.name)
def test_lookup3_is_hdf5s_superblock_checksum(path: Path) -> None:
    sb = path.read_bytes()[:48]
    assert sb[8] in (2, 3)
    assert lookup3(sb[:44]) == int.from_bytes(sb[44:48], "little")


def test_superblock_flags() -> None:
    assert superblock_flags(V3) == 0x00
    assert superblock_flags(KILLED) == 0x05
    assert superblock_flags(V0) is None


# --- step 1 -------------------------------------------------------------------------------

@pytest.mark.parametrize("path, flags", [(V3, 0x00), (V0, None)], ids=["v3", "v0"])
def test_a_closed_file_opens_plainly(path: Path, flags) -> None:
    before = sha(path)
    with open_ladruno(path) as o:
        assert (o.step, o.flags, o.copy) == (STEP_PLAIN, flags, None)
        assert o.h5.filename == str(path)
    assert sha(path) == before


# --- step 2: the fork's killed run --------------------------------------------------------

def test_the_killed_run_needs_step_2(fork) -> None:
    with pytest.raises(OSError, match="already open for write"):
        h5py.File(KILLED, "r")
    before = sha(KILLED)
    with open_ladruno(KILLED) as o:
        assert (o.step, o.flags, o.copy) == (STEP_SWMR, 0x05, None)
        assert o.h5.swmr_mode
    assert sha(KILLED) == before


def test_the_killed_run_reads_the_oracles_committed_outputs(fork) -> None:
    mc, _ = fork
    from apeGmsh.results.readers._ladruno_particles import committed_outputs

    assert mc.committed_outputs(str(KILLED), stage="MODEL_STAGE[0]") == REPLAY_COMMITTED
    with open_ladruno(KILLED) as o:
        op = o.h5[ON_P]
        # the image holds a partial fifth output past the committed four
        assert op["STEP"].shape[0] == REPLAY_COMMITTED + 1
        assert op["ID"].shape[0] == (REPLAY_COMMITTED + 1) * REPLAY_IDS.size
        assert committed_outputs(op) == REPLAY_COMMITTED
        rows = _committed_rows(o.h5, REPLAY_COMMITTED)
    for k in range(REPLAY_COMMITTED):
        a, b = int(rows["OFFSETS"][k]), int(rows["OFFSETS"][k + 1])
        np.testing.assert_array_equal(rows["ID"][a:b], REPLAY_IDS)
        assert rows["STEP"][k] == k + 1 and rows["TIME"][k] == (k + 1) * 1e-6
        for name, nc in REPLAY_FIELDS:
            np.testing.assert_array_equal(rows[name][a:b], replay_rows(k, nc))


def test_a_repaired_copy_of_the_killed_run_reads_as_swmr_read(tmp_path, fork) -> None:
    """The fork's claim behind the h5wasm line of §2.5: step 3's repair on a
    flags-0x05 file gives the committed outputs and byte-identical rows of
    the SWMR-read open."""
    mc, _ = fork
    before = sha(KILLED)
    dst = tmp_path / "repaired.h5"
    repair_copy(KILLED, dst)
    assert sha(KILLED) == before
    assert superblock_flags(dst) == 0x00
    with h5py.File(KILLED, "r", swmr=True) as f:
        n_swmr = mc.committed_outputs(f["MODEL_STAGE[0]"])
        swmr_rows = _committed_rows(f, n_swmr)
    with h5py.File(dst, "r") as f:
        n_copy = mc.committed_outputs(f["MODEL_STAGE[0]"])
        copy_rows = _committed_rows(f, n_copy)
    assert n_swmr == n_copy == REPLAY_COMMITTED
    for k in swmr_rows:
        assert swmr_rows[k].tobytes() == copy_rows[k].tobytes(), k


# --- step 3 -------------------------------------------------------------------------------

@pytest.mark.parametrize("eoa_extra", [0, 4096], ids=["eoa=size", "eoa-past-end"])
def test_a_layout_kill_opens_a_repaired_copy(flagged, tmp_path, eoa_extra) -> None:
    p = flagged(0x01, eoa_extra)
    before = sha(p)
    size = p.stat().st_size
    with pytest.raises(OSError):
        h5py.File(p, "r")
    with pytest.raises(OSError, match="not already open for SWMR writing"):
        h5py.File(p, "r", swmr=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with open_ladruno(p, scratch_dir=scratch) as o:
        assert (o.step, o.flags) == (STEP_REPAIRED, 0x01)
        assert o.copy is not None and o.copy.parent.parent == scratch
        assert o.copy.stat().st_size == size + eoa_extra   # zero-padded to the EOA
        assert superblock_flags(o.copy) == 0x00
        assert "MODEL_STAGE[1]" in o.h5
    assert not o.copy.exists() and list(scratch.iterdir()) == []
    assert sha(p) == before


def test_flags_0x05_on_a_closed_layout_open_swmr_read(flagged) -> None:
    p = flagged(0x05)
    before = sha(p)
    with open_ladruno(p) as o:
        assert (o.step, o.flags, o.copy) == (STEP_SWMR, 0x05, None)
    assert sha(p) == before


def test_any_other_refusal_reraises_the_plain_error(tmp_path) -> None:
    p = tmp_path / "not.ladruno"
    p.write_bytes(b"not an HDF5 file at all" * 4)
    with pytest.raises(OSError) as e:
        open_ladruno(p)
    assert not isinstance(e.value, LadrunoFileBusyError)


def test_repair_copy_refuses_the_file_itself_and_other_superblocks(tmp_path) -> None:
    with pytest.raises(ValueError, match="dst must differ"):
        repair_copy(V3, V3)
    with pytest.raises(ValueError, match="version-2/3"):
        repair_copy(V0, tmp_path / "x.h5")
    bad = tmp_path / "bad.ladruno"
    raw = bytearray(V3.read_bytes())
    raw[44] ^= 0xFF
    bad.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum"):
        repair_copy(bad, tmp_path / "y.h5")


# --- a live writer -------------------------------------------------------------------------

def _no_repair(*_a, **_k):
    raise AssertionError("a busy file must never be copied or repaired")


def test_a_lock_error_is_busy_and_nothing_is_repaired(flagged, monkeypatch, tmp_path) -> None:
    """HDF5's "unable to lock file" (the message h5py raises on every
    platform) is a live writer: a distinct error, and no copy."""
    p = flagged(0x01)
    before = sha(p)
    real = h5py.File

    def locked(name, mode="r", *a, **k):
        if Path(name) == p:
            raise OSError("Unable to synchronously open file (unable to lock file, "
                          "errno = 11, error message = 'Resource temporarily unavailable')")
        return real(name, mode, *a, **k)

    monkeypatch.setattr(h5py, "File", locked)
    monkeypatch.setattr(lo, "repair_copy", _no_repair)
    monkeypatch.setattr(lo.tempfile, "mkdtemp", _no_repair)
    with pytest.raises(LadrunoFileBusyError, match="live writer"):
        open_ladruno(p)
    assert sha(p) == before


def test_a_held_lock_is_busy(flagged, monkeypatch) -> None:
    """A real lock on the file, held through another descriptor of this
    process: ``flock`` on POSIX (HDF5's own lock call), a byte-range lock on
    Windows (where HDF5 locks the whole file, and a locked range also refuses
    the raw superblock read)."""
    p = flagged(0x05)
    before = sha(p)
    monkeypatch.setattr(lo, "repair_copy", _no_repair)
    fd = os.open(p, os.O_RDWR)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1 << 20)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(LadrunoFileBusyError):
            open_ladruno(p)
    finally:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1 << 20)
        os.close(fd)
    assert sha(p) == before
