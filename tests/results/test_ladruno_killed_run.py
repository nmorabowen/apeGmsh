"""M2-S8 (#3003): ``Results.from_ladruno`` reads a killed recorder run.

The contract is the fork's ``Ladruno_implementation/ladruno_schema_v1.md``
§7b.2 at the release ``ladruno-v20261116`` (``6ee968060``, the ``FORK_PIN``):
the reader rule's open (plain, else SWMR-read, else a repaired copy; never the
user's file), the stage rule (a torn LAST stage is skipped after a plain open or
a repaired copy, refused after an SWMR-read open; an earlier stage that does not
read is corruption), and the crash rule (a killed file's stage is read up to
T' = the fewest rows any of its streaming results holds in DATA, TIME or STEP;
a cleanly closed file is read as it is).

The oracles:

- the fork's reference ``ladruno_format.py`` at the pin, vendored byte for byte
  in ``tests/fixtures/ladruno_killed/fork_6ee968060/``: its ``committed_steps``
  and ``stage_report`` name every T' and every torn-stage verdict;
- ``killed_swmr.ladruno``, a run of the release build killed while it recorded
  (``make_killed_run.py``; see the fixture README), and the raw rows h5py reads
  from it;
- ``sha256`` of the user's file before and after every read;
- live (``ladruno_fork``), the same deck run to completion at the pin: a killed
  run's rows are a prefix of it, bit for bit (the analysis is deterministic).

Before this slice ``LadrunoReader`` opened with a plain ``h5py.File(path, "r")``,
which refuses every file killed with SWMR armed ("already open for write").
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import shutil
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers import _ladruno_open as lo
from apeGmsh.results.readers._ladruno import LadrunoReader, read_stage_stamp
from apeGmsh.results.readers._ladruno_open import (
    STEP_PLAIN,
    STEP_REPAIRED,
    STEP_SWMR,
    SWMR_TAIL_READ,
    LadrunoFileBusyError,
    LadrunoStageCorrupt,
    LadrunoStageSkippedWarning,
    open_ladruno,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno_killed"
KILLED = FIX / "killed_swmr.ladruno"
ORACLE = FIX / "fork_6ee968060"
S1, S2 = "MODEL_STAGE[1]", "MODEL_STAGE[2]"
DISP = "RESULTS/ON_NODES/DISPLACEMENT"
FORCE = "RESULTS/ON_ELEMENTS/force/3-ElasticBeam2d[1:0:0]"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def lf():
    """The fork's ``ladruno_format`` at the pin, with ``sys.path`` and
    ``sys.modules`` restored."""
    names = ("ladruno_format", "ladruno_basis")
    saved_path = list(sys.path)
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    sys.path.insert(0, str(ORACLE))
    try:
        yield importlib.import_module("ladruno_format")
    finally:
        sys.path[:] = saved_path
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def _reference_report(lf, path: Path, flags: int) -> list:
    """The fork's stage report of ``path`` by the reader rule: steps 1-2 with
    ``open_ladruno``, step 3 with ``recover_ladruno`` on a copy."""
    try:
        with lf.open_ladruno(str(path)) as f:
            return lf.stage_report(f)
    except lf.StageLayoutKilledError:
        assert flags == 0x01
        dst = path.with_name(path.stem + ".ref-recovered.ladruno")
        try:
            return lf.recover_ladruno(str(path), str(dst))
        finally:
            dst.unlink(missing_ok=True)


def _copy(tmp_path: Path, lf, name: str, flags: "int | None" = None, *,
          eoa_to_size: bool = False, src: Path = KILLED) -> Path:
    p = tmp_path / name
    shutil.copyfile(src, p)
    if flags is not None or eoa_to_size:
        lf.set_superblock(p, flags=flags,
                          eoa=p.stat().st_size if eoa_to_size else None)
    return p


def _ragged(tmp_path: Path, lf, flags: int) -> Path:
    """The fixture with MODEL_STAGE[2] ragged the way a kill inside a flush
    leaves it (schema §7b.2: DATA, TIME and STEP are separate datasets):
    DISPLACEMENT/DATA holds 14 rows and force/STEP 13, the rest 15. Built on a
    copy repaired as the rule repairs one, then given ``flags``."""
    p = _copy(tmp_path, lf, f"ragged_{flags:02x}.ladruno", 0x00, eoa_to_size=True)
    with h5py.File(p, "r+") as f:
        f[f"{S2}/{DISP}/DATA"].resize(14, axis=0)
        f[f"{S2}/{FORCE}/STEP"].resize(13, axis=0)
    lf.set_superblock(p, flags=flags)
    return p


def _tear(path: Path, stage: str) -> None:
    """Flip the signature of ``stage``'s object header: the group's link in
    the root stays, the group no longer opens (a stage torn while laid out)."""
    with open_ladruno(path) as o:
        addr = h5py.h5o.get_info(o.h5[stage].id).addr
    raw = bytearray(path.read_bytes())
    assert raw[addr:addr + 4] == b"OHDR"
    raw[addr] ^= 0xFF
    path.write_bytes(bytes(raw))


def _raw(path: Path, stage: str, result: str, rows: int) -> np.ndarray:
    with open_ladruno(path) as o:
        return np.asarray(o.h5[f"{stage}/{result}/DATA"][:rows])


def _time(path: Path, stage: str, rows: int) -> np.ndarray:
    with open_ladruno(path) as o:
        return np.asarray(o.h5[f"{stage}/{DISP}/TIME"][:rows], dtype=np.float64)


# --- the fixture --------------------------------------------------------------------------

def test_the_fixture_is_a_run_killed_with_swmr_armed(lf) -> None:
    """Flags 0x05, which a plain open refuses: the open the reader used before."""
    assert lo.superblock_flags(KILLED) == 0x05
    with pytest.raises(OSError, match="already open for write"):
        h5py.File(KILLED, "r")
    assert _reference_report(lf, KILLED, 0x05) == [
        {"stage": S1, "committed": 20}, {"stage": S2, "committed": 15}]


# --- a killed run reads to its committed steps --------------------------------------------

def test_from_ladruno_reads_a_killed_run_to_its_committed_steps(lf) -> None:
    before = sha(KILLED)
    ref = {r["stage"]: r["committed"] for r in _reference_report(lf, KILLED, 0x05)}
    with Results.from_ladruno(KILLED) as results:
        assert {s.name: s.n_steps for s in results.stages
                if s.kind != "mode"} == ref
    assert sha(KILLED) == before
    with LadrunoReader(KILLED) as reader:
        assert (reader.open_step, reader.killed) == (STEP_SWMR, True)
        hosts = {s.name: s for s in reader.stages() if s.kind != "mode"}
        assert {n: s.n_steps for n, s in hosts.items()} == ref
        for name, s in hosts.items():
            assert reader.committed_steps(s.id) == ref[name]
            nodes = reader.read_nodes(s.id, "displacement_x")
            np.testing.assert_array_equal(
                nodes.values, _raw(KILLED, name, DISP, ref[name])[:, :, 0])
            np.testing.assert_array_equal(nodes.time, _time(KILLED, name, ref[name]))
            assert reader.read_elements(s.id, "force").values.shape[0] == ref[name]
    assert sha(KILLED) == before


@pytest.mark.parametrize("flags, step", [(0x05, STEP_SWMR), (0x01, STEP_REPAIRED)],
                         ids=["swmr-0x05", "repaired-0x01"])
def test_a_ragged_stage_of_a_killed_run_is_cut_to_t_prime(tmp_path, lf, flags, step) -> None:
    p = _ragged(tmp_path, lf, flags)
    before = sha(p)
    ref = {r["stage"]: r["committed"] for r in _reference_report(lf, p, flags)}
    assert ref == {S1: 20, S2: 13}
    reader = LadrunoReader(p)
    try:
        assert reader.open_step == step
        copy = reader._opened.copy
        assert (copy is not None) == (step == STEP_REPAIRED)
        hosts = {s.name: s for s in reader.stages() if s.kind != "mode"}
        assert {n: s.n_steps for n, s in hosts.items()} == ref
        nodes = reader.read_nodes(hosts[S2].id, "displacement_y")
        assert nodes.values.shape == (13, 5)
        np.testing.assert_array_equal(nodes.values, _raw(p, S2, DISP, 13)[:, :, 1])
        np.testing.assert_array_equal(
            reader.read_elements(hosts[S2].id, "force").values,
            reader.read_elements(hosts[S2].id, "force", time_slice=slice(0, 13)).values)
        assert reader.read_elements(hosts[S2].id, "force").values.shape[0] == 13
    finally:
        reader.close()
    if copy is not None:
        assert not copy.exists()
    assert sha(p) == before


def test_a_cleanly_closed_ragged_file_reads_as_it_is(tmp_path, lf) -> None:
    """Flags 0x00: no cut. The stage keeps the length of its first TIME (15),
    as the reader read every closed file before this slice."""
    p = _ragged(tmp_path, lf, 0x00)
    reader = LadrunoReader(p)
    try:
        assert (reader.open_step, reader.killed) == (STEP_PLAIN, False)
        hosts = {s.name: s for s in reader.stages() if s.kind != "mode"}
        assert {n: s.n_steps for n, s in hosts.items()} == {S1: 20, S2: 15}
        assert reader.committed_steps(hosts[S2].id) is None
        np.testing.assert_array_equal(reader.time_vector(hosts[S2].id),
                                      _time(p, S2, 15))
    finally:
        reader.close()


# --- the stage rule -----------------------------------------------------------------------

@pytest.mark.parametrize("flags", [0x00, 0x01], ids=["plain-0x00", "repaired-0x01"])
def test_a_torn_last_stage_is_skipped_after_a_plain_open_or_a_repair(tmp_path, lf, flags) -> None:
    p = _copy(tmp_path, lf, f"torn_last_{flags:02x}.ladruno", flags,
              eoa_to_size=flags == 0x00)
    _tear(p, S2)
    ref = _reference_report(lf, p, flags)
    assert ref[0] == {"stage": S1, "committed": 20} and ref[1]["torn"] is True
    before = sha(p)
    with pytest.warns(LadrunoStageSkippedWarning, match=r"MODEL_STAGE\[2\].*torn"):
        reader = LadrunoReader(p)
    try:
        assert [s.name for s in reader.stages() if s.kind != "mode"] == [S1]
        assert reader.stages()[0].n_steps == 20
        assert [s.id for s in reader.stages() if s.kind == "mode"] == [
            "stage_0_mode_1", "stage_0_mode_2"]
        assert [r.group for r in read_stage_stamp([p]).stages] == [S1]
    finally:
        reader.close()
    assert sha(p) == before


def test_a_torn_last_stage_refuses_a_file_killed_with_swmr_armed(tmp_path, lf) -> None:
    p = _copy(tmp_path, lf, "torn_last_05.ladruno")
    _tear(p, S2)
    with pytest.raises(ValueError, match="SWMR armed"):
        _reference_report(lf, p, 0x05)
    with pytest.raises(LadrunoStageCorrupt, match=r"MODEL_STAGE\[2\].*SWMR armed"):
        LadrunoReader(p)


@pytest.mark.parametrize("flags", [0x00, 0x01, 0x05], ids=["0x00", "0x01", "0x05"])
def test_an_earlier_stage_that_does_not_read_refuses_the_file(tmp_path, lf, flags) -> None:
    p = _copy(tmp_path, lf, f"torn_first_{flags:02x}.ladruno", flags,
              eoa_to_size=flags == 0x00)
    _tear(p, S1)
    # The reference names the SWMR rule first on a 0x05 file; both refuse.
    with pytest.raises(ValueError, match="corruption"):
        _reference_report(lf, p, flags)
    with pytest.raises(LadrunoStageCorrupt, match=r"MODEL_STAGE\[1\].*corruption"):
        LadrunoReader(p)


# --- the modes-snapshot crash rule --------------------------------------------------------
#
# Schema §7b.2 (the crash rule, KILLED files only) and T's ruling of 2026-11-18 on #3003:
# a modes snapshot that does not open, or lacks a MODE_<j> (a gap in its mode numbers, or
# modal groups that disagree), is skipped with every later one at flags 0x01; at flags
# 0x05 (every stage must read) it refuses by name; a cleanly closed file (0x00) keeps
# main's named refusals and refuses a snapshot that does not open by name, never a raw
# KeyError. A smaller snapshot after a larger one is a whole snapshot: the writer stores
# one snapshot per eigen call, with that call's mode count.

MODES = "RESULTS/ON_NODES/MODES_OF_VIBRATION(U)"
MODES_U = f"{MODES}/DATA"
MODES_R = "RESULTS/ON_NODES/MODES_OF_VIBRATION(R)"
#: The object header of MODEL_STAGE[1]'s modes snapshot STEP_10 in KILLED (asserted below).
STEP_10_OHDR = 260827
#: The mode stages of KILLED: MODEL_STAGE[1]'s STEP_10 holds MODE_0 and MODE_1.
STAGE_1_MODES = ["stage_0_mode_1", "stage_0_mode_2"]


def _mode_ids(reader: LadrunoReader) -> list:
    return [s.id for s in reader.stages() if s.kind == "mode"]


def _corrupt_step_10(tmp_path: Path, lf, flags: int) -> Path:
    p = _copy(tmp_path, lf, f"snap_ohdr_{flags:02x}.ladruno", flags,
              eoa_to_size=flags == 0x00)
    with open_ladruno(p) as o:
        assert h5py.h5o.get_info(o.h5[f"{S1}/{MODES_U}/STEP_10"].id).addr == STEP_10_OHDR
    raw = bytearray(p.read_bytes())
    assert raw[STEP_10_OHDR:STEP_10_OHDR + 4] == b"OHDR"
    raw[STEP_10_OHDR] ^= 0xFF
    p.write_bytes(bytes(raw))
    return p


@pytest.mark.parametrize("flags", [0x00, 0x05], ids=["0x00", "0x05"])
def test_a_modes_snapshot_that_does_not_open_is_refused_by_name(tmp_path, lf, flags) -> None:
    """A cleanly closed file (0x00) and one killed with SWMR armed (0x05) must read
    every snapshot: one that does not open refuses by name, never a raw KeyError."""
    from apeGmsh.results.readers._buckets import ModeSnapshotCorrupt

    p = _corrupt_step_10(tmp_path, lf, flags)
    before = sha(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ModeSnapshotCorrupt, match=r"MODEL_STAGE\[1\]/STEP_10 does not"):
            with Results.from_ladruno(p):
                pass
        with pytest.raises(ModeSnapshotCorrupt, match=r"MODEL_STAGE\[1\]/STEP_10 does not"):
            with LadrunoReader(p) as reader:
                reader.stages()
    assert sha(p) == before


def test_a_modes_snapshot_that_does_not_open_is_skipped_at_0x01(tmp_path, lf) -> None:
    """Killed while a layout was written (flags 0x01, a repaired copy): the snapshot
    that does not open was being written; it is skipped with every later one, named."""
    from apeGmsh.results.readers._buckets import ModeSnapshotSkippedWarning

    p = _corrupt_step_10(tmp_path, lf, 0x01)
    before = sha(p)
    with pytest.warns(ModeSnapshotSkippedWarning, match=r"MODEL_STAGE\[1\]/STEP_10"):
        with Results.from_ladruno(p) as results:
            assert [s.kind for s in results.stages] == ["static", "static"]
    with pytest.warns(ModeSnapshotSkippedWarning):
        with LadrunoReader(p) as reader:
            assert reader.open_step == STEP_REPAIRED
            assert _mode_ids(reader) == []
            assert [s.n_steps for s in reader.stages()] == [20, 15]
    assert sha(p) == before


def _last_snapshot(tmp_path: Path, lf, name: str, flags: int, modes: "list[int]", *,
                   r_modes: "list[int] | None" = None,
                   lam: bool = True) -> Path:
    """KILLED with a last-stage snapshot MODEL_STAGE[2]/STEP_25 holding ``modes``
    (copies of STEP_10's MODE_0, each a whole mode with its LAMBDA unless ``lam`` is
    False); with ``r_modes``, a MODES_OF_VIBRATION(R) group too, whose STEP_25 holds
    ``r_modes``. Built on a copy repaired as the rule repairs one, then given ``flags``."""
    p = _copy(tmp_path, lf, name, 0x00, eoa_to_size=True)
    with h5py.File(p, "r+") as f:
        src = f[f"{S1}/{MODES_U}/STEP_10/MODE_0"]

        def snapshot(data: "h5py.Group", which: "list[int]") -> None:
            g = data.create_group("STEP_25")
            for j in which:
                f.copy(src, g, f"MODE_{j}")
                if not lam:
                    del g[f"MODE_{j}"].attrs["LAMBDA"]

        snapshot(f[f"{S2}/{MODES_U}"], modes)
        if r_modes is not None:
            f.copy(f[f"{S2}/{MODES}"], f[f"{S2}/RESULTS/ON_NODES"], "MODES_OF_VIBRATION(R)")
            del f[f"{S2}/{MODES_R}/DATA/STEP_25"]
            snapshot(f[f"{S2}/{MODES_R}/DATA"], r_modes)
    lf.set_superblock(p, flags=flags)
    return p


@pytest.mark.parametrize("flags", [0x00, 0x01], ids=["0x00", "0x01"])
def test_a_whole_one_mode_last_snapshot_is_listed(tmp_path, lf, flags) -> None:
    """``eigen 1`` after ``eigen 2``: a last-stage STEP_25 holding MODE_0 and its
    LAMBDA is a whole snapshot, listed as a mode stage (as main lists it)."""
    from apeGmsh.results.readers._buckets import ModeSnapshotSkippedWarning

    p = _last_snapshot(tmp_path, lf, f"one_mode_{flags:02x}.ladruno", flags, [0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModeSnapshotSkippedWarning)
        with LadrunoReader(p) as reader:
            assert _mode_ids(reader) == STAGE_1_MODES + ["stage_1_mode_1"]
            assert [s.id for s in reader.stages()][-2:] == ["stage_1", "stage_1_mode_1"]


@pytest.mark.parametrize("case", ["gap", "disagree"])
def test_a_last_snapshot_that_lacks_a_mode_is_skipped_at_0x01(tmp_path, lf, case) -> None:
    """At 0x01 a last-stage STEP_25 that lacks a mode was being written: one whose
    mode numbers have a gap (MODE_0, MODE_2), or whose (U) group holds MODE_0 and
    MODE_1 while (R) holds only MODE_0. It is skipped and named; STEP_10 still reads."""
    from apeGmsh.results.readers._buckets import ModeSnapshotSkippedWarning

    if case == "gap":
        p = _last_snapshot(tmp_path, lf, "gap.ladruno", 0x01, [0, 2])
    else:
        p = _last_snapshot(tmp_path, lf, "disagree.ladruno", 0x01, [0, 1], r_modes=[0])
    with pytest.warns(ModeSnapshotSkippedWarning,
                      match=r"MODEL_STAGE\[2\]/STEP_25 does not read whole \(.*lacks a mode\)"):
        with LadrunoReader(p) as reader:
            assert _mode_ids(reader) == STAGE_1_MODES


@pytest.mark.parametrize("case", ["gap", "disagree"])
def test_a_last_snapshot_that_lacks_a_mode_is_refused_at_0x05(tmp_path, lf, case) -> None:
    """At 0x05 every snapshot must read: a snapshot that lacks a mode refuses by name."""
    from apeGmsh.results.readers._buckets import ModeSnapshotCorrupt

    if case == "gap":
        p = _last_snapshot(tmp_path, lf, "gap5.ladruno", 0x05, [0, 2])
    else:
        p = _last_snapshot(tmp_path, lf, "disagree5.ladruno", 0x05, [0, 1], r_modes=[0])
    with pytest.raises(ModeSnapshotCorrupt, match=r"MODEL_STAGE\[2\]/STEP_25 does not read whole"):
        with LadrunoReader(p) as reader:
            reader.stages()


def test_a_clean_file_with_a_mode_missing_lambda_is_refused_as_on_main(tmp_path, lf) -> None:
    """A cleanly closed file (0x00) whose last snapshot's MODE_0 has no LAMBDA: main's
    named refusal, never a skip."""
    p = _last_snapshot(tmp_path, lf, "no_lambda.ladruno", 0x00, [0], lam=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(ValueError, match=r"STEP_25/MODE_0 has no LAMBDA attribute"):
            with LadrunoReader(p) as reader:
                reader.stages()


# --- the energy read's cut ----------------------------------------------------------------

def test_read_energy_stops_at_t_prime(tmp_path, lf) -> None:
    """MODEL_STAGE[2] with 15 energy rows and 13 DISPLACEMENT/DATA rows, killed
    (flags 0x05): T' = 13, and the energy history stops there."""
    p = _copy(tmp_path, lf, "energy.ladruno", 0x00, eoa_to_size=True)
    rows = np.arange(15 * 3, dtype=np.float64).reshape(15, 1, 3)
    with h5py.File(p, "r+") as f:
        stage = f[S2]
        time = stage[f"{DISP}/TIME"][...]
        steps = stage[f"{DISP}/STEP"][...]
        eg = stage.require_group("RESULTS/ON_DOMAIN").create_group("energyBalance")
        eg.attrs["COMPONENTS"] = np.bytes_("kinetic,strain,external")
        eg.create_dataset("ID", data=np.array([[0]], dtype=np.int32))
        eg.create_dataset("DATA", data=rows, maxshape=(None, 1, 3), chunks=(64, 1, 3))
        eg.create_dataset("TIME", data=time, maxshape=(None,), chunks=(1024,))
        eg.create_dataset("STEP", data=steps, maxshape=(None,), chunks=(1024,))
        stage[f"{DISP}/DATA"].resize(13, axis=0)
    lf.set_superblock(p, flags=0x05)
    ref = {r["stage"]: r["committed"] for r in _reference_report(lf, p, 0x05)}
    assert ref == {S1: 20, S2: 13}
    with LadrunoReader(p) as reader:
        names, values, t = reader.read_energy("stage_1")
        assert names == ["kinetic", "strain", "external"]
        assert values.shape == (13, 3)
        np.testing.assert_array_equal(values, rows[:13, 0, :])
        np.testing.assert_array_equal(t, time[:13])


# --- a live writer ------------------------------------------------------------------------

def test_a_live_writers_lock_is_busy_and_names_the_swmr_tail_read(tmp_path, lf, monkeypatch) -> None:
    """HDF5's "unable to lock file" (what a locking reader meets while the
    recorder's guard lock is held) is busy; nothing is copied or repaired."""
    p = _copy(tmp_path, lf, "live.ladruno")
    before = sha(p)
    real = h5py.File

    def locked(name, mode="r", *a, **k):
        if Path(name) == p:
            raise OSError("Unable to synchronously open file (unable to lock file, "
                          "errno = 11, error message = 'Resource temporarily unavailable')")
        return real(name, mode, *a, **k)

    def no_repair(*_a, **_k):
        raise AssertionError("a busy file must never be copied or repaired")

    monkeypatch.setattr(h5py, "File", locked)
    monkeypatch.setattr(lo, "repair_copy", no_repair)
    for open_it in (LadrunoReader, Results.from_ladruno):
        with pytest.raises(LadrunoFileBusyError) as e:
            open_it(p)
        assert "live writer" in str(e.value)
        assert SWMR_TAIL_READ in str(e.value)
    assert sha(p) == before


# --- the stamp's stage read ---------------------------------------------------------------

def test_the_stage_stamp_reads_a_killed_run_while_a_reader_holds_it() -> None:
    reader = LadrunoReader(KILLED)
    try:
        stamp = read_stage_stamp([KILLED])
        assert [r.group for r in stamp.stages] == [S1, S2]
    finally:
        reader.close()


# --- live, at the pin ---------------------------------------------------------------------

def _maker():
    spec = importlib.util.spec_from_file_location(
        "_apegmsh_make_killed_run", FIX / "make_killed_run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fem_archive(path: Path) -> Path:
    """A bare FEMData archive (``fem.to_h5``): the model a stamp pairs with."""
    from apeGmsh.mesh._element_types import ElementGroup, make_type_info
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        ElementComposite,
        FEMData,
        MeshInfo,
        NodeComposite,
    )

    ids = np.arange(1, 6, dtype=np.int64)
    coords = np.column_stack([np.zeros(5), 0.5 * np.arange(5), np.zeros(5)])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=4)
    group = ElementGroup(element_type=info, ids=np.arange(1, 5, dtype=np.int64),
                         connectivity=np.column_stack([ids[:-1], ids[1:]]))
    pg = {(1, 1): {"name": "Beam", "node_ids": ids, "node_coords": coords,
                   "element_ids": group.ids}}
    nodes = NodeComposite(node_ids=ids, node_coords=coords,
                          physical=PhysicalGroupSet(pg), labels=LabelSet({}))
    elements = ElementComposite(groups={1: group}, physical=PhysicalGroupSet(pg),
                                labels=LabelSet({}))
    FEMData(nodes=nodes, elements=elements,
            info=MeshInfo(n_nodes=5, n_elems=4, bandwidth=2, types=[info])).to_h5(str(path))
    return path


def _require_pin(build: str, pin: str) -> None:
    """The SWMR recorder is the pinned build's (OpenSees#1084): another build
    skips here, and fails in the live-fork lane, which runs the pin."""
    if build == pin:
        return
    msg = f"the child ran build {build[:12]!r}, not the FORK_PIN build {pin[:12]}"
    if os.environ.get("APEGMSH_FORK_PIN_ENFORCE") == "1":
        pytest.fail(msg)
    pytest.skip(msg + "; set APEGMSH_OPENSEES_BIN to the pinned build's dist/bin")


def _wait_for_step(proc, k: int) -> str:
    build = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("BUILD "):
            build = line.split()[1]
        if line.strip() == f"STEP {k}":
            return build
    raise AssertionError(f"the child ended before step {k} (exit {proc.wait()})")


@pytest.mark.ladruno_fork
def test_live_a_killed_recorder_run_reads_as_the_completed_runs_prefix(tmp_path, lf, monkeypatch) -> None:
    """At the pin: while the recorder runs, every locking open is busy and the
    stamp refuses; after a kill the reader reads the committed steps the fork's
    ``committed_steps`` names, and they equal the completed run's rows."""
    from apeGmsh.results.writers import ProvenanceStampRefused, stamp_provenance

    monkeypatch.delenv("HDF5_USE_FILE_LOCKING", raising=False)
    mk = _maker()
    bin_dir = os.environ.get("APEGMSH_OPENSEES_BIN") or "-"
    pin = mk.pinned_sha()

    reference = tmp_path / "reference.ladruno"
    proc = mk.start(bin_dir, reference, tmp_path, sleep=0.0, nsteps=40)
    try:
        build = _wait_for_step(proc, 40)
        assert proc.wait(timeout=120) == 0
    finally:
        proc.kill()
        proc.wait(timeout=120)
    _require_pin(build, pin)
    assert lo.superblock_flags(reference) == 0x00

    killed = tmp_path / "killed.ladruno"
    model = _fem_archive(tmp_path / "model.h5")
    proc = mk.start(bin_dir, killed, tmp_path, sleep=0.05)
    try:
        assert _wait_for_step(proc, 30) == pin
        # Live: the recorder's guard lock refuses every locking open.
        with pytest.raises(LadrunoFileBusyError, match="live writer"):
            open_ladruno(killed)
        with pytest.raises(LadrunoFileBusyError) as e:
            LadrunoReader(killed)
        assert SWMR_TAIL_READ in str(e.value)
        with pytest.raises(ProvenanceStampRefused, match="cannot be opened"):
            stamp_provenance(killed, model)
        # The SWMR tail read the message names does read the live file.
        with h5py.File(killed, "r", swmr=True, locking=False) as f:
            assert S1 in f
    finally:
        proc.kill()
        proc.wait(timeout=120)

    before = sha(killed)
    ref = {r["stage"]: r["committed"] for r in _reference_report(
        lf, killed, lo.superblock_flags(killed))}
    assert ref[S1] == 20 and 10 <= ref[S2] <= 20
    with warnings.catch_warnings():
        warnings.simplefilter("error", LadrunoStageSkippedWarning)
        r_k, r_ref = LadrunoReader(killed), LadrunoReader(reference)
    try:
        assert r_k.killed and not r_ref.killed
        k_hosts = [s for s in r_k.stages() if s.kind != "mode"]
        ref_hosts = [s for s in r_ref.stages() if s.kind != "mode"]
        assert [s.n_steps for s in k_hosts] == [ref[S1], ref[S2]]
        for sk, sr in zip(k_hosts, ref_hosts):
            n = sk.n_steps
            for comp in ("displacement_x", "displacement_y"):
                a = r_k.read_nodes(sk.id, comp)
                b = r_ref.read_nodes(sr.id, comp, time_slice=slice(0, n))
                np.testing.assert_array_equal(a.values, b.values)
                np.testing.assert_array_equal(a.time, b.time)
            np.testing.assert_array_equal(
                r_k.read_elements(sk.id, "force").values,
                r_ref.read_elements(sr.id, "force", time_slice=slice(0, n)).values)
    finally:
        r_k.close()
        r_ref.close()
    assert sha(killed) == before
