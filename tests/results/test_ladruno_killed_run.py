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
