"""#1625 numeric oracle: RUN the serial and the partitioned deck, compare eigen.

The model of ``tests/opensees/integration/test_emit_partitioned_mp_penalty.py``:
a shell plate on ground springs whose every node is a slave of one
``rigidDiaphragm`` to a decoupled centre master, Gmsh-partitioned into 2
and 4 ranks. The serial deck runs on ``OpenSees.exe``; the partitioned one
on ``mpiexec -n K OpenSeesMP.exe``. The gates: the 6 lowest eigenvalues
to 1e-8 relative, and, for a static load on the master, every node's
displacement to 1e-8 of the peak.

* **Transformation**, the refusal's first remedy: the partitioned run must
  match the serial one, eigen and static.
* **Serial**, its second remedy: the partitioned FEM emitted with
  ``flat=True`` under Penalty runs on ``OpenSees.exe`` and must match the
  serial one.
* **Penalty**: on main the emit wrote the diaphragm on every rank, each
  copy assembled its own penalty, and the run drifted (measured: 4.85e-5
  at 2 ranks, 7.28e-5 at 4). The test accepts exactly two outcomes: the
  emit refuses (``BridgeError`` naming #1625), or the run matches the
  serial one to 1e-8. A deck that runs and drifts fails it.

Requires ``APEGMSH_OPENSEES_BIN`` (a dist ``bin`` with ``OpenSees.exe`` and
``OpenSeesMP.exe``) and Intel MPI; without them the test cannot run and
says so. No CI lane provides MPI (the ``subprocess`` marker is outside
every required lane), so this runs on a workstation.
"""
from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from tests.opensees.integration.test_emit_partitioned_mp_penalty import (
    diaphragm_ops,
)

EIG_TOL = 1.0e-8
DISP_TOL = 1.0e-8
RUN_TIMEOUT_S = 300


def _dist_bin() -> "Path | None":
    d = os.environ.get("APEGMSH_OPENSEES_BIN")
    if not d:
        return None
    p = Path(d)
    if (p / "OpenSees.exe").is_file() and (p / "OpenSeesMP.exe").is_file():
        return p
    return None


def _mpi_root() -> Path:
    oneapi = os.environ.get(
        "ONEAPI_ROOT", r"C:\Program Files (x86)\Intel\oneAPI")
    return Path(os.environ.get("I_MPI_ROOT") or Path(oneapi) / "mpi" / "latest")


def _mpiexec() -> "Path | None":
    c = _mpi_root() / "bin" / "mpiexec.exe"
    return c if c.is_file() else None


pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.slow,
    pytest.mark.skipif(
        _dist_bin() is None,
        reason="APEGMSH_OPENSEES_BIN unset or lacks OpenSees.exe + "
               "OpenSeesMP.exe"),
    pytest.mark.skipif(
        _mpiexec() is None,
        reason="Intel MPI mpiexec.exe not found (set I_MPI_ROOT)"),
    pytest.mark.filterwarnings("ignore:partitioned run"),
]

DRIVER = """
# ---- #1625 parity driver (appended) ----
set _pid 0
catch {{set _pid [getPID]}}
set _lam [eigen {eig} 6]
if {{$_pid == 0}} {{
  set _f [open eig.txt w]
  puts $_f $_lam
  close $_f
}}
puts "B5_DONE $_pid"
"""

STATIC = """
# ---- #1625 static driver (appended) ----
set _pid 0
catch {{set _pid [getPID]}}
timeSeries Constant 991
pattern Plain 991 991 {{
  if {{[lsearch [getNodeTags] {master}] >= 0 && $_pid == 0}} {{
    load {master} 1.0e6 5.0e5 0.0 0.0 0.0 2.0e6
  }}
}}
wipeAnalysis
constraints {handler}
numberer {numberer}
system {system}
test NormDispIncr 1.0e-12 10
algorithm Linear
integrator LoadControl 1.0
analysis Static
set _ok [analyze 1]
set _f [open disp_$_pid.txt w]
foreach _n [getNodeTags] {{ puts $_f "$_n [nodeDisp $_n]" }}
close $_f
puts "B5_DONE $_pid"
"""


def _emit(ops, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(path))
    return path


def _exec(model_tcl: Path, ranks: "int | None", driver: str) -> Path:
    """Append ``driver`` to the deck, run it serial (``ranks=None``) or on
    ``ranks``; return the run directory."""
    dist, mpi = _dist_bin(), _mpiexec()
    assert dist is not None and mpi is not None
    deck = model_tcl.with_name("run_" + model_tcl.name)
    deck.write_text(model_tcl.read_text() + driver)
    env = dict(os.environ)
    root = _mpi_root()
    dirs = [dist, root / "bin", root / "opt" / "mpi" / "libfabric" / "bin"]
    env["PATH"] = os.pathsep.join(
        [str(d) for d in dirs if d.is_dir()] + [env.get("PATH", "")])
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    cmd = ([str(dist / "OpenSees.exe"), deck.name] if ranks is None else
           [str(mpi), "-n", str(ranks), str(dist / "OpenSeesMP.exe"), deck.name])
    r = subprocess.run(cmd, cwd=deck.parent, env=env, capture_output=True,
                       text=True, timeout=RUN_TIMEOUT_S, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("B5_DONE") == (ranks or 1), (
        f"{cmd} exited {r.returncode}:\n{out[-3000:]}")
    return deck.parent


def _run(model_tcl: Path, ranks: "int | None") -> np.ndarray:
    d = _exec(model_tcl, ranks, DRIVER.format(
        eig="-genBandArpack" if ranks is None else ""))
    lam = np.array([float(v) for v in (d / "eig.txt").read_text().split()])
    assert lam.size == 6, lam
    return lam


def _master(text: str) -> int:
    """The diaphragm master, read off the deck's ``rigidDiaphragm`` line."""
    line = next(ln for ln in text.splitlines()
                if ln.strip().startswith("rigidDiaphragm "))
    return int(line.split()[2])


def _static(model_tcl: Path, ranks: "int | None", handler: str) -> "dict[int, np.ndarray]":
    d = _exec(model_tcl, ranks, STATIC.format(
        master=_master(model_tcl.read_text()), handler=handler,
        numberer="RCM" if ranks is None else "ParallelPlain",
        system="UmfPack" if ranks is None else "Mumps"))
    disp: dict[int, np.ndarray] = {}
    for f in sorted(d.glob("disp_*.txt")):
        for line in f.read_text().splitlines():
            w = line.split()
            n, v = int(w[0]), np.array([float(x) for x in w[1:]])
            if n in disp:
                np.testing.assert_array_equal(disp[n], v, err_msg=f"node {n} across ranks")
            disp[n] = v
    return disp


def _assert_static(serial: "dict[int, np.ndarray]", mp: "dict[int, np.ndarray]") -> None:
    assert set(mp) == set(serial)
    peak = max(float(np.abs(v).max()) for v in serial.values())
    assert peak > 0.0
    worst = max(float(np.abs(mp[n] - v).max()) for n, v in serial.items())
    assert worst <= DISP_TOL * peak, (worst, peak)


@pytest.fixture(scope="module")
def serial(tmp_path_factory) -> "dict[str, np.ndarray]":
    out = {}
    for name in ("Transformation", "Penalty"):
        d = tmp_path_factory.mktemp(f"serial_{name}")
        out[name] = _run(_emit(diaphragm_ops(1, name), d / "model.tcl"), None)
    return out


def _rel(mp: np.ndarray, ser: np.ndarray) -> float:
    return float((np.abs(mp - ser) / np.abs(ser)).max())


@pytest.mark.parametrize("ranks", [2, 4])
def test_transformation_ranks_match_serial(serial, tmp_path: Path, ranks: int):
    deck = _emit(diaphragm_ops(ranks, "Transformation"), tmp_path / "model.tcl")
    rel = _rel(_run(deck, ranks), serial["Transformation"])
    assert rel < EIG_TOL, rel


@pytest.mark.parametrize("ranks", [2, 4])
def test_transformation_ranks_match_serial_static(tmp_path: Path, ranks: int):
    ser = _static(_emit(diaphragm_ops(1, "Transformation"),
                        tmp_path / "serial" / "model.tcl"), None, "Transformation")
    mp = _static(_emit(diaphragm_ops(ranks, "Transformation"),
                       tmp_path / "mp" / "model.tcl"), ranks, "Transformation")
    _assert_static(ser, mp)


@pytest.mark.parametrize("ranks", [2, 4])
def test_the_serial_remedy_matches_serial(serial, tmp_path: Path, ranks: int):
    """The partitioned FEM under Penalty, emitted ``flat=True`` and run
    serially, is the serial model: eigen and static."""
    path = tmp_path / "model.tcl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diaphragm_ops(ranks, "Penalty", serial_chain=True).tcl(
            str(path), flat=True)
    assert _rel(_run(path, None), serial["Penalty"]) < EIG_TOL
    ser = _static(_emit(diaphragm_ops(1, "Penalty"),
                        tmp_path / "serial" / "model.tcl"), None, "Penalty 1e12 1e12")
    _assert_static(ser, _static(path, None, "Penalty 1e12 1e12"))


@pytest.mark.parametrize("ranks", [2, 4])
def test_penalty_ranks_refuse_or_match_serial(serial, tmp_path: Path, ranks: int):
    try:
        deck = _emit(diaphragm_ops(ranks, "Penalty"), tmp_path / "model.tcl")
    except BridgeError as err:
        assert "#1625" in str(err)
        assert not (tmp_path / "model.tcl").exists()
        return
    rel = _rel(_run(deck, ranks), serial["Penalty"])
    assert rel < EIG_TOL, (
        f"partitioned Penalty deck ran and drifted from serial by {rel:.3e} "
        f"on {ranks} ranks (#1625)")
