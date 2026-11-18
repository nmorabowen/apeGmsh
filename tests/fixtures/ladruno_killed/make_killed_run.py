"""Make the killed-run fixture of M2-S8 (#3003) with the fork's release build.

The fork's recorder (``recorder ladruno``, OpenSees#1084) writes under HDF5
SWMR, so a run killed while it records leaves a file the reader rule opens
(fork ``Ladruno_implementation/ladruno_schema_v1.md`` §7b.2). This script runs
a small deck with the build ``--opensees-bin`` names (a ``dist/bin`` holding
``opensees.pyd`` / ``opensees*.so``), kills it with ``Popen.kill``
(TerminateProcess on Windows, SIGKILL on POSIX) while it is recording, and
writes ``killed_swmr.ladruno`` here. It refuses a build other than the one
``FORK_PIN`` names.

The deck (:data:`CHILD`): a 2-D cantilever of four ``elasticBeamColumn``,
lumped masses, a sine tip load, Newmark, one step per ``analyze(1)``;
``recorder ladruno -N displacement -N modesOfVibration -E force -flush 0``.

- ``MODEL_STAGE[1]``: steps 1..20, with ``eigen 2`` after step 10 (one modes
  snapshot, ``DATA/STEP_<k>/MODE_<j>``); ``domainChange`` after step 20 ends it.
- ``MODEL_STAGE[2]``: no ``eigen``, so its ``MODES_OF_VIBRATION(U|R)`` groups
  are the empty ones OpenSees#1084 writes on every stage (no ``STEP_<k>``); the
  deck runs until killed, after it has printed step :data:`KILL_AFTER`.

Run from the repository root (any Python with h5py; the child imports the
build's ``opensees`` module)::

    python tests/fixtures/ladruno_killed/make_killed_run.py --opensees-bin <dist/bin>

The kill lands wherever the child is, so a rebuild is not byte-identical: the
README records what the committed file holds.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

#: Steps of the first stage, and the step after which it takes a modes snapshot.
STAGE1_STEPS = 20
EIGEN_AT = 10
#: The child is killed once it has printed this (global) step.
KILL_AFTER = 35

CHILD = r"""
import os, sys, time
bin_dir, out, stage1, eigen_at, sleep = sys.argv[1:6]
stage1, eigen_at, sleep = int(stage1), int(eigen_at), float(sleep)
if bin_dir != "-":
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(bin_dir)
    sys.path.insert(0, bin_dir)
import opensees as ops
print("BUILD", ops.ladrunoBuild(), flush=True)
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 3)
N = 4
for i in range(N + 1):
    ops.node(i + 1, 0.0, 0.5 * i)
    if i:
        ops.mass(i + 1, 1.0, 1.0, 0.01)
ops.fix(1, 1, 1, 1)
ops.geomTransf("Linear", 1)
for i in range(N):
    ops.element("elasticBeamColumn", i + 1, i + 1, i + 2, 0.01, 2.0e8, 1.0e-5, 1)
ops.recorder("ladruno", out, "-N", "displacement", "-N", "modesOfVibration",
             "-E", "force", "-flush", 0.0)
ops.timeSeries("Sine", 1, 0.0, 1.0e9, 0.37)
ops.pattern("Plain", 1, 1)
ops.load(N + 1, 10.0, 0.0, 0.0)
ops.constraints("Plain"); ops.numberer("RCM"); ops.system("BandGen")
ops.test("NormDispIncr", 1.0e-12, 10); ops.algorithm("Linear")
ops.integrator("Newmark", 0.5, 0.25); ops.analysis("Transient")
k = 0
while True:
    k += 1
    if ops.analyze(1, 0.001) != 0:
        raise SystemExit("analyze failed")
    if k == eigen_at:
        ops.eigen(2)
    if k == stage1:
        ops.domainChange()
    print("STEP", k, flush=True)
    time.sleep(sleep)
"""


def pinned_sha() -> str:
    for line in (ROOT / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("FORK_PIN has no sha= line")


def run_and_kill(bin_dir: str, out: Path, *, kill_after: int = KILL_AFTER,
                 sleep: float = 0.02, timeout: float = 120.0,
                 python: str = sys.executable) -> str:
    """Run :data:`CHILD` recording to ``out``; kill it after it has printed
    step ``kill_after``. Returns the build the child reported."""
    with tempfile.TemporaryDirectory(prefix="apegmsh-killed-run-") as tmp:
        script = Path(tmp) / "child.py"
        script.write_text(CHILD, encoding="utf-8")
        env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
        env.pop("HDF5_USE_FILE_LOCKING", None)
        proc = subprocess.Popen(
            [python, str(script), bin_dir, str(out), str(STAGE1_STEPS),
             str(EIGEN_AT), str(sleep)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        build, seen = "", []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                seen.append(line.rstrip())
                if line.startswith("BUILD "):
                    build = line.split()[1]
                if line.strip() == f"STEP {kill_after}":
                    break
            else:
                raise RuntimeError("the child ended before step "
                                   f"{kill_after}: {seen[-10:]}")
        finally:
            proc.kill()
            proc.wait(timeout=timeout)
        return build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--opensees-bin", required=True,
                    help="the fork build's dist/bin (holds the opensees module)")
    ap.add_argument("--out", default=str(HERE / "killed_swmr.ladruno"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        out.unlink()
    build = run_and_kill(a.opensees_bin, out)
    if build != pinned_sha():
        out.unlink(missing_ok=True)
        raise SystemExit(f"the build is {build}, FORK_PIN names {pinned_sha()}")
    with open(out, "rb") as fh:
        flags = fh.read(12)[11]
    print(f"wrote {out} ({out.stat().st_size} bytes), build {build}, "
          f"superblock flags 0x{flags:02x}")


if __name__ == "__main__":
    main()
