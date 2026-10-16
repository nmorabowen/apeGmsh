"""Write the ``.ladruno`` fixtures of ``reader/ladruno.ts`` (ADR 0122 slice 4, #2126).

    python apeGmshViewer/fixtures/make_ladruno_fixtures.py          # the h5py ones
    python apeGmshViewer/fixtures/make_ladruno_fixtures.py --modes  # + the fork run

run from the repository root. The first form needs h5py and numpy only; the
second also needs a Ladruno fork build of OpenSees behind ``openseespy`` (the
banner lists "Ladruno — modular HDF5 .ladruno recorder"). It writes, beside
this file:

* ``ladruno_truss2d.ladruno``: a byte copy of the fork-written
  ``tests/fixtures/ladruno/truss2d.ladruno`` (fork build ``605affeb``,
  FORMAT_VERSION 1), unstamped: ``INFO/PROVENANCE`` is the empty group the
  recorder creates. Two trusses (EA = 1000, L = 1) in a line, a load of 10 at
  node 3 under LoadControl 0.25 for 4 steps, so
  ``Ux(node j, t) = 10 (j - 1) t / 1000`` and ``Uy = 0``. It records
  ``DISPLACEMENT`` and an element ``basicForce`` (which the reader lists as
  unread).
* ``ladruno_truss2d.stamped.ladruno``: the same file with the six lowercase
  ``INFO/PROVENANCE`` attributes of the fork schema §7b.1 written by hand,
  each a scalar variable-length UTF-8 string (``STAMP`` below).
* ``ladruno_truss2d.badstamp.ladruno``: the same file with a stamp whose
  values are not all the §7b.1 encoding (``BAD_STAMP`` below): a fixed-length
  ``model_hash``, a variable-length ASCII ``fem_hash``, a ``stage_names`` that
  is not a JSON array of strings, an integer ``unsupported``, and a
  ``writer`` that is variable-length UTF-8 but a shape-(1,) array, not a
  scalar. The reader reports each one, and pairs nothing.
* ``ladruno_modes.ladruno`` (``--modes``): a fork run. Two trusses (EA = 1000,
  L = 1) along x, node 1 fixed, y and the rotation fixed everywhere, a mass
  of 1 in x at nodes 2 and 3; ``eigen -fullGenLapack 2`` then ``record``, with
  ``-N displacement modesOfVibration modesOfVibrationRotational``. The two
  free DOFs give the closed form ``lambda = (k / m) (3 -/+ sqrt 5) / 2`` with
  ``k = 1000``, ``m = 1``, and mode shapes ``Ux3 / Ux2 = 2 - lambda / k``
  (the golden ratio for mode 1). The fork promotes the stage to
  ``KIND=eigen``; it holds one recorded step of ``DISPLACEMENT`` and the two
  modes under ``MODES_OF_VIBRATION(U)`` and ``(R)`` (the rotation column is
  fixed, so it is all zeros). Committed from fork build ``79e0623670``, the
  build the opensees venv imported on 2026-10-16. It predates the repository's
  ``FORK_PIN`` (``c744d6e70``) and writes no ``INFO/FORMAT_MINOR_VERSION``, so
  the file is FORMAT_MINOR_VERSION 0 (schema §2a: its ``KIND`` is the
  recorder-wide label, here the eigen promotion). Regenerate it on a pin
  build to get minor 1; the closed forms do not change.

No apeGmsh import: the app never runs apeGmsh (ADR 0112 D4), and these files
are what the fork writes, not what apeGmsh would make of them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "tests" / "fixtures" / "ladruno" / "truss2d.ladruno"
VLEN_UTF8 = h5py.string_dtype(encoding="utf-8")

#: The hand-written §7b.1 stamp: the six lowercase attributes, as strings.
STAMP: dict[str, str] = {
    "model_hash": "0123456789abcdef0123456789abcdef",
    "fem_hash": "fedcba9876543210fedcba9876543210",
    "stage_names": json.dumps(["push"], separators=(",", ":")),
    "element_index_space": "fem",
    "unsupported": "[]",
    "writer": "apeGmsh fixture (make_ladruno_fixtures.py)",
}


def truss2d() -> None:
    shutil.copyfile(SOURCE, HERE / "ladruno_truss2d.ladruno")


def stamped() -> None:
    out = HERE / "ladruno_truss2d.stamped.ladruno"
    shutil.copyfile(SOURCE, out)
    with h5py.File(out, "r+") as f:
        prov = f["INFO/PROVENANCE"]
        for name, value in STAMP.items():
            prov.attrs.create(name, data=value, dtype=VLEN_UTF8)


def badstamp() -> None:
    out = HERE / "ladruno_truss2d.badstamp.ladruno"
    shutil.copyfile(SOURCE, out)
    with h5py.File(out, "r+") as f:
        prov = f["INFO/PROVENANCE"]
        # Fixed-length bytes: the value reads as the right text, the encoding is wrong.
        prov.attrs.create("model_hash", data=np.bytes_(STAMP["model_hash"]))
        prov.attrs.create("fem_hash", data=STAMP["fem_hash"], dtype=h5py.string_dtype(encoding="ascii"))
        prov.attrs.create("stage_names", data='{"push": 1}', dtype=VLEN_UTF8)
        prov.attrs.create("unsupported", data=np.int32(0))
        prov.attrs.create("element_index_space", data="fem", dtype=VLEN_UTF8)
        # The right encoding at the wrong shape: a (1,) array, not a scalar.
        prov.attrs.create("writer", data=[STAMP["writer"]], dtype=VLEN_UTF8)


_MODES_RUN = r"""
import sys
import openseespy.opensees as ops

path = sys.argv[1]
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 3)
for i in range(3):
    ops.node(i + 1, float(i), 0.0)
ops.fix(1, 1, 1, 1)
ops.fix(2, 0, 1, 1)
ops.fix(3, 0, 1, 1)
ops.uniaxialMaterial("Elastic", 1, 1000.0)
ops.element("truss", 1, 1, 2, 1.0, 1)
ops.element("truss", 2, 2, 3, 1.0, 1)
ops.mass(2, 1.0, 0.0, 0.0)
ops.mass(3, 1.0, 0.0, 0.0)
ops.recorder("ladruno", path, "-N", "displacement", "modesOfVibration", "modesOfVibrationRotational")
print("lambda", ops.eigen("-fullGenLapack", 2))
ops.record()
ops.wipe()
"""


def modes() -> None:
    out = HERE / "ladruno_modes.ladruno"
    if out.exists():
        out.unlink()
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    subprocess.run([sys.executable, "-c", _MODES_RUN, str(out)], check=True, env=env)
    with h5py.File(out, "r") as f:
        on = f["MODEL_STAGE[1]/RESULTS/ON_NODES"]
        assert {"DISPLACEMENT", "MODES_OF_VIBRATION(U)", "MODES_OF_VIBRATION(R)"} <= set(on), list(on)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", action="store_true", help="also run the fork to write ladruno_modes.ladruno")
    args = ap.parse_args()
    truss2d()
    stamped()
    badstamp()
    if args.modes:
        modes()


if __name__ == "__main__":
    main()
