"""Write the VU7b-1 fixture: a SYNTHETIC nodal spectrum case on San Ramon (ADR 0130 D5).

    PYTHONPATH=<worktree>/src python make_sanramon_case.py <sanramon set dir> <full out dir>

<sanramon set dir> holds ``sanramon_3A.h5`` and ``sanramon_3A.ladruno`` (the reference
set; not in the repository). Two cases are written with apeGmsh's own writer
(``write_user_file``), each with its golden from Python's
``Results.combinations().evaluate(name, component, level="nodes")``:

- ``<full out dir>/sanramon_3A.user.h5`` and ``golden.json``: all 13459 nodes (a probe
  outside the repository; copies of the set go beside it, the reference is never written);
- ``sanramon_case.user.h5`` and ``sanramon_case.golden.json`` beside this script: every
  ``STRIDE``-th node only, which test/reader/userSpectraCase.test.ts reads.

The case is synthetic. Its shapes and eigenvalues are the .ladruno's own
(``MODES_OF_VIBRATION(U)``, ``LAMBDA``); its participation factors are drawn from a
seeded generator, because ``modalProperties`` needs a live solve. Each increment is
``gamma_k * sa_k / lambda_k * phi_k``. Modes 1 and 2 share one ``sa_period > 0``, so a
whole case groups them; mode 5 has NaN damping, so every row over it names its own.
"""
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

import apeGmsh
from apeGmsh.results import Results
from apeGmsh.results.user import CaseQuantity, SpectrumCase, write_user_file
from apeGmsh.results.user._descriptor import Combination, Term

STRIDE = 337
CASE = "NEC-15 Ex (synthetic nodal)"
COMPONENTS = ("displacement_x", "displacement_y", "displacement_z")
HERE = Path(__file__).resolve().parent


def rows() -> list[Combination]:
    whole = (Term("spectrum", CASE, -1, 0, 1.0),)
    return [
        Combination("CQC Ex 2%", "cqc", whole, damping=0.02),
        Combination("CQC Ex", "cqc", (Term("spectrum", CASE, -1, 0, -1.5),), damping=0.05),
        Combination("SRSS Ex", "srss", whole),
        Combination("CQC Ex modes 3-6", "cqc",
                    tuple(Term("spectrum", CASE, -1, k, 1.0) for k in (3, 4, 6, 5)), damping=0.03),
    ]


def write(model: Path, run: Path, user_h5: Path, golden: Path, keep: np.ndarray) -> None:
    with h5py.File(run, "r") as f:
        g = f["MODEL_STAGE[1]/RESULTS/ON_NODES/MODES_OF_VIBRATION(U)"]
        ids = np.asarray(g["ID"][...], dtype="i4").ravel()[keep]
        steps = g["DATA/STEP_0"]
        m = len(steps)
        phi = np.stack([np.asarray(steps[f"MODE_{k}"][...], dtype="f8")[keep] for k in range(m)])
        lam = np.array([float(steps[f"MODE_{k}"].attrs["LAMBDA"][0]) for k in range(m)])
    rng = np.random.default_rng(2740)
    gamma = rng.uniform(5.0, 40.0, m) * np.where(rng.uniform(size=m) < 0.5, -1.0, 1.0)
    periods = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 4.0])
    sa_tab = np.array([0.4, 1.0, 1.0, 0.6, 0.3, 0.15])
    scale = 9.81
    own_t = 2 * np.pi / np.sqrt(lam)
    sa_period = np.zeros(m)
    sa_period[0] = sa_period[1] = own_t[0]
    sa = scale * np.interp(np.where(sa_period > 0, sa_period, own_t), periods, sa_tab)
    damping = np.full(m, 0.05)
    damping[4] = np.nan
    inc = (gamma * sa / lam)[:, None, None] * phi
    labels = ("Ux", "Uy", "Uz")
    case = SpectrumCase(
        name=CASE, direction=1, scale=scale, spectrum_periods=periods, spectrum_sa=sa_tab,
        mode=np.arange(1, m + 1, dtype="i4"), kind=("eigen",) * m, lam=lam,
        sa_period=sa_period, gamma=gamma, sa=sa, damping=damping,
        results=(CaseQuantity("DISPLACEMENT", "nodes", labels, inc, ids),),
        reference=(CaseQuantity("DISPLACEMENT", "nodes", labels, np.zeros((1, ids.size, 3)), ids),),
    )
    write_user_file(user_h5, model, results_file=run, spectra=[case], combinations=rows(),
                    writer="VU7b-1 fixture (synthetic nodal case)")
    uc = Results.from_ladruno(run, model_h5=model).combinations(user_h5)
    out = {"about": "Python UserCombinations.evaluate(name, component, level='nodes') on a SYNTHETIC case",
           "case": CASE, "nodes": int(ids.size), "modes": m, "combinations": {}}
    for r in rows():
        out["combinations"][r.name] = {}
        for comp in COMPONENTS:
            c = uc.evaluate(r.name, comp, level="nodes")
            out["combinations"][r.name][comp] = {
                "ids": [int(x) for x in c.field.keys["id"]],
                "values": [float(v) for v in np.asarray(c.field.values).ravel()],
                "signed": bool(c.signed),
            }
    golden.write_text(json.dumps(out, indent=0) + "\n")


def main() -> None:
    print("apeGmsh from", apeGmsh.__file__, file=sys.stderr)
    ref, full = Path(sys.argv[1]), Path(sys.argv[2])
    full.mkdir(parents=True, exist_ok=True)
    for name in ("sanramon_3A.h5", "sanramon_3A.ladruno"):
        if not (full / name).exists():
            shutil.copy2(ref / name, full / name)
    model, run = full / "sanramon_3A.h5", full / "sanramon_3A.ladruno"
    with h5py.File(run, "r") as f:
        n = f["MODEL_STAGE[1]/RESULTS/ON_NODES/MODES_OF_VIBRATION(U)/ID"].shape[0]
    write(model, run, full / "sanramon_3A.user.h5", full / "golden.json", np.arange(n))
    write(model, run, HERE / "sanramon_case.user.h5", HERE / "sanramon_case.golden.json",
          np.arange(0, n, STRIDE))


if __name__ == "__main__":
    main()
