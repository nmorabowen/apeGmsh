# VU9b fixtures: a User Combination the app writes into `<stem>.user.h5`

Slice VU9b (#2693, ADR 0130 A4). Every file here was written by apeGmsh's own
writers, never by hand.

- `plate.h5` is `test/fixtures/vu16/plate.h5`, copied (the plate model: 91 nodes).
- `plate.user.h5` is written by Python (`write_user_file`, the first script below): one
  **synthetic** `/spectra` case, `Ex (synthetic)` (4 modes, `DISPLACEMENT` `Ux,Uy,Uz` on every
  node; smooth mode shapes and seeded participation factors, never engineering), and
  `/combinations/g0` with three rows: `SRSS Ex` and `CQC Ex 5%` over the whole case, and
  `1.2D + 1.6L` over two stage terms.
- `app_written/plate.user.h5` is `plate.user.h5` after three saves by the app's writer,
  main/combinations.ts `saveCombination` (`test/main/combinationsDriver.ts`, run with
  `plate.h5` beside a copy): the default row `SRSS Ex modes 1-2`, `--preset=cqc`
  (`CQC Ex modes 1-3 3%`) and `--preset=linear` (`1.3 SRSS Ex`). It holds g0 to g3.
- `app_written.golden.json` is what Python reads of it (the second script below):
  `UserFile` (`/meta@revision`, `@writer`, `@current`, every `g<N>`, every row of the
  current generation) and Python's evaluator over `UserSpectra` for each row and
  component. `test/combination/crossRead.test.ts` holds the app to it.

Neither script is a `.py` file of the app (ADR 0112 D4, `scripts/check_app_wall.py` W1).
Save each outside the app and run it with the opensees venv interpreter and
`PYTHONPATH=<checkout>/src`:

```text
python make_vu9b_fixture.py apeGmshViewer/test/fixtures/vu16 apeGmshViewer/test/fixtures/vu9b
# copy plate.h5 and plate.user.h5 into <dir>, then, from apeGmshViewer/:
node test/main/combinationsDriver.ts --dir=<dir>
node test/main/combinationsDriver.ts --dir=<dir> --preset=cqc
node test/main/combinationsDriver.ts --dir=<dir> --preset=linear
python crossread.py <dir> <dir>/app_written.golden.json
# copy <dir>/plate.user.h5 to app_written/, and the golden beside this README
```

## `make_vu9b_fixture.py`

```python
"""Write the VU9b fixture: plate.h5 with a SYNTHETIC nodal spectrum case and three combinations.

    PYTHONPATH=<worktree>/src python make_vu9b_fixture.py <vu16 dir> <vu9b dir>

<vu16 dir> holds ``plate.h5`` (test/fixtures/vu16). It is copied into <vu9b dir>, and
``plate.user.h5`` is written beside it with apeGmsh's own writer (``write_user_file``):
one ``/spectra`` case and generation g0 of ``/combinations``.

The case is synthetic: four modes with smooth shapes over the plate's nodes, participation
factors from a seeded generator, and each increment ``gamma_k * sa_k / lambda_k * phi_k``.
Only the layout and the combined values are under test, never the engineering.
"""
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

import apeGmsh
from apeGmsh.results.user import CaseQuantity, SpectrumCase, write_user_file
from apeGmsh.results.user._descriptor import Combination, Term

CASE = "Ex (synthetic)"


def rows() -> list[Combination]:
    whole = (Term("spectrum", CASE, -1, 0, 1.0),)
    return [
        Combination("SRSS Ex", "srss", whole),
        Combination("CQC Ex 5%", "cqc", whole, damping=0.05),
        Combination("1.2D + 1.6L", "linear", (Term("stage", "D", -1, 0, 1.2), Term("stage", "L", -1, 0, 1.6))),
    ]


def main() -> None:
    print("apeGmsh from", apeGmsh.__file__, file=sys.stderr)
    src, here = Path(sys.argv[1]), Path(sys.argv[2])
    here.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "plate.h5", here / "plate.h5")
    model = here / "plate.h5"
    with h5py.File(model, "r") as f:
        ids = np.asarray(f["nodes/ids"][...], dtype="i4")
        xyz = np.asarray(f["nodes/coords"][...], dtype="f8")
    span = np.ptp(xyz, axis=0)
    span[span == 0] = 1.0
    u = (xyz - xyz.min(axis=0)) / span
    m = 4
    periods_own = np.array([0.8, 0.5, 0.3, 0.2])
    lam = (2 * np.pi / periods_own) ** 2
    rng = np.random.default_rng(2693)
    gamma = rng.uniform(5.0, 40.0, m) * np.where(rng.uniform(size=m) < 0.5, -1.0, 1.0)
    periods = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    sa_tab = np.array([0.4, 1.0, 1.0, 0.6, 0.3])
    scale = 9.81
    sa = scale * np.interp(periods_own, periods, sa_tab)
    phi = np.zeros((m, ids.size, 3))
    for k in range(m):
        phi[k, :, 0] = np.sin((k + 1) * np.pi * u[:, 0]) * np.cos(np.pi * u[:, 1] / (k + 1))
        phi[k, :, 1] = 0.5 * np.cos((k + 1) * np.pi * u[:, 1]) * u[:, 0]
        phi[k, :, 2] = 0.1 * np.sin(np.pi * u[:, 0]) * np.sin((k + 1) * np.pi * u[:, 1])
    inc = (gamma * sa / lam)[:, None, None] * phi
    labels = ("Ux", "Uy", "Uz")
    case = SpectrumCase(
        name=CASE, direction=1, scale=scale, spectrum_periods=periods, spectrum_sa=sa_tab,
        mode=np.arange(1, m + 1, dtype="i4"), kind=("eigen",) * m, lam=lam,
        sa_period=np.zeros(m), gamma=gamma, sa=sa, damping=np.full(m, 0.05),
        results=(CaseQuantity("DISPLACEMENT", "nodes", labels, inc, ids),),
        reference=(CaseQuantity("DISPLACEMENT", "nodes", labels, np.zeros((1, ids.size, 3)), ids),),
    )
    write_user_file(here / "plate.user.h5", model, spectra=[case], combinations=rows(),
                    writer="VU9b fixture (synthetic nodal case)")


if __name__ == "__main__":
    main()
```

## `crossread.py`

```python
"""Read an app-written plate.user.h5 with apeGmsh and evaluate every combination over /spectra.

    PYTHONPATH=<worktree>/src python crossread.py <dir> <golden.json>

Writes what Python's UserFile reads (meta, generations, every row) and what
Python's evaluator gives for each row and component through UserSpectra (the
/spectra cases; with no results file, a stage or mode term is refused).
"""
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

import apeGmsh
from apeGmsh.results.user import CombinationRefused, UserFile, UserSpectra, evaluate

COMPONENTS = ("displacement_x", "displacement_y", "displacement_z")


def num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def main() -> None:
    print("apeGmsh from", apeGmsh.__file__, file=sys.stderr)
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    user, model = d / "plate.user.h5", d / "plate.h5"
    uf = UserFile(user)
    table = uf.combinations(model_h5=model)
    with h5py.File(user, "r") as f:
        zone = f["combinations"]
        current = int(np.asarray(zone.attrs["current"]).ravel()[0])
        held = sorted(int(k[1:]) for k in zone if k.startswith("g"))
        writer = f["meta"].attrs["writer"]
    spectra = UserSpectra(user, model_h5=model)

    class Src:
        """The /spectra cases as the evaluator's Source, for one quantity (no Results: no stage term here)."""
        linear = True

        def __init__(self, q):
            self.q = q

        def stored(self, name):
            return None

        def field(self, address):
            return spectra.field(address, self.q)

        def eigenvalue(self, address):
            raise CombinationRefused("<eigen>", "unresolved", "no results file")

        def case(self, name):
            return spectra.case(name, self.q)

    class UC:
        @staticmethod
        def evaluate(name, comp, level):
            return evaluate(table, name, Src((level, comp)))

    uc = UC()
    res = {"about": "Python UserFile, and apeGmsh.results.user.evaluate over UserSpectra, on the app-written file",
           "revision": uf.revision, "writer": str(writer), "current": current, "generations": held,
           "rows": [{"name": r.name, "rule": r.rule, "envelope": r.envelope, "fraction": num(r.fraction),
                     "damping": num(r.damping),
                     "terms": [[t.kind, t.source, t.step, t.mode, t.factor] for t in r.terms]} for r in table.rows],
           "refused": [[r.name, r.reason] for r in table.refused], "values": {}}
    res["refusals"] = {}
    for r in table.rows:
        try:
            uc.evaluate(r.name, COMPONENTS[0], level="nodes")
        except CombinationRefused as exc:
            res["refusals"][r.name] = exc.reason
            continue
        res["values"][r.name] = {}
        for comp in COMPONENTS:
            c = uc.evaluate(r.name, comp, level="nodes")
            res["values"][r.name][comp] = {"ids": [int(x) for x in c.field.keys["id"]],
                                           "values": [float(v) for v in np.asarray(c.field.values).ravel()],
                                           "signed": bool(c.signed)}
    out.write_text(json.dumps(res, indent=0) + "\n")


if __name__ == "__main__":
    main()
```
