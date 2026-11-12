# VU19 fixture: a `.user.h5` for each variant of the VU10 study, written by Python's own writer

Every `.user.h5` here was written by `apeGmsh.results.user.write_user_file` against its own
variant's model in `../vu10/` (and, for the cracked one, its own results file). `vu19.golden.json`
is what Python's `staleness` (A6) and `UserFile.read_result` say of them.
`test/main/compareDoors.test.ts` and `test/variants/compareUser.test.ts` copy the VU10 study into
a temporary folder and put these files beside its models (slice VU19, #2861).

| File | Goes beside | What it is |
|---|---|---|
| `plate_cracked.user.h5` | `cracked/plate_cracked.h5` (the compare set) | an empty `/combinations` zone; `footfall` (nodes, state-free, basis model) and `drift` (nodes, state-free, basis results). Current against its own model and results; against the uncracked ones (same `fem_hash`, another `model_hash`, another results file) both are stale |
| `plate_uncracked.user.h5` | `plate_uncracked.h5` (the window set) | one `footfall` entry with other values: a case whose name matches the compare set's |
| `vu19.golden.json` | | Python's verdicts on the compare file against its own set and against the window set, and both `footfall` columns |

Only the layout and the values are under test, never the engineering: the values are a
synthetic ramp over the node coordinates.

The script below is the whole generator. It imports apeGmsh, so it is kept here as text and
never as a `.py` file of the app (ADR 0112 D4, `scripts/check_app_wall.py` W1). Save it as
`make_vu19.py` outside the app and run it with the opensees venv interpreter and
`PYTHONPATH=<checkout>/src`:

    python make_vu19.py <app>/test/fixtures/vu10 <app>/test/fixtures/vu19

```python
"""Write the VU19 fixtures: a .user.h5 for each variant of the VU10 study.

    PYTHONPATH=<worktree>/src python make_vu19.py <vu10 dir> <vu19 dir>

Each ``.user.h5`` is written by ``apeGmsh.results.user.write_user_file`` against
its own variant's model (and, for the cracked one, its own results file). The
golden is what Python's ``staleness`` (A6) and ``UserFile.read_result`` give:

- ``plate_cracked.user.h5`` (the compare set's, beside ``cracked/plate_cracked.h5``):
  an empty ``/combinations`` zone (a generation with no row); ``footfall``,
  state-free on nodes, ``@basis="model"``; ``drift``, state-free on nodes,
  ``@basis="results"``. Judged against its own model and results, both are
  current; against the uncracked model (same fem_hash, another model_hash)
  ``footfall`` is stale, and against the uncracked results ``drift`` is stale.
- ``plate_uncracked.user.h5`` (the window set's): one ``footfall`` entry with
  other values: a case whose name matches the compare set's.

Only the layout and the values are under test, never the engineering: the
values are a synthetic ramp over the node coordinates.
"""
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

from apeGmsh.results.user import UserFile, UserResult, staleness, write_user_file

VU10 = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

CRACKED = VU10 / "cracked" / "plate_cracked.h5"
CRACKED_RES = VU10 / "cracked" / "plate_cracked.ladruno"
UNCRACKED = VU10 / "plate_uncracked.h5"
UNCRACKED_RES = VU10 / "plate_uncracked.ladruno"


def nodes(model):
    with h5py.File(model, "r") as f:
        return np.asarray(f["nodes/ids"][()], dtype=np.int64), np.asarray(f["nodes/coords"][()], dtype=float)


def column(ids, values):
    return np.asarray(values, dtype=np.float64).reshape(1, ids.size, 1)


def write(path, model, results_file, entries, combinations):
    # The writer refuses a file that exists; start clean.
    if path.exists():
        path.unlink()
    return write_user_file(path, model, results_file=results_file, results=entries, combinations=combinations, writer="vu19 fixture")


ids, xyz = nodes(CRACKED)
x, y = xyz[:, 0], xyz[:, 1]
span = max(float(x.max() - x.min()), 1.0)
footfall_cracked = 0.2 + 1.2 * (x - x.min()) / span + 0.1 * y
drift_cracked = 0.001 * (1.0 + (x - x.min()) / span)

cracked_user = OUT / "plate_cracked.user.h5"
write(
    cracked_user,
    CRACKED,
    CRACKED_RES,
    (
        UserResult(name="footfall", target="nodes", basis="model", components=("ratio",), data=column(ids, footfall_cracked), ids=ids),
        UserResult(name="drift", target="nodes", basis="results", components=("d",), data=column(ids, drift_cracked), ids=ids),
    ),
    [],
)

ids_u, xyz_u = nodes(UNCRACKED)
footfall_uncracked = 0.1 + 0.4 * (xyz_u[:, 0] - xyz_u[:, 0].min()) / span
window_user = OUT / "plate_uncracked.user.h5"
write(
    window_user,
    UNCRACKED,
    None,
    (UserResult(name="footfall", target="nodes", basis="model", components=("ratio",), data=column(ids_u, footfall_uncracked), ids=ids_u),),
    None,
)


def verdicts(user, model, results):
    st = staleness(user, model_h5=model, results_file=results)
    return {
        "zones": {k: {"stale": v.stale, "reason": v.reason} for k, v in st.zones.items()},
        "results": {k: {"stale": v.stale, "reason": v.reason} for k, v in st.results.items()},
    }


def read(user, name):
    back = UserFile(user).read_result(name)
    data = np.asarray(back.data)
    return {
        "components": list(back.components),
        "ids": [int(i) for i in np.asarray(back.ids).ravel()],
        "data": [[None if math.isnan(v) else float(v) for v in row] for row in data[0]],
    }


golden = {
    "compare_user": "plate_cracked.user.h5",
    "window_user": "plate_uncracked.user.h5",
    # A6 of the compare set's file against its own model and results, and against the window set's.
    "compare_own": verdicts(cracked_user, CRACKED, CRACKED_RES),
    "compare_against_window": verdicts(cracked_user, UNCRACKED, UNCRACKED_RES),
    "window_own": verdicts(window_user, UNCRACKED, UNCRACKED_RES),
    "compare_footfall": read(cracked_user, "footfall"),
    "window_footfall": read(window_user, "footfall"),
    "compare_combinations": len(UserFile(cracked_user).combinations().rows) if hasattr(UserFile(cracked_user), "combinations") else None,
}
(OUT / "vu19.golden.json").write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
print("written:", sorted(p.name for p in OUT.iterdir()))
```
