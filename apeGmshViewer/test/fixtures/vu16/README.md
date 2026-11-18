# VU16 fixture: footfall's state-free user result, written by W3's own writer

Every `.user.h5` here was written by `FootfallResult.to_results` (W3, #2782 / PR #2799),
which calls `apeGmsh.results.user.write_user_file`. Each `*.golden.json` is what Python's
`UserFile(path).read_result("footfall")` reads back: the ids, the components, and `DATA[0]`
row by row, with `null` for NaN. `test/state/userResult.test.ts` and
`test/main/userDoors.test.ts` read them (slice VU16, #2800).

| File | What it is |
|---|---|
| `w3_bay.h5`, `w3_bay.user.h5`, `w3_bay.golden.json` | W3's own unit fixture (`tests/opensees/unit/test_footfall_user_h5.py`): five FEM ids, response nodes 40 and 10, an `a_p` of 1e-300, NaN elsewhere. The archive holds only `/meta/lineage`, as W3's test writes it. |
| `w3_other.h5` | the archive of another fem (same ids, salted coordinates): `w3_bay.user.h5` is stale against it (A6, `fem_hash`). |
| `plate.h5`, `plate.user.h5`, `plate.golden.json` | a meshed 12 m x 6 m plate (apeGmsh + gmsh, no OpenSees) with a **synthetic** footfall map on the nodes with x <= 9 m and NaN on the rest, so a still shows the ramp and the no-data grey. Only the layout and the values are under test, never the engineering. |

The script below is the whole generator. It imports apeGmsh, so it is kept here as text and
never as a `.py` file of the app (ADR 0112 D4, `scripts/check_app_wall.py` W1). Save it as
`make_vu16.py` outside the app and run it with the opensees venv interpreter:

    python make_vu16.py <this directory> <checkout>/src

```python
"""Write the VU16 fixtures: footfall's state-free user result, by W3's own writer.

    PYTHONPATH=<worktree>/src python make_vu16.py <fixture dir>

Every ``.user.h5`` here is written by ``FootfallResult.to_results`` (W3, #2799),
which calls ``apeGmsh.results.user.write_user_file``; each golden is what
Python's ``UserFile(path).read_result("footfall")`` reads back.

- ``w3_bay.h5`` / ``w3_bay.user.h5`` / ``w3_bay.golden.json``: W3's own unit
  fixture (tests/opensees/unit/test_footfall_user_h5.py): five FEM ids, two
  response nodes, one a_p of 1e-300, NaN elsewhere. The archive holds only
  ``/meta/lineage``, as W3's test writes it.
- ``w3_other.h5``: the archive of another fem (same ids, salted coordinates),
  so the same user file is stale against it (A6, fem_hash).
- ``plate.h5`` / ``plate.user.h5`` / ``plate.golden.json``: a meshed 12 m x 6 m
  plate (apeGmsh + gmsh, no OpenSees), with a SYNTHETIC footfall map on the
  nodes with x <= 9 m and NaN on the rest, so a still shows both the ramp and
  the no-data grey. Only the layout and the values are under test, never the
  engineering.
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

if len(sys.argv) > 2:  # <worktree>/src, in place of PYTHONPATH
    sys.path.insert(0, sys.argv[2])

import h5py
import numpy as np

from apeGmsh.opensees._target import BackendInfo
from apeGmsh.opensees.analysis.footfall_result import FootfallResult
from apeGmsh.opensees.emitter import live
from apeGmsh.results.user import UserFile

# W3 stamps the backend the live modal basis ran on; no OpenSees runs here.
live.get_backend_info = lambda: BackendInfo(kind="stock", build=None, version=None, source="vu16 fixture")

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


class MockFem:
    """W3's ``tests.conftest._MockFem``: node ids, zero coordinates (``salt`` offsets them), the real snapshot id."""

    def __init__(self, ids, salt=0):
        ids = np.asarray(ids, dtype=np.int64)
        self.nodes = SimpleNamespace(ids=ids, coords=np.zeros((ids.size, 3)) + float(salt) * 1e-9)
        self.elements = []

    @property
    def snapshot_id(self):
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)


def result(nodes, a_p, ratio, f_dom):
    res = object.__new__(FootfallResult)
    for k, v in {"nodes": tuple(int(n) for n in nodes), "a_p": np.asarray(a_p, float),
                 "ratio": np.asarray(ratio, float), "f_dom": np.asarray(f_dom, float)}.items():
        object.__setattr__(res, k, v)
    return res


def archive(path, fem, model_hash=None):
    with h5py.File(path, "w") as f:
        lineage = f.create_group("meta/lineage")
        lineage.attrs["fem_hash"] = fem.snapshot_id
        if model_hash is not None:
            lineage.attrs["model_hash"] = model_hash
    return path


def golden(user_h5, path):
    back = UserFile(user_h5).read_result("footfall")
    data = np.asarray(back.data)
    out = {
        "name": "footfall",
        "components": list(back.components),
        "ids": [int(i) for i in np.asarray(back.ids).ravel()],
        "shape": list(data.shape),
        # DATA[0], row by row; null is NaN (no value at that node)
        "data": [[None if math.isnan(v) else float(v) for v in row] for row in data[0]],
    }
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


# -- W3's own unit fixture ------------------------------------------------------
FEM_IDS = [10, 20, 30, 40, 50]
fem = MockFem(FEM_IDS)
archive(OUT / "w3_bay.h5", fem, model_hash="ab" * 16)
user = result((40, 10), [0.0123456789, 1e-300], [1.1772, 0.25], [16.326, 4.0]).to_results(fem, OUT / "w3_bay.user.h5", model_h5=OUT / "w3_bay.h5")
golden(user, OUT / "w3_bay.golden.json")
archive(OUT / "w3_other.h5", MockFem(FEM_IDS, salt=1), model_hash="ab" * 16)

# -- a meshed plate, for the app's draw path and its stills -----------------------
from apeGmsh import apeGmsh  # noqa: E402

with apeGmsh(model_name="vu16-plate") as g:
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 12.0, 6.0)
    g.model.sync()
    import gmsh  # noqa: E402
    g.physical.add(2, [t for _, t in gmsh.model.getEntities(2)], name="Slab")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.recipe.structured(size=1.0, dim=2, generate=True)
    plate = g.mesh.queries.get_fem_data()
plate.to_h5(str(OUT / "plate.h5"))
ids = np.asarray(plate.nodes.ids, dtype=np.int64)
xyz = np.asarray(plate.nodes.coords, dtype=float)
on = xyz[:, 0] <= 9.0 + 1e-9
x, y = xyz[on, 0], xyz[on, 1]
ratio = 1.4 * np.abs(np.sin(np.pi * x / 6.0) * np.sin(np.pi * y / 6.0)) + 0.05
user = result(ids[on], 0.005 * ratio, ratio, 5.0 + x / 4.0).to_results(plate, OUT / "plate.user.h5", model_h5=OUT / "plate.h5")
golden(user, OUT / "plate.golden.json")
print("written:", sorted(p.name for p in OUT.iterdir()))
```
