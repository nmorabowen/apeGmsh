# VF3-c fixture: model.h5 files rebuilt from a `.ladruno`, every block file-coded

Each `.h5` here is the `model.h5` that `FEMData.to_h5` writes for the FEMData
`read_fem_from_ladruno` rebuilds from the last stage's `MODEL` group of one of
`tests/fixtures/simplex_gp/{tri,tet}.ladruno`. Such a FEMData keys every element
block by `code = -class_tag`, so each block here is file-coded.
`test/meshFileCoded.test.ts` reads them (slice VF3-c, #2961).

| File | Blocks (class, dim, npe) |
|---|---|
| `tri_model.h5` | FourNodeQuad (2, 4), Tri31 (2, 3), SixNodeTri (2, 6), BezierTri6 (2, 6), LadrunoLST (2, 6) |
| `tet_model.h5` | Brick (3, 8), FourNodeTetrahedron (3, 4), TenNodeTetrahedron (3, 10), BezierTet10 (3, 10) |
| `file_coded.golden.json` | each block's attributes and the gmsh code Python's `_GMSH_CODE_BY_PARENT_DIM_NPE` gives its `(dim, npe)`; and what Python's Qt viewer scene (`build_fem_scene` over `ViewerData.from_h5` of the same file) draws: its cells by VTK type, the codes it skips, and the triangles of its outer surface (`extract_surface().triangulate()`) |

The script below is the whole generator. It imports apeGmsh, so it is kept here as text and
never as a `.py` file of the app (ADR 0112 D4, `scripts/check_app_wall.py` W1). Save it as
`make_file_coded.py` outside the app and run it with the opensees venv interpreter and
`PYTHONPATH=<checkout>/src`:

    python make_file_coded.py <checkout> <app>/test/fixtures/file_coded

```python
"""Write the VF3-c fixtures: a model.h5 rebuilt from each simplex_gp .ladruno.

    PYTHONPATH=<worktree>/src python make_file_coded.py <checkout> <out dir>

Each ``.ladruno`` in ``tests/fixtures/simplex_gp`` holds one element of each of
several classes. ``read_fem_from_ladruno`` rebuilds a FEMData from the last
stage's ``MODEL`` group (every block file-coded, ``code = -class_tag``) and
``FEMData.to_h5`` writes it as a ``model.h5``. The golden is what Python's Qt
viewer scene (``build_fem_scene`` over ``ViewerData.from_h5`` of that same
file) draws: its cells by VTK type, and the triangles of its outer surface.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from apeGmsh.mesh._femdata_ladruno_io import read_fem_from_ladruno
from apeGmsh.results._gauss_extrapolation import _GMSH_CODE_BY_PARENT_DIM_NPE
from apeGmsh.viewers.data._viewer_data import ViewerData
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

VTK_NAME = {3: "line", 5: "tri", 9: "quad", 10: "tet", 12: "hex", 13: "prism", 14: "pyramid"}

checkout, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
golden = {}
for stem in ("tri", "tet"):
    src = checkout / "tests" / "fixtures" / "simplex_gp" / f"{stem}.ladruno"
    with h5py.File(src, "r") as f:
        stage = sorted(k for k in f if k.startswith("MODEL_STAGE"))[-1]
        fem = read_fem_from_ladruno(f[stage]["MODEL"])
    dst = out / f"{stem}_model.h5"
    fem.to_h5(str(dst), model_name=f"simplex_gp_{stem}")

    with h5py.File(dst, "r") as f:
        blocks = {
            k: {
                "code": int(f["elements"][k].attrs["code"]),
                "dim": int(f["elements"][k].attrs["dim"]),
                "npe": int(f["elements"][k].attrs["npe"]),
                "cells": int(f["elements"][k]["ids"].shape[0]),
                "gmshCode": _GMSH_CODE_BY_PARENT_DIM_NPE.get(
                    (int(f["elements"][k].attrs["dim"]), int(f["elements"][k].attrs["npe"]))
                ),
            }
            for k in f["elements"]
        }
    scene = build_fem_scene(ViewerData.from_h5(str(dst)))
    types = np.asarray(scene.grid.celltypes)
    surface = scene.grid.extract_surface().triangulate()
    golden[stem] = {
        "file": dst.name,
        "blocks": blocks,
        "pythonCells": {VTK_NAME[int(t)]: int((types == t).sum()) for t in np.unique(types)},
        "pythonSkipped": [int(c) for c in scene.skipped_types],
        "pythonSurfaceTriangles": int(surface.n_cells),
    }
(out / "file_coded.golden.json").write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
print(json.dumps(golden, indent=2, sort_keys=True))
```
