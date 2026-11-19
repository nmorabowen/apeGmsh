# VF3-b fixture: a third variant of the VU10 plate on another mesh

`test/spec/study.e2e.ts` (slice VF3-b, #2960) draws a compare variant's user result in render
mode, and refuses one whose model is another mesh. It lays these files over a copy of the VU10
study (`../vu10/`), with VU19's `plate_cracked.user.h5` (`../vu19/`) beside the cracked model.

| File | What it is |
|---|---|
| `remeshed/plate_remeshed.h5` | the VU10 plate (2 x 1, `FourNodeQuad` on `Panel`, `ndm=2`, E = 30e9) meshed 6 x 3 instead of 4 x 2: another `fem_hash` than the VU10 pair, which share one |
| `remeshed/plate_remeshed.ladruno` | VU10's synthetic closed form with r = 1, stamped against that model by `write_nodal_history`; it pairs with its own model, so the compare set opens |
| `remeshed/plate_remeshed.user.h5` | one `footfall` entry (nodes, state-free, `@basis="model"`, component `ratio`), current against its own model: only the mesh refuses it beside the VU10 window |
| `study.variants.json` | VU10's two rows plus `Remeshed`, written by `write_variants` |
| `hashes.golden.json` | each model's `fem_hash` and `model_hash` from its `/meta/lineage`; the refusal names them |

Only the layout and the hashes are under test, never the engineering. The script below is the
whole generator. It imports apeGmsh, so it is kept here as text and never as a `.py` file of the
app (ADR 0112 D4, `scripts/check_app_wall.py` W1). Save it as `make_compare_render.py` outside
the app and run it with the opensees venv interpreter and `PYTHONPATH=<checkout>/src`:

    python make_compare_render.py <app>/test/fixtures/vu10 <app>/test/fixtures/compare_render

```python
"""Write the VF3-b fixture: a third variant of the VU10 plate on another mesh.

    PYTHONPATH=<worktree>/src python make_compare_render.py <vu10 dir> <out dir>

``<out dir>`` (the app's ``test/fixtures/compare_render``) receives:

- ``remeshed/plate_remeshed.h5``: the VU10 plate (a 2 x 1 rectangle, ``FourNodeQuad``
  on ``Panel``, ``ndm=2``, E = 30e9) meshed 6 x 3 instead of 4 x 2, so its
  ``fem_hash`` differs from the VU10 pair's (which share one);
- ``remeshed/plate_remeshed.ladruno``: the VU10 closed form with r = 1, stamped
  against that model by ``write_nodal_history`` (synthetic, as VU10's);
- ``remeshed/plate_remeshed.user.h5``: one ``footfall`` entry (nodes, state-free,
  ``@basis="model"``, component ``ratio``), current against its own model, so the
  only thing that refuses it beside the VU10 window is the mesh;
- ``study.variants.json``: VU10's two rows plus ``Remeshed``, written by
  ``write_variants`` beside a copy of the VU10 study;
- ``hashes.golden.json``: each model's ``fem_hash`` and ``model_hash`` as
  ``FEMData`` stamps them in ``/meta/lineage``.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import gmsh
import h5py
import numpy as np

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import apeSees
from apeGmsh.results.readers._ladruno_writer import write_nodal_history
from apeGmsh.results.user import UserResult, VariantDef, read_variants, staleness, write_user_file, write_variants

VU10 = Path(sys.argv[1])
OUT = Path(sys.argv[2])
STEPS = 3


def mesh(tmp: Path, nx: int, ny: int):
    with apeGmsh(model_name="plate", verbose=False, units="N_m_s", save_to=tmp / "plate.session.h5") as g:
        p0 = gmsh.model.geo.addPoint(0, 0, 0)
        edge = gmsh.model.geo.extrude([(0, p0)], 2, 0, 0, numElements=[nx])
        surf = gmsh.model.geo.extrude([edge[1]], 0, 1, 0, numElements=[ny], recombine=True)
        gmsh.model.geo.synchronize()
        g.model.sync()
        g.physical.add(2, [surf[1][1]], name="Panel")
        g.mesh.generation.generate(2)
        return g.mesh.queries.get_fem_data(remove_orphans=True)


def lineage(model: Path) -> dict:
    with h5py.File(model, "r") as f:
        a = f["meta/lineage"].attrs
        return {k: (a[k].decode() if isinstance(a[k], bytes) else str(a[k])) for k in ("fem_hash", "model_hash")}


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    fem = mesh(tmp, 6, 3)
    study = tmp / "study"
    shutil.copytree(VU10, study, ignore=shutil.ignore_patterns("*.py", "codes.golden.json", "study.variants.json"))
    model = study / "remeshed" / "plate_remeshed.h5"
    model.parent.mkdir(parents=True)
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    plate = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0, name="plate")
    ops.element.FourNodeQuad(pg="Panel", thickness=0.2, material=plate)
    ops.h5(str(model))
    back = FEMData.from_h5(str(model))
    ids = np.asarray(back.nodes.ids, dtype=np.int64)
    xyz = np.asarray(back.nodes.coords, dtype=np.float64).reshape(-1, 3)
    x, y = xyz[:, 0], xyz[:, 1]
    disp = np.stack([np.stack([-0.01 * (k + 1) * x * y, 0.5 * 0.01 * (k + 1) * x * x], axis=-1) for k in range(STEPS)])
    results = model.with_suffix(".ladruno")
    write_nodal_history(
        results, model_h5=model, node_ids=ids, coords=xyz, ndm=2,
        kind="static", time=(np.arange(STEPS) + 1.0) / STEPS, results={"DISPLACEMENT": disp},
        source="viewer-fixture", opensees_backend="unknown",
    )
    user = model.with_name("plate_remeshed.user.h5")
    footfall = 0.3 + 0.5 * x / 2.0
    write_user_file(
        user, model, results_file=results,
        results=(UserResult(name="footfall", target="nodes", basis="model", components=("ratio",), data=footfall.reshape(1, ids.size, 1), ids=ids),),
        combinations=None, writer="vf3b fixture",
    )
    st = staleness(user, model_h5=model, results_file=results)
    assert not st.results["footfall"].stale, st.results["footfall"].reason
    manifest = study / "study.variants.json"
    write_variants(manifest, [
        VariantDef(label="Uncracked", path="plate_uncracked.h5", params={"state": "uncracked", "Ie/Ig": 1.0, "E": 30e9}),
        VariantDef(label="Cracked", path="cracked/plate_cracked.h5", params={"state": "cracked", "Ie/Ig": 0.35, "E": 10.5e9}),
        VariantDef(label="Remeshed", path="remeshed/plate_remeshed.h5", params={"state": "uncracked", "Ie/Ig": 1.0, "E": 30e9, "mesh": "6 x 3"}),
    ])
    assert [r.label for r in read_variants(manifest).variants] == ["Uncracked", "Cracked", "Remeshed"]
    hashes = {
        "Uncracked": lineage(study / "plate_uncracked.h5"),
        "Cracked": lineage(study / "cracked" / "plate_cracked.h5"),
        "Remeshed": lineage(model),
    }
    assert hashes["Uncracked"]["fem_hash"] == hashes["Cracked"]["fem_hash"] != hashes["Remeshed"]["fem_hash"], hashes
    (OUT / "remeshed").mkdir(parents=True, exist_ok=True)
    for f in (model, results, user):
        shutil.copy2(f, OUT / "remeshed" / f.name)
    shutil.copy2(manifest, OUT / "study.variants.json")
    (OUT / "hashes.golden.json").write_text(json.dumps(hashes, indent=1) + "\n", encoding="utf-8")
    print(f"remeshed: {ids.size} nodes; hashes {json.dumps(hashes)}")
```
