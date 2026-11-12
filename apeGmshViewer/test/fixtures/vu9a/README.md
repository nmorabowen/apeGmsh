# VU9a fixtures: workspaces whose verdicts Python states

Every `.workspace.h5` here except `rows.workspace.h5` and `app_written/` was written by
`apeGmsh.results.user.write_workspace`, and every `*.golden.json` is what Python's
`WorkspaceFile` reads and judges. The app's reader must agree with each
(`test/reader/workspace.test.ts`, slice VU9a, #2803); the save tests copy `frame.h5`
(`test/main/workspaceSave.test.ts`, `test/main/workspaceDoors.test.ts`).

| File | What it is |
|---|---|
| `frame.h5`, `frame.workspace.h5` | h5-schema.md's worked case (`tests/results/user/_workspace_helpers.py` `frame`): five sets, one plot, one view |
| `cases/<case>.h5`, `verdicts.golden.json` | the frame re-written as each case of `tests/results/user/test_workspace_verdict.py` describes, and Python's verdict of every set against each |
| `rows.workspace.h5`, `rows.golden.json` | rows that refuse alone, written with h5py (the writer never writes one), and Python's refusals |
| `plate/plate.h5`, `plate/remeshed.h5`, `plate/plate.workspace.h5`, `remesh.golden.json` | a real gmsh re-mesh of a 12 m x 6 m plate (h = 1.5, then 1.0); node sets saved against the first, judged against the second: the corners survive, so `corners and inside` reads `partial` |
| `truncated.workspace.h5`, `truncated.golden.json` | the first half of `frame.workspace.h5`: Python reads it `not-hdf5`, and its writer renames it aside (D6); the app must do both |
| `app_written/frame.workspace.h5`, `app_written.golden.json` | a file the app's own `saveWorkspace` wrote against `frame.h5` (four creates and a rename), and what Python read and judged in it |

The generators import apeGmsh (and the app's main code), so they are kept here as text and
never as files of the app (ADR 0112 D4, `scripts/check_app_wall.py` W1). Save the first as
`make_vu9a.py` outside the app and run it with the opensees venv interpreter, then the second
as `app_write.mts` (run with Node from anywhere), then the
first again with `--read-app-written`:

    python make_vu9a.py <this directory> <checkout>/src <checkout>
    node app_write.mts <checkout>/apeGmshViewer
    python make_vu9a.py <this directory> <checkout>/src <checkout> --read-app-written
    python make_vu9a.py <this directory> <checkout>/src <checkout> --truncated

```python
"""Write the VU9a fixtures: workspaces and models whose verdicts Python states.

    python make_vu9a.py <fixture dir> <worktree>/src <worktree>
    python make_vu9a.py <fixture dir> <worktree>/src <worktree> --read-app-written

Every ``.workspace.h5`` here except ``rows.workspace.h5`` is written by
``apeGmsh.results.user.write_workspace``; every golden is what Python's
``WorkspaceFile`` reads and judges. The app's reader must agree on each
(test/reader/workspace.test.ts).

- ``frame.h5`` / ``frame.workspace.h5``: h5-schema.md's worked case (the frame
  of ``tests/results/user/_workspace_helpers.py``), with the five sets of
  ``test_workspace_verdict.py``, one plot and one view.
- ``cases/<case>.h5``: that frame re-written the way each case of
  ``test_workspace_verdict.py`` describes; ``verdicts.golden.json`` is
  Python's verdict of every set against each.
- ``rows.workspace.h5``: rows that refuse alone, written with h5py (the
  writer never writes one); ``rows.golden.json`` is Python's refusals.
- ``plate/plate.h5`` (h = 1.5) and ``plate/remeshed.h5`` (h = 1.0): a real
  gmsh re-mesh of one 12 m x 6 m plate; ``plate/plate.workspace.h5`` holds
  node sets saved against the first, and ``remesh.golden.json`` is Python's
  verdict of them against the second (the corners survive: ``partial``).
- ``--read-app-written``: after the app wrote ``app_written/frame.workspace.h5``
  (test/reader/workspace.test.ts writes it with ``VU9A_WRITE_FIXTURE=1``),
  record what Python reads from it in ``app_written.golden.json``.
"""
import json
import math
import shutil
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[3])

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from apeGmsh.results.user import PlotDef, SetDef, ViewDef, WorkspaceFile, write_workspace  # noqa: E402
from tests.results.user._workspace_helpers import frame  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)


def verdict_json(table):
    out = {}
    for v in table.verdicts:
        out[v.name] = {"verdict": v.verdict, "matched": v.matched, "total": v.total,
                       "members": list(v.members), "checked": v.checked,
                       "max_distance": None if math.isnan(v.max_distance) else v.max_distance}
    for r in table.refused:
        out[r.name] = {"refused": r.reason}
    return out


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if "--read-app-written" in sys.argv:
    ws = WorkspaceFile(OUT / "app_written" / "frame.workspace.h5")
    dump(OUT / "app_written.golden.json", {
        "revision": ws.revision, "writer": ws.writer, "version": ws.version,
        "sets": {s.name: {"kind": s.kind, "ids": s.ids.tolist(), "xyz": s.xyz.tolist(),
                          "fem_hash": s.fem_hash, "model_hash": s.model_hash} for s in ws.sets},
        "plots": [r.name for r in ws.plots], "views": [r.name for r in ws.views],
        "refused": [[r.table, r.row, r.name, r.reason] for r in ws.refused],
        "verdicts": verdict_json(ws.verdicts(OUT / "frame.h5")),
    })
    print("app_written.golden.json written")
    sys.exit(0)

if "--truncated" in sys.argv:
    # The first half of a real workspace: what Python reads it as, and what its writer does with it (D6).
    import tempfile
    import warnings
    from apeGmsh.results.user import WorkspaceRefused, WorkspaceRenamedAside
    whole = (OUT / "frame.workspace.h5").read_bytes()
    (OUT / "truncated.workspace.h5").write_bytes(whole[: len(whole) // 2])
    try:
        WorkspaceFile(OUT / "truncated.workspace.h5")
        read = "read"
    except WorkspaceRefused as exc:
        read = exc.reason
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(OUT / "frame.h5", Path(tmp) / "frame.h5")
        shutil.copy(OUT / "truncated.workspace.h5", Path(tmp) / "frame.workspace.h5")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            write_workspace(Path(tmp) / "frame.workspace.h5", sets=[SetDef("roof", "nodes", [31])])
        aside = sorted(p.name.split(".unreadable-")[0] for p in Path(tmp).iterdir() if ".unreadable-" in p.name)
        renamed = any(issubclass(w.category, WorkspaceRenamedAside) for w in caught)
    dump(OUT / "truncated.golden.json", {"bytes": len(whole) // 2, "read": read, "write_renamed_aside": renamed, "aside": aside})
    print("truncated.golden.json written")
    sys.exit(0)

# -- the worked case ------------------------------------------------------------
SETS = [
    SetDef("roof", "nodes", [31, 32, 33]),
    SetDef("base", "nodes", [1, 2, 3]),
    SetDef("column line", "nodes", [1, 11, 21, 31]),
    SetDef("springs", "elements", [501, 502]),
    SetDef("columns", "elements", [201, 202, 203]),
]
PLOT = {"kind": "history", "over": {"kind": "stage", "source": "pushover", "axis": "step"},
        "series": [{"quantity": {"group": "results", "field": "DISPLACEMENT", "component": "Ux"},
                    "target": {"kind": "set", "name": "roof"}, "reduce": "mean"}]}
VIEW = {"name": "roof_view", "camera": "iso"}
frame(OUT / "frame.h5")
ws = write_workspace(OUT / "frame.workspace.h5", sets=SETS, plots=[PlotDef("roof drift", PLOT)],
                     views=[ViewDef("iso", VIEW)], writer="make_vu9a")

diag = float(np.linalg.norm([12.0, 0.0, 9.0]))
CASES = {
    "current": {},
    "relabel": {"fem_hash": "b" * 32},
    "base3_moved": {"fem_hash": "b" * 32, "nodes": {3: (12.5, 0.0, 0.0)}},
    "column_up": {"fem_hash": "b" * 32,
                  "nodes": {n: (0.0, 0.0, z + 0.5) for n, z in ((1, 0.0), (11, 3.0), (21, 6.0), (31, 9.0))}},
    "springs_swap": {"model_hash": "c" * 32, "springs": [502, 501]},
    "tags_shift": {"model_hash": "c" * 32, "beam_tags": list(range(203, 212)), "springs": [201, 202]},
    "no_fem_hash": {"fem_hash": None},
    "no_model_hash": {"model_hash": None},
    "no_opensees": {"opensees": False},
    "node103_off": {"fem_hash": "b" * 32, "nodes": {103: (12.0, -1.0, 0.0)}},
    "tolerance": {"fem_hash": "b" * 32, "nodes": {31: (0.5e-9 * diag, 0.0, 9.0), 32: (6.0 + 2e-9 * diag, 0.0, 9.0)}},
}
(OUT / "cases").mkdir(exist_ok=True)
golden = {}
for name, kw in CASES.items():
    model = frame(OUT / "cases" / f"{name}.h5", **kw)
    golden[name] = verdict_json(WorkspaceFile(ws).verdicts(model))
dump(OUT / "verdicts.golden.json", golden)

# -- rows that refuse alone (the writer never writes one) -------------------------
STR = h5py.string_dtype(encoding="utf-8")
with h5py.File(OUT / "rows.workspace.h5", "w") as f:
    m = f.create_group("meta")
    m.attrs["workspace_schema_version"] = "1.0.0"
    m.attrs["revision"] = np.int32(7)
    m.attrs["writer"] = "make_vu9a (h5py)"
    m.attrs["save_id"] = "rowsfixture"
    rows = [  # name, kind, id_space, fem_hash, model_hash, ids
        ("ok", "nodes", "opensees", "f" * 32, "", [1, 2]),
        ("", "nodes", "opensees", "f" * 32, "", [1]),
        (" padded", "nodes", "opensees", "f" * 32, "", [1]),
        ("twin", "nodes", "opensees", "f" * 32, "", [1]),
        ("twin", "nodes", "opensees", "f" * 32, "", [2]),
        ("both", "both", "opensees", "f" * 32, "", [1]),
        ("fem ids", "nodes", "fem", "f" * 32, "", [1]),
        ("none", "nodes", "opensees", "f" * 32, "", []),
        ("again", "nodes", "opensees", "f" * 32, "", [3, 3]),
        ("no model hash", "elements", "opensees", "f" * 32, "", [201]),
    ]
    s = f.create_group("sets")
    for i, c in enumerate(("name", "kind", "id_space", "fem_hash", "model_hash")):
        s.create_dataset(c, data=np.array([r[i] for r in rows], dtype=object), dtype=STR)
    sizes = [len(r[5]) for r in rows]
    s.create_dataset("member_offsets", data=np.concatenate(([0], np.cumsum(sizes))).astype("i4"))
    ids = [t for r in rows for t in r[5]]
    s.create_dataset("ids", data=np.array(ids, dtype="i4"))
    s.create_dataset("xyz", data=np.arange(3 * len(ids), dtype="f8").reshape(-1, 3))
    good_plot = json.dumps(PLOT)
    plots = [
        ("drift", "apegmsh-plot/1", good_plot),
        ("old grammar", "apegmsh-plot/0", good_plot),
        ("not json", "apegmsh-plot/1", "{kind: history}"),
        ("a list", "apegmsh-plot/1", "[1, 2]"),
        ("nan", "apegmsh-plot/1", '{"kind": "history", "scale": NaN}'),
        ("dup key", "apegmsh-plot/1", '{"kind": "history", "kind": "xy"}'),
        ("open key", "apegmsh-plot/1", json.dumps({**PLOT, "colour": "red"})),
        ("float steps", "apegmsh-plot/1", json.dumps({**PLOT, "over": {**PLOT["over"], "steps": [0, 1.0]}})),
        ("int steps", "apegmsh-plot/1", json.dumps({**PLOT, "over": {**PLOT["over"], "steps": [0, -1]}})),
        ("each in xy", "apegmsh-plot/1", json.dumps({"kind": "xy", "over": PLOT["over"], "x": PLOT["series"][0],
                                                     "y": [{**PLOT["series"][0], "reduce": "each"}]})),
        ("profile float step", "apegmsh-plot/1", json.dumps({"kind": "profile", "axis": "x", "series": [{**PLOT["series"][0], "reduce": "each"}],
                                                             "state": {"kind": "stage", "source": "pushover", "step": 2.0, "mode": 0}})),
        ("profile ok", "apegmsh-plot/1", json.dumps({"kind": "profile", "axis": "x", "series": [{**PLOT["series"][0], "reduce": "each"}],
                                                     "state": {"kind": "stage", "source": "pushover", "step": 2, "mode": 0}})),
        ("huge float", "apegmsh-plot/1", '{"kind": "history", "scale": 1e400}'),
        ("drift", "apegmsh-plot/1", good_plot),
    ]
    p = f.create_group("plots")
    for i, c in enumerate(("name", "grammar", "body")):
        p.create_dataset(c, data=np.array([r[i] for r in plots], dtype=object), dtype=STR)
    views = [("iso", "apegmsh-view/1", json.dumps(VIEW)), ("v2", "apegmsh-view/2", json.dumps(VIEW))]
    v = f.create_group("views")
    for i, c in enumerate(("name", "grammar", "body")):
        v.create_dataset(c, data=np.array([r[i] for r in views], dtype=object), dtype=STR)
rw = WorkspaceFile(OUT / "rows.workspace.h5")
dump(OUT / "rows.golden.json", {
    "revision": rw.revision,
    "sets": [s.name for s in rw.sets], "plots": [r.name for r in rw.plots], "views": [r.name for r in rw.views],
    "refused": [[r.table, r.row, r.name, r.reason] for r in rw.refused],
})

# -- a real re-mesh ---------------------------------------------------------------
from apeGmsh import apeGmsh  # noqa: E402


def plate(path, h):
    with apeGmsh(model_name=f"vu9a-plate-{h}") as g:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 12.0, 6.0)
        g.model.sync()
        import gmsh  # noqa: E402
        g.physical.add(2, [t for _, t in gmsh.model.getEntities(2)], name="Slab")
        g.mesh.sizing.set_global_size(h)
        g.mesh.generation.generate(2)
        fem = g.mesh.queries.get_fem_data()
    fem.to_h5(str(path))
    return np.asarray(fem.nodes.ids, dtype=np.int64), np.asarray(fem.nodes.coords, dtype=float)


(OUT / "plate").mkdir(exist_ok=True)
ids, xyz = plate(OUT / "plate" / "plate.h5", 1.5)
corners = [int(i) for i, p in zip(ids, xyz) if p[0] in (0.0, 12.0) and p[1] in (0.0, 6.0)]
interior = [int(i) for i, p in zip(ids, xyz) if 0.5 < p[0] < 11.5 and 0.5 < p[1] < 5.5][:6]
pws = write_workspace(OUT / "plate" / "plate.workspace.h5", writer="make_vu9a", sets=[
    SetDef("corners", "nodes", corners),
    SetDef("corners and inside", "nodes", corners + interior),
    SetDef("inside", "nodes", interior),
])
plate(OUT / "plate" / "remeshed.h5", 1.0)
remesh = verdict_json(WorkspaceFile(pws).verdicts(OUT / "plate" / "remeshed.h5"))
assert remesh["corners and inside"]["verdict"] == "partial", remesh
dump(OUT / "remesh.golden.json", remesh)
print("written:", sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()))
```

```ts
// Write test/fixtures/vu9a/app_written/frame.workspace.h5 with the app's own door code.
import { copyFileSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
const APP = process.argv[2]; // <checkout>/apeGmshViewer
const h5wasm = await import(pathToFileURL(`${APP}/node_modules/h5wasm/dist/node/hdf5_hl.js`).href);
const ws = await import(pathToFileURL(`${APP}/src/main/workspace.ts`).href);
await h5wasm.ready;
const dir = `${APP}/test/fixtures/vu9a/app_written`;
rmSync(dir, { recursive: true, force: true });
mkdirSync(dir);
copyFileSync(`${APP}/test/fixtures/vu9a/frame.h5`, join(dir, "frame.h5"));
const deps = { ...ws.nodeDeps(h5wasm, h5wasm.ready, "apeGmshViewer 0.0.1"), saveId: () => "appwrittenfixture" };
const set = { model: join(dir, "frame.h5"), geometry: null, results: null, user: null, workspace: null };
for (const edit of [
  { op: "create", name: "roof", kind: "nodes", ids: [31, 32, 33] },
  { op: "create", name: "springs", kind: "elements", ids: [501, 502] },
  { op: "create", name: "columns", kind: "elements", ids: [201, 202, 203] },
  { op: "create", name: "Ünïcode base", kind: "nodes", ids: [3, 2, 1] },
  { op: "rename", table: "sets", from: "roof", to: "roof line" },
]) {
  const r = await ws.saveWorkspace(deps, set, edit);
  if (!r.ok) throw new Error(JSON.stringify(r));
}
rmSync(join(dir, "frame.h5"));
console.log("written", dir);
```
