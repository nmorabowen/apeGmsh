# Assess + render — agent check after mesh / solve
<!-- skill-freshness: verified against apeGmsh main@1a9b64aa (2026-08-13) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

After `get_fem_data` / `Results.from_*` the check is `assess()`, then
`read_file` each PNG in `report.figures`. Not `viewer()` / `show_web()`.
Qt is for humans who asked to look, and only after assess.

Three verbs stay distinct:

| Verb | Job | Home |
|---|---|---|
| `inspect` | Inventory (tables, strings) | `fem.inspect`, `results.inspect` |
| `diagnose` | Routing of *one* empty query | `results.inspect.diagnose(component)` |
| `assess` | Verdict: findings + short markdown | `fem.assess()`, `results.assess()` |

There is no `g.assess`. Live CAD health stays
`g.model.io.diagnose() → ImportHealth`. There is no
`OpenSeesModel.assess()`.

## Doors

```python
report = fem.assess()                       # findings only
report = results.assess()                   # + lineage as RES.LINEAGE
report = results.assess(figures=True)       # + results.render_pack
report = fem.assess(figures=True)           # + fem.render → mesh.png

print(report.text)                          # 40–80 lines markdown
for path in report.figures:                 # () if skipped / figures=False
    # read_file each PNG — do not invent a file that was not written
    pass
for f in report.findings:
    if f.severity == "error":
        # do not proceed to a human memo
        ...
```

`# src/apeGmsh/mesh/FEMData.py` (`assess` / `render`)
`# src/apeGmsh/results/Results.py` (`assess` / `render` / `render_pack`)
`# verified: tests/assess/test_assess.py`

`figures=True` default is `False`. Headless stays cheap. Pass `True`
when you need eyes. `APEGMSH_SKIP_VIEWER=1` or no GL: findings still
run, `figures` is `()`, prints
`[skip viewer] APEGMSH_SKIP_VIEWER set` or
`[skip viewer] no GL context` (exact). Skip ≠ pass.

Stills come from `viewers.render` (VTK offscreen). Closed `view=`:
`mesh` / `contour` / `deformed` / `reactions`. No `setup=`. In-process
is the default; `python -m apeGmsh.viewers render …` is the
out-of-kernel door so a GL crash does not kill the notebook.

Live session (needs an open Gmsh kernel; `from_h5` / closed →
`RuntimeError`): `g.model.render("geom.png")` (BRep, coarse tessellation
if no mesh), `g.mesh.render("mesh.png")` (live Gmsh, not `fem.render`).
Snapshot mesh stills stay on `fem.render`.

Do not call `g.model.viewer()` / `g.mesh.viewer()` / `results.viewer()`
/ `show_web` / `gui()` for diagnosis.

## Finding catalog (v1) → next action

New codes are an ADR 0094 amendment. The library does not emit `fix=`
Python. Branch on `finding.code`.

| Code | Sev | Next action |
|---|---|---|
| `MODEL.EMPTY` | error | No nodes or elements. Do not solve. Rebuild geometry / mesh. |
| `MESH.INVERTED` | error | Negative/zero linear-cell volume. Remesh or reverse winding. Quadratic / Bézier cells are skipped, not judged. |
| `MESH.COINCIDENT_UNBRIDGED` | warning | Pair shares XYZ and the bridge list is empty. `g.model.queries.make_conformal(...)` or `g.constraints.equal_dof` / `rigid_link` / contact. Bridged pairs (element / constraint strings present) are not findings. |
| `MESH.EMPTY_PG` | warning | Named PG with 0 nodes. Fix the group or drop it. |
| `RES.UNBOUND_FEM` | error | `results.fem is None`. Construct with `model=` / `model_h5=` or `results.bind(fem)`. |
| `RES.NO_STAGE` | error | No stages recorded. Re-run with a recorder / stage. |
| `RES.NAN` / `RES.INF` | error | Non-finite value on a recorded component (not a known fill sentinel). Re-record; do not treat as OK. |
| `RES.LINEAGE` | warning | `results.lineage.warnings` — integrity, not physics. Assess never raises (ADR 0021). |
| `RES.U_VS_DIAG` | info | ‖u‖_max / model diagonal. Never warning/error in v1. Skip `kind="mode"`. |
| `RES.ENERGY_ERR` | warning | Ladruno `ERR` channel. Native/MPCO → skip (`TypeError`), not OK. |

A check that cannot run is omitted, not scored OK. The markdown lists
skipped checks.

## Out of v1

`CAD.SLIVER_*`, `MESH.SICN_LOW` from H5, `g.assess`,
`OpenSeesModel.assess()`, AISC/ACI/fy, “in equilibrium”, invented
units. Design checks belong in apeSteel / apeConcrete.

`inspect` / `diagnose` remain for tables and one-component routing.
Do not dump `node_table()` into context as the verdict.
