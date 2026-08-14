# Assess + render — agent check after mesh / solve
<!-- skill-freshness: verified against apeGmsh grok/adr94-closeout (2026-08-14) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

After `get_fem_data` / `Results.from_*` the check is `assess()`, then
`read_file` each PNG in `report.figures`. Not `viewer()` / `show_web()`.
Qt is for humans who asked to look, and only after assess.

Read `report.skipped` **and** the Verdict line before treating
`0 error(s)` as a pass. A FAIL-reserved check that did not run is
listed as **Not evaluated**, not OK. Scope is the last recorded step
only (INV-9) — a mid-history blow-up that finishes finite is
invisible.

Three verbs stay distinct:

| Verb | Job | Home |
|---|---|---|
| `inspect` | Inventory (tables, strings) | `fem.inspect`, `results.inspect` |
| `diagnose` | Routing of *one* empty query | `results.inspect.diagnose(component)` |
| `assess` | Verdict: findings + short markdown | `fem.assess()`, `results.assess()` |

There is no `g.assess`. Live CAD health stays
`g.model.io.diagnose() → ImportHealth`. There is no
`OpenSeesModel.assess()` — assess does **not** check supports, loads,
or “case never imported.”

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
for code, reason in report.skipped:         # branchable (Amendment 1)
    ...
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
Unbound `results.render` / `render_pack` raise even under that env
(need `model=` / `model_h5=` / `bind`).

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

## Finding catalog (v1 + Amendment 1) → next action

New codes are an ADR 0094 amendment. The library does not emit `fix=`
Python. Branch on `finding.code` and `report.skipped`.

| Code | Sev | Next action |
|---|---|---|
| `MODEL.EMPTY` | error | No nodes or elements. Do not solve. Rebuild geometry / mesh. |
| `MESH.INVERTED` | error | `tet4`/`hex8`: negative/zero corner volume. Planar `tri3`/`quad4`: **degeneracy only** (zero area / bowtie). Winding is a convention — a clockwise slab is not inverted. **Not judged** (listed in `skipped`, not OK): `prism6`, `pyramid5`, quadratic/Bézier, all 1-D, 2-D cells in a non-planar 3-D mesh. Planarity is mesh-wide. |
| `MESH.COINCIDENT_UNBRIDGED` | warning | Pair shares XYZ and the bridge list is empty. `g.model.queries.make_conformal(...)` or `g.constraints.equal_dof` / `rigid_link` / contact. Bridged pairs (including a zero-length `line2`) are not findings. `tol=1e-6` is **absolute** (unit trap in mm models — misses, does not over-report). |
| `MESH.EMPTY_PG` | warning | Named PG with 0 nodes. Fix the group or drop it. |
| `RES.UNBOUND_FEM` | error | `results.fem is None`. Construct with `model=` / `model_h5=` or `results.bind(fem)`. |
| `RES.NO_STAGE` | error | No stages recorded. Re-run with a recorder / stage. |
| `RES.NAN` / `RES.INF` | error | Unexplained non-finite on a recorded component at last step. Node-level **union-merge fill** (displacement everywhere + reactions on supports) is skip, not FAIL — see `skipped`. Re-record unexplained NaNs. |
| `RES.LINEAGE` | warning | `results.lineage.warnings` — integrity, not physics. Assess never raises (ADR 0021). |
| `RES.U_VS_DIAG` | info | ‖u‖_max / model diagonal. Never warning/error in v1. Skip `kind="mode"`. |
| `RES.ENERGY_ERR` | info | Ladruno `ERR` in **%**. Measurement, not a gate. Native/MPCO → skip (`TypeError`), not OK. |
| `RES.ENERGY_NONFINITE` | warning | Last-step `ERR` is NaN/Inf. The one real energy signal. |

A check that cannot run is omitted, not scored OK. `report.skipped`
and the Verdict **Not evaluated (FAIL-reserved)** line name those
codes.

## Out of v1

`CAD.SLIVER_*`, `MESH.SICN_LOW` from H5, `g.assess`,
`OpenSeesModel.assess()`, AISC/ACI/fy, “in equilibrium”, invented
units. Design checks belong in apeSteel / apeConcrete.

`inspect` / `diagnose` remain for tables and one-component routing.
Do not dump `node_table()` into context as the verdict.
