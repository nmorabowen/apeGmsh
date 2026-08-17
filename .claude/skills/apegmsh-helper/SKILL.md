---
name: apegmsh-helper
description: Use whenever the user is working with apeGmsh — the structural-FEM wrapper around Gmsh with OpenSees integration. Triggers on building FEM models from CAD/STEP imports, Part-based assembly workflows, composite-based geometry/mesh/constraint APIs (g.model, g.mesh, g.physical, g.constraints, etc.), void/hole authoring (as_void, add_polyline, add_void_sweep/loft, boolean.apply_voids — ADR 0097), labelled g.decouple_node as RBE2/RBE3 master, the apeSees(fem) OpenSees bridge with typed primitives and automatic MP-constraint emission, staged analysis (ops.stage), loads/masses/constraints resolution into the FEMData broker, native model.h5 persistence (FEMData.to_h5/from_h5, save_to=/g.save(), apeGmsh.from_h5), model composition (g.compose), post-processing OpenSees output via Results (from_native/from_mpco/from_recorders) with the interactive and web viewers (results.viewer / results.show_web), and exporting models to OpenSees Tcl or openseespy scripts. Covers apeGmsh's own abstractions on top of Gmsh and OpenSees. For raw gmsh API questions see the gmsh-structural skill; for raw OpenSees analysis commands see opensees-expert; for FEM theory first principles see fem-mechanics-expert.
---

# apeGmsh — structural FEM wrapper around Gmsh
<!-- skill-freshness: verified against apeGmsh main@5c92ca92 (2026-08-15) · signatures live in the references; src/ is not the authoring lookup (ADR 0096) -->

apeGmsh is the user's in-house Gmsh wrapper. It lives at
`C:\Users\nmora\Github\apeGmsh`, and the *core idea* is:

> Describe a model **once** — geometry, labels, loads, masses, constraints,
> per-node ndf — then hand a solver-agnostic **snapshot** (`FEMData`) to any
> FEM solver. OpenSees gets a first-class bridge; anyone else reads the
> snapshot. The same snapshot persists to `model.h5`, composes into larger
> assemblies, and drives post-processing of OpenSees output.

Whenever the user is working in this project or writing code against
apeGmsh, prefer the apeGmsh API over raw `gmsh.*` calls. Raw `gmsh` calls
still work (you're holding the same session), but the whole point of the
wrapper is that you don't have to write them.

## Lookup (ADR 0096)

Pick **one** row, read that file (or that heading), then write code.
Do not slurp this whole skill. Do not grep `src/apeGmsh/` to author a
model script. Prefer labels / physical-group names over tags.

`src/` read/grep is an **index miss**: allowed only when the reference
has no hit **and** you are changing apeGmsh itself.

| Task | Read only |
|---|---|
| First script / session skeleton | `references/workflows.md` |
| Public composite signature | MCP `lookup(symbol)` or `python -m apeGmsh.studio.lookup SYMBOL` |
| Public API map (headings) | `references/api-cheatsheet.md` (matching heading) |
| FEMData / `model.h5` | `references/fem-broker.md` |
| `apeSees` / stages / emit | `references/opensees-bridge.md` |
| After mesh or solve check | `references/assess.md` |
| Results / plot / stills | `references/results.md` |
| `compose` / `from_h5` / Assembly | `references/compose.md` |
| Rebar cages | `references/rebar.md` |
| ETABS / analytical import | `references/interop.md` |
| Section properties / fiber section | `references/section-properties.md` |
| Ladruno fork-only | `references/ladruno.md` |
| Build failed / anti-pattern | `references/gotchas.md` |
| Studio / MCP / picks | this file, **Studio** paragraph |
| Promote a miss / skill lie | `python -m apeGmsh.studio.profile --promote` (MCP `kind=miss` only; writes nothing) |

What each file contains (do not read them all):

- `api-cheatsheet.md` — public `g.*` / `apeSees` map (matching heading; signatures via MCP `lookup` / lookup CLI)
- `fem-broker.md` — `FEMData`, `model.h5`
- `opensees-bridge.md` — `apeSees`, stages, emit, ndf, remote
- `assess.md` — verdict + stills after mesh/solve (not Qt)
- `results.md` — `Results.from_*`, plot, render, lineage
- `compose.md` — `g.compose`, `from_h5`, Assembly
- `rebar.md` — `g.rebar` cages
- `interop.md` — ETABS / analytical import
- `section-properties.md` — `SectionProperties`, fiber handoff
- `workflows.md` — end-to-end patterns (not a forced pipeline)
- `gotchas.md` — anti-patterns
- `ladruno.md` — fork-only emit/read

Published docs: <https://nmorabowen.github.io/apeGmsh/> (`docs/`, ADR 0079).
`internal_docs/` is unpublished. Changelog Unreleased is the moving truth.

## Mental model

Four concepts, in this order:

**1. A session (`g`) owns a single Gmsh kernel.** Open it with
`g.begin()` / `g.end()` or a `with apeGmsh(...) as g:` block. Every
composite — `g.model`, `g.mesh`, `g.loads`, `g.masses`, `g.constraints`,
`g.embed`, `g.rebar`, `g.decouple_node`, etc. — is a thin namespace that talks
to that shared kernel.
At the top level the session *is* the assembly — `apeGmsh.Assembly` does
**not** exist (a deliberate v1.0 guard). OpenSees is **not** a session
composite (`g.opensees` was removed) — it is the separate post-session
bridge `apeSees(fem)`.

> For spatially coupling several saved `model.h5` modules there is now a
> declarative, **sub-path** builder: `from apeGmsh.assembly import Assembly`
> → `.add(...).couple(...).materialize()` (shipped v2.0.0, PR #433; couple
> kinds `equal_dof` / `tied_contact` / `tie` incl. `enforce="equation"` +
> `method="mortar"`, ADR 0085/0086). It's a thin wrapper that *produces* a
> composed session; see `references/compose.md`. For everything else, build
> multi-part models with `g.compose(...)` / `apeGmsh.from_h5`. **Mixed
> element ORDER across parts (hex20 + hex8) is ONLY possible via this
> compose route** — `set_order` is session-global, and `Part` is
> geometry-only by contract (ADR 0085); order-mismatched interfaces want
> `tie(method="mortar", enforce="equation")` (ADR 0086) — see
> `references/compose.md` §"Independent meshes per part".

**2. Composites split by concern.** `g.model` splits into
`geometry / boolean / transforms / io / queries`. `g.mesh` splits into
`generation / sizing / field / structured / editing / queries /
partitioning`. The OpenSees bridge `apeSees(fem)` exposes typed namespaces
(`uniaxialMaterial / nDMaterial / section / geomTransf / beamIntegration /
element / timeSeries / pattern / recorder`) plus flat verbs (`model / fix /
mass / build / stage / tcl / py / h5 / run`). Every method has a home — you
rarely reach into `gmsh.*`.

**3. `FEMData` is the solver contract.** When the mesh is ready, call
`fem = g.mesh.queries.get_fem_data(dim=3)` (or `dim=2`, or `None` for all
dims). The returned `FEMData` is an immutable snapshot with `fem.nodes`,
`fem.elements`, `fem.info`, `fem.inspect`. It works without a live Gmsh
session, round-trips through `model.h5` (`to_h5` / `from_h5`), composes into
larger assemblies, and every solver bridge consumes it. **Turn coordinates
into labels with the `.select()` chain** — `g.model.select(...)` pre-mesh,
`fem.nodes.select(...)` / `fem.elements.select(...)` / `g.mesh_selection.select(...)`
post-mesh; never raw tags (see `references/api-cheatsheet.md`).

**4. MP constraints + per-node ndf emit automatically.** When the snapshot
carries multi-point constraints (`fem.nodes.constraints`,
`fem.elements.constraints`) the `apeSees(fem)` bridge auto-emits the matching
`equalDOF` / `rigidLink` / `rigidDiaphragm` / `ASDEmbeddedNodeElement` deck
lines (and an `ops.constraints.Transformation()` handler when present); per-node
ndf is inferred from element classes and wired in too. **Do not hand-emit
these** — it double-constrains the model. (Shipped v2.0.0, ADR 0022.) The
same is true of the side-lists that emit *elements* rather than MP equations:
`fem.elements.contacts` and `fem.elements.interfaces`
(`g.constraints.interface()`, the unilateral + slip-capped coincident-pair
`zeroLength` bed — ADR 0093).

## Core workflow

Every apeGmsh script follows the same skeleton. Learn this, and everything
else is filling in blanks:

```python
# verified: tests/test_femdata_from_h5.py::test_session_save_then_from_h5
#           tests/opensees/integration/test_runnable_deck.py::test_tcl_deck_contains_constraint_lines
from apeGmsh import apeGmsh

with apeGmsh(model_name="my_model", save_to="my_model.h5") as g:
    # 1-5: GEOMETRY (occ) → PHYSICAL GROUPS (dim-unique names) → pre-mesh
    #      LOADS/MASSES/CONSTRAINTS (reference labels/PGs) → MESH → SNAPSHOT.
    #      (workflows.md §1 has the fully-commented version.)
    g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")
    g.physical.add_volume("body", name="Body")
    with g.loads.case("dead"):
        g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)
    g.masses.volume("Body", density=2400)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)   # the solver contract; autosaved to my_model.h5 on exit
    print(fem.info)                            # "N nodes, M elements, bandwidth=..."

# 6. OPENSEES (optional) — post-session bridge, typed primitives.
#    Masses/fixities are re-declared explicitly; loads are OPT-IN — a
#    g.loads.case reaches the deck only via p.from_model(case) inside a
#    pattern (ADR 0051). MP constraints + per-node ndf emit AUTOMATICALLY.
from apeGmsh.opensees import apeSees

ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(pg="Body", material=conc)
ops.fix(pg="Base", dofs=(1, 1, 1))
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("dead")                    # import the session's "dead" case
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))   # + an ad-hoc bridge load
ops.py("model.py")     # or ops.tcl(...), ops.h5(...), ops.run()
```

This is the **happy path**. For anything more involved (multi-part assembly,
coupled shell + frame, pushover, staged SSI, reading results back), read the
matching reference — don't improvise.

**Studio (ADR 0095).** `python -m apeGmsh.studio script.py` replays
up to `--phase` (default `model`: **stops before** `generate()`;
`--phase mesh` stops before `apeSees` / `Results`). The session stays
open. MeshViewer if the replay produced elements, otherwise
ModelViewer. Picks write `.apegmsh/selection.json` under the resolved project root
(`root=` / `--root` / `APEGMSH_ROOT` / nearest `.apegmsh/` / cwd —
ADR 0095 INV-15). A successful
stop writes `.apegmsh/names.json` — labels / PGs / counts / bboxes of
what exists, not what was clicked — and appends `.apegmsh/runs.jsonl`
(timestamp, hash, phase, counts). `python -m apeGmsh.studio --status`
reads those files without replaying. The host window refreshes itself
(ADR 0095 Amendment 4): a Refresh toolbar action / F5 replays the
entry script, and a file watch replays it automatically when the
saved file's mtime settles (`--no-watch` or the Watch toggle turn it
off) — both honor the `busy.json` lock and append `runs.jsonl` with
`trigger: "refresh"` / `"watch"`, so after editing the script the
human's window follows the save; the agent does not need to reopen
the host, and must not treat a `BUSY` `run_until` as an error while
the host is replaying. Before a spatial edit, read the
envelope, `names.json`, and `--status` — identity is
`labels` / `physical_groups` / `phase`, not dimtags.
An unnamed pick's first script edit is `.to_label()` / `.to_physical()`.
After solve, still `assess()` + `render()`, not `viewer()`.
When the agent must look, write visors under `.apegmsh/visors/`
(`results.render` / labeled matplotlib) and read those PNGs — never
capture the OS desktop or the human Qt window.
Read-only MCP (ADR 0095 S4a–S4e / S5a / 0096 S3): `python -m apeGmsh.studio.mcp` (needs
`pip install mcp`). Tools: `status`, `get_selection`, `lookup`, `run_until`,
`assess`, `render`, `animate(kind=history|yield)`, `results_pin`,
`emit_report(format=markdown|html|canvas)`, `highlight`,
`promote_selection` — names first, no Qt, no `setup=`. Every tool
accepts optional `root=` (INV-15). `status(mode=brief|full)` defaults
to **brief** (root / last-run / labels / PGs / counts / pick + `text`);
use `mode=full` only for debugging. **Root check → brief; entity
hunting → `lookup` / `names.json`, not a full status dump.**
`lookup(symbol)`
is See-family inspect of the generated index (~20 lines); not CAD;
does not grep `src/`. `kind=yield`
is a von Mises contour of `von_mises_stress` on auto-scaled deform
(0.12 of diagonal; not an iso-clip; π-plane later). `highlight`
writes `.apegmsh/highlight.json` only (host file-poll + `select_batch`;
does not write `selection.json`). `promote_selection` suggests
`g.model.select(None, dim=).in_box(lo, hi).to_label()` /
`.to_physical()` and does not write the `.py`. `emit_report`:
markdown is the archive (`docs/`); html is a print skin there;
canvas is a live Cursor projection (IDE `canvases/`, not
git-portable) and still writes the markdown chapter.
Run `assess` any time before `emit_report` — it writes
`.apegmsh/assess.json` and the chapter's Assess section picks up that
snapshot's verdict, findings, and skipped list automatically (`--pin`
copies it, ADR 0095 Amendment 5); do not paste `assess` output into
the chapter by hand.
`kind=formation` is later. Do not invent MCP tools for
`g.model.*` / `apeSees` primitives. Token-budget profiler is
`python -m apeGmsh.studio.profile` (sidecar; not an MCP tool).
Studio generation order (CAD then
mesh, quotations on, parallel processes) is a **recommendation** for
a given script, not a required path (ADR 0096 INV-14). Authored
chapters go in `docs/`;
visors stay under `.apegmsh/visors/`. Cursor config is
`.cursor/mcp.json` → `scripts/studio-mcp.ps1` (quiet env only; do not
pin cwd to the library repo — pass `root=` or set `APEGMSH_ROOT`). Set
`LADRUNO_OPENSEES_QUIET=1` before Python starts (the office venv
`.pth` banner goes to stdout and would break JSON-RPC)::

    {
      "mcpServers": {
        "apegmsh-studio": {
          "command": "powershell.exe",
          "args": [
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", "scripts/studio-mcp.ps1"
          ],
          "env": {
            "LADRUNO_OPENSEES_QUIET": "1",
            "APEGMSH_QUIET": "1"
          }
        }
      }
    }

## When things go wrong: the usual suspects

1. **`RuntimeError: session is already open`** — `begin()` called twice, or a
   crashed notebook cell left Gmsh initialized. Fix: `gmsh.finalize()` or
   restart the kernel. Prefer the `with` form.
2. **Physical group resolves to wrong dimension** — `g.physical.add(1, ...)`
   and `g.physical.add(2, ...)` are different groups. Keep PG names
   dimension-unique so `apeSees(fem)` `pg=` selectors resolve unambiguously.
3. **"No label, physical group, or part named X"** — resolution order is
   `label → physical group → part label`; the name exists in none. Inspect
   with `g.labels.get_all()`, `g.physical.summary()`, `g.parts.labels()`.
4. **`fem.elements.connectivity` raises `TypeError`** — the mesh has multiple
   element types. Iterate `for group in fem.elements:` or filter with
   `fem.elements.select(element_type="tet4").connectivity` (or
   `.result().resolve(element_type=...)` for a mixed selection).
5. **Labels without physical groups** — labels on the main session don't
   auto-promote to PGs (only `Part` sessions do). Call
   `g.labels.promote_to_physical("name")` or add `g.physical.add(...)`.
6. **`Results.from_native(...)` raises `TypeError`** — the constructor now
   *requires* `model=` (`from_mpco` requires `model_h5=`). See
   `references/results.md`.
7. **Don't open Qt to check a solve — and don't capture the human's
   screen.** After `get_fem_data` / `Results.from_*`,
   `report = fem.assess()` / `results.assess(figures=True)`; print
   `report.text`; `read_file` each PNG in `report.figures`; act on
   `severity="error"`. Inspect is inventory, not the verdict. Extra
   stills go under `.apegmsh/visors/` via `results.render` / labeled
   `results.plot.*` — never OS-screenshot the desktop or the live
   viewer. Qt / web viewers are for humans who asked to look.
   `results.viewer()` default is `blocking=None` (auto: scripts block,
   a Jupyter kernel spawns a subprocess or falls back to `show_web()`
   for in-memory Results). An explicit `blocking=True` still kills a
   Jupyter kernel. See `references/assess.md`.

Which reference covers a given failure: `BridgeError` / staged / ndf →
`opensees-bridge.md`; `MalformedH5Error` / `SchemaVersionError` →
`fem-broker.md`; finding codes / after-solve check → `assess.md`;
`LineageError` / viewer crash → `results.md`;
`MeshRecipeError` / selection → `api-cheatsheet.md`; `GeometryValidationError`
→ `gotchas.md`; `SectionMeshError` / `CompositeSectionError` /
`SectionAnalysisError` → `section-properties.md`.

## Version & layout facts you can rely on

These are the anti-hallucination guardrails — the removed/renamed surfaces an
agent most often gets wrong. Everything quantitative (schema integers, full
signatures) lives in the references.

- **Version is v2.0.0** (`pyproject.toml`). A stale editable install may print
  `v1.6.0` in the banner — trust the source, not the banner. Anything claiming
  "v1.0" is stale. v2.0.0 shipped the three-broker chain
  `FEMData ⊂ OpenSeesModel ⊂ Results`, auto MP-constraint emission, and deleted
  `BindError`.
- **Removed/renamed** (never recommend these): `g.mass` → `g.masses`;
  `g.opensees` → the post-session `apeSees(fem)` bridge; `g.node_ndf` → ndf is
  inferred from element classes + `ops.ndf` for element-less nodes (ADR
  0048/0049, see `opensees-bridge.md`); flat v0.x forms (`g.add_point`,
  `g.model.fuse`, `g.initialize`) → sub-composite forms; `apeGmshViewer` →
  `results.show_web()` / `ResultsViewer`.
- **Sidecar modules** (separate imports, NOT session composites):
  `from apeGmsh.hpc import Cluster, Job` (remote SLURM, pairs with
  `ops.run_remote`), `from apeGmsh.sensitivity import Sensitivity` (FD
  gradient/calibration), `from apeGmsh.studio import SelectionEnvelope`
  (`python -m apeGmsh.studio script.py --phase model` — stops before
  `generate()`; `.apegmsh/selection.json` + `names.json` + `runs.jsonl`;
  `--status` inspects them, ADR 0095). Details in `api-cheatsheet.md`.
- **Schema constants** live in `fem-broker.md` (neutral + bridge zones, ADR
  0023) and the viewer session schema in `results.md` — they drift, so read the
  number from there, not from memory.

Before claiming a method or signature exists, call MCP `lookup(symbol)`
or run `python -m apeGmsh.studio.lookup SYMBOL` (ADR 0096). If that
misses, read `references/api-cheatsheet.md` (the matching heading).
Do not `Read` `src/apeGmsh/studio/_api_index.json`.
`src/apeGmsh/` is for maintaining the library, not for writing a
model script.

Promotion (ADR 0096 S5): a MCP `lookup` **miss** (`kind=miss`, not
ambiguous) that repeats ≥3 times in the **model-cwd**
`.apegmsh/mcp_calls.jsonl` is eligible for a reviewed index/router
PR. `--promote` prints that list and writes nothing. A skill error
(skill vs index) owes a PR after one confirmation. Working ImageMage
steps stay comments in *that* `.py` and/or a Recommendation in
`workflows.md` — never a required pipeline. Propose a PR; do not
write `skills/apegmsh/` or run `sync_skill.py` in-session; do not
edit `.claude/skills/apegmsh-helper/`.
