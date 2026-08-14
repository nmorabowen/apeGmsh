---
name: apegmsh-helper
description: Use whenever the user is working with apeGmsh — the structural-FEM wrapper around Gmsh with OpenSees integration. Triggers on building FEM models from CAD/STEP imports, Part-based assembly workflows, composite-based geometry/mesh/constraint APIs (g.model, g.mesh, g.physical, g.constraints, etc.), the apeSees(fem) OpenSees bridge with typed primitives and automatic MP-constraint emission, staged analysis (ops.stage), loads/masses/constraints resolution into the FEMData broker, native model.h5 persistence (FEMData.to_h5/from_h5, save_to=/g.save(), apeGmsh.from_h5), model composition (g.compose), post-processing OpenSees output via Results (from_native/from_mpco/from_recorders) with the interactive and web viewers (results.viewer / results.show_web), and exporting models to OpenSees Tcl or openseespy scripts. Covers apeGmsh's own abstractions on top of Gmsh and OpenSees. For raw gmsh API questions see the gmsh-structural skill; for raw OpenSees analysis commands see opensees-expert; for FEM theory first principles see fem-mechanics-expert.
---

# apeGmsh — structural FEM wrapper around Gmsh
<!-- skill-freshness: verified against apeGmsh main@56cd64ec (2026-08-04) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

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

## Using this skill with any model (Claude or local)

This skill works whether or not you can inspect the source:

- **If you can run shell/grep** (Claude Code, capable agents): use the
  references as a map, but **verify exact signatures in `src/apeGmsh/`**
  before relying on them — apeGmsh moves fast. Each reference's
  `skill-freshness` stamp says which commit it was last checked against.
- **If you can't** (limited tools/context): trust the freshness-stamped
  snapshot, but **prefer labels / physical-group names over exact tags or
  numeric IDs** (names are stable; tags drift), and treat an exact signature
  as "try it; if a keyword errors, the arg names moved." Read one reference
  at a time — each is self-contained.

## Before writing code: read the right reference

The library is big. Don't try to remember every composite — read the
reference file that matches the task, *then* write the code. The
references are tight; reading them is cheap. **New to apeGmsh? Read
`api-cheatsheet.md` then `workflows.md` first.**

- **`references/api-cheatsheet.md`** — one-page map of every session
  composite (`g.model.*`, `g.mesh.*` incl. `g.mesh.recipe` one-call meshing,
  `g.parts`, `g.loads`, `g.displacements`, `g.masses`, `g.constraints` incl.
  RBE2/RBE3 coupling knobs + fork `contact()`/`contact_plane()` + `enforce=` tie routes
  + `interface()` unilateral coincident-pair springs,
  `g.embed`, `g.rebar`, `g.decouple_node`, `g.physical`, `g.labels`,
  `g.mesh_selection`) plus the post-session `apeSees(fem)` bridge and the
  standalone modules `apeGmsh.hpc` / `apeGmsh.sensitivity` / `apeGmsh.interop`
  / `apeGmsh.studio`,
  and the methods on each. **Read this first** for any non-trivial apeGmsh task
  — it saves you from guessing signatures.
- **`references/fem-broker.md`** — deep dive on `FEMData`, the broker
  returned by `g.mesh.queries.get_fem_data(dim=...)`, **plus native
  persistence** (`FEMData.to_h5` / `from_h5`, `save_to=` / `g.save()`,
  schema constants, integrity checks). Read when the task touches nodes,
  elements, iteration, solver hand-off, or saving/reloading a model.
- **`references/opensees-bridge.md`** — the `apeSees(fem)` bridge:
  typed-primitive materials/sections/elements, explicit `ops.fix`/`ops.mass`/
  `ops.pattern`, **automatic MP-constraint emission** (incl. `enforce=` tie
  routes + fork contact/embed, ADR 0068/0073), **damping**
  (`ops.damping`), the **moment-tensor seismic source** (`p.moment_tensor` /
  `ops.fault.from_shakermaker` + `MomentStep`/`Yoffe` S(t) helpers, ADR 0062),
  **staged analysis** (`ops.stage(...)` + `s.*` verbs), **per-node ndf** wiring,
  `ops.tcl/py/h5/run` (incl. `per_rank=` partitioned decks, ADR 0061, and
  `stream=` constant-memory write-through emit, ADR 0065),
  **remote SLURM runs** (`ops.run_remote` / `apeGmsh.hpc`, ADR 0060), and
  **which OpenSees runs** (`OpenSeesTarget` / `ops.capabilities()`). Read this
  for any OpenSees generation task.
- **`references/assess.md`** — after `generate()` / `get_fem_data` /
  `Results.from_*`: `assess()` (verdict) then stills; closed finding
  catalog and code → next action. Not Qt. Read this for any
  post-mesh / post-solve check.
- **`references/results.md`** — `Results` post-processing of OpenSees output
  (`from_native` / `from_mpco` / `from_recorders`, all of which now
  **require `model=` / `model_h5=`**), the `results.model.fem` broker chain,
  `results.lineage`, the **after-solve assess check**
  (`fem.assess()` / `results.assess()`, then `read_file` on
  `report.figures`; inspect stays inventory — see `assess.md`),
  **user-defined scalar expressions**
  (`results.nodes.define` / `results.elements.gauss.define`, `mag(...)`,
  ADR 0076 — custom fields like `"von_mises_stress/250"` that show in the
  viewer picker), the **web viewers** (`show_web` / `serve_web`,
  kernel-safe; humans who asked to look), **headless video/GIF export**
  (`results.export_animation`), and the desktop-viewer **concurrent
  geometries** API (`director.geometries`, ADR 0058 — multiple deform
  states / offsets / stage pins side-by-side). Read for anything reading
  back solver results or plotting.
- **`references/compose.md`** — model composition: `g.compose(...)`,
  `apeGmsh.from_h5(...)` chain-phase sessions, anchors vs translate, nested
  compose, and the string-keyed `'Module'` viewer color modes. Read when
  assembling several saved `model.h5` modules into one model.
- **`references/rebar.md`** — `g.rebar` RC reinforcement-cage authoring (ADR
  0066/0067): `column` / `beam` / `circular_column` / `wall` generators, ACI-318
  detailing standards, hand-authored bars/stirrups/bundles, and `place(...)`
  (conformal vs embedded coupling to the host continuum). Read when the user
  detail's reinforcement / asks for rebar cages.
- **`references/interop.md`** — `apeGmsh.interop`: import an analytical model
  (apeETABS `*.sm.json`) into a conformal beam+shell mesh and an `apeSees` deck
  (`import_structural_model` / `apply_subgrade_springs` / `build_opensees` /
  `solve_and_extract`, ADR 0009). Read when the user brings an ETABS / analytical
  model into apeGmsh. (ADR 0072's `emit_elements(skip=)` decomposition is
  *Proposed*, not shipped — the ref flags it.)
- **`references/section-properties.md`** — the `SectionProperties`
  cross-section analyzer (ADR 0078): geometric / warping / plastic / stress on
  any meshed 2-D face, the **rigidity-form naming law** (`EIxx_c` always valid;
  `Ixx_c` raises on composites → `transformed(e_ref=)`), composite authoring
  (PGs must **partition** the face — cut-then-fragment, shared-line law),
  `disconnected="raise"|"sum"`, the flat-face builders (`W_face`, …), the
  OpenSees handoff (`ops.section.ComputedSection(analysis=)` /
  `to_elastic_section` — one lowering owns `Ixx_c→Iz` / `As_y/A→alphaY`,
  `ndm=` form selection, composite reference-moduli rules), and the Qt
  inspector (`sec.viewer(blocking=False)` in notebooks). §10 covers ADR
  0080 — `SectionDocument` (versioned JSON, continuum + fiber lanes, RC
  templates, the `add_embed` composite primitive, the `bars=` overlay),
  the `launch_builder()` Qt editor and its parity law, `moment_curvature`
  (fiber lane, wipes the OpenSees domain, never on a worker), and
  `handoff_snippet` / `export_script`. Read for any section-property,
  torsion-constant, shear-area, RC-section, moment–curvature, or
  "compute / author my custom section" task.
- **`references/workflows.md`** — end-to-end patterns: single-session,
  multi-part assembly, solid–frame coupling, pushover, staged SSI. Read when
  the user asks for a complete example or a workflow they haven't built.
- **`references/gotchas.md`** — the ❌→✅ anti-patterns list plus the subtle
  pitfalls that aren't obvious from the API (unit-dependent `remove_duplicates`
  tolerance, half-open `in_box`, Selection-v2 ADR-0017 gaps, and the **silent
  analysis-chain substitutions** — a static integrator under
  `analysis Transient` is discarded for `Newmark`, ADR 0082). Read when a build
  "should work" but doesn't, when a run converges to a *plausible but wrong*
  answer, or before writing constraint/selection/Results code from memory.
- **`references/ladruno.md`** — targeting the **Ladruno fork** of OpenSees
  (`nmorabowen/OpenSees@ladruno`): `OpenSeesTarget` (pin which build runs)
  vs `ops.capabilities()` (what it can do), the live backend resolver (now
  fork-preferring), fork-only Bézier elements, ExplicitBathe + SMS integrators,
  the Ladruno material / beam-column / analysis clusters, contact/embed handlers,
  EnergyBalance + `.ladruno` recorder, the `≥33000` class-tag band. Read **only**
  when wiring fork-specific emit/read or pinning a build; stock `openseespy`
  stays first-class.

The published docs site is <https://nmorabowen.github.io/apeGmsh/> (built
from `docs/` — ADR 0079): concept pages at `concepts/<page>/` (session,
geometry-and-cad, meshing, gmsh-under-the-hood, selection,
parts-and-assembly, sections, constraints, loads-and-masses, fem-broker,
opensees-bridge, results) and internals at `design/<page>/` (architecture,
principles, broker, parts-assembly, results). `internal_docs/` is
unpublished working notes — don't cite it as user-facing docs. When in
doubt, the `CHANGELOG.md` v2.0.0 section + Unreleased block is the source
of truth.

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
ModelViewer. Picks write `.apegmsh/selection.json` (cwd). A successful
stop writes `.apegmsh/names.json` — labels / PGs / counts / bboxes of
what exists, not what was clicked — and appends `.apegmsh/runs.jsonl`
(timestamp, hash, phase, counts). `python -m apeGmsh.studio --status`
reads those files without replaying. Before a spatial edit, read the
envelope, `names.json`, and `--status` — identity is
`labels` / `physical_groups` / `phase`, not dimtags.
An unnamed pick's first script edit is `.to_label()` / `.to_physical()`.
After solve, still `assess()` + `render()`, not `viewer()`.
When the agent must look, write visors under `.apegmsh/visors/`
(`results.render` / labeled matplotlib) and read those PNGs — never
capture the OS desktop or the human Qt window.
Read-only MCP (ADR 0095 S4a–S4d): `python -m apeGmsh.studio.mcp` (needs
`pip install mcp`). Tools: `status`, `get_selection`, `run_until`,
`assess`, `render`, `animate(kind=history|yield)`, `results_pin`,
`emit_report(format=markdown|html|canvas)`, `highlight`,
`promote_selection` — names first, no Qt, no `setup=`. `kind=yield`
is a von Mises contour of `von_mises_stress` on auto-scaled deform
(0.12 of diagonal; not an iso-clip; π-plane later). `highlight`
writes `.apegmsh/highlight.json` only (host file-poll + `select_batch`;
does not write `selection.json`). `promote_selection` suggests
`g.model.select(None, dim=).in_box(lo, hi).to_label()` /
`.to_physical()` and does not write the `.py`. `emit_report`:
markdown is the archive (`docs/`); html is a print skin there;
canvas is a live Cursor projection (IDE `canvases/`, not
git-portable) and still writes the markdown chapter.
`kind=formation` is later. Do not invent MCP tools for
`g.model.*` / `apeSees` primitives. Authored chapters go in `docs/`;
visors stay under `.apegmsh/visors/`. Cursor `mcp.json`::

    {
      "mcpServers": {
        "apegmsh-studio": {
          "command": "python",
          "args": ["-m", "apeGmsh.studio.mcp"]
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
signatures) lives in the references, which the file tells you to re-verify.

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

Before claiming a method or signature exists, confirm it in `src/apeGmsh/`;
`references/api-cheatsheet.md` indexes the public surface.
