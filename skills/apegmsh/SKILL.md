---
name: apegmsh
description: >
  apeGmsh is a Gmsh wrapper for structural FEM with first-class OpenSees
  integration. Use this skill any time the user is writing, reading,
  debugging, or reviewing code that imports apeGmsh (``from apeGmsh
  import apeGmsh`` / ``Part`` / ``FEMData`` / ``Results``) or touches the
  apeGmsh project at ``C:\Users\nmora\Github\apeGmsh``. Trigger for:
  building an ``apeGmsh`` session, composing geometry via
  ``g.model.geometry/boolean/transforms/io/queries``, meshing via
  ``g.mesh.generation/sizing/field/structured/editing/queries/
  partitioning``, labels, physical groups, FEMData broker
  (``g.mesh.queries.get_fem_data``), multi-part assemblies (``Part`` +
  ``g.parts``), loads / masses / constraints (``g.loads``, ``g.masses``,
  ``g.constraints``), per-node ndf (inferred + ``ops.ndf``), the ``apeSees(fem)``
  OpenSees bridge with typed primitives, post-processing OpenSees output
  via ``Results`` (``from_native`` / ``from_mpco`` / ``from_recorders``)
  and the web/Qt viewers, native ``model.h5`` persistence
  (``FEMData.to_h5`` / ``from_h5``, ``save_to=`` / ``g.save()``), model
  composition (``g.compose`` / ``apeGmsh.from_h5``), and exporting models
  to OpenSees Tcl or openseespy scripts. Covers apeGmsh's own abstractions
  on top of Gmsh and OpenSees. Also use it when the user says "meshing",
  "FEA mesh", "structural mesh", or "OpenSees from gmsh" in a context where
  apeGmsh is the wrapper of choice — the user built apeGmsh specifically
  to replace hand-rolled Gmsh + OpenSees glue and always prefers the
  apeGmsh API over raw gmsh calls.
---

# apeGmsh — structural FEM wrapper around Gmsh
<!-- skill-freshness: verified against apeGmsh main@8eeda7a3 (2026-07-06) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

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
  RBE2/RBE3 coupling knobs + fork `contact()`/`contact_plane()` + `enforce=` tie routes,
  `g.embed`, `g.rebar`, `g.decouple_node`, `g.physical`, `g.labels`,
  `g.mesh_selection`) plus the post-session `apeSees(fem)` bridge and the
  standalone modules `apeGmsh.hpc` / `apeGmsh.sensitivity` / `apeGmsh.interop`,
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
- **`references/results.md`** — `Results` post-processing of OpenSees output
  (`from_native` / `from_mpco` / `from_recorders`, all of which now
  **require `model=` / `model_h5=`**), the `results.model.fem` broker chain,
  `results.lineage`, **user-defined scalar expressions**
  (`results.nodes.define` / `results.elements.gauss.define`, `mag(...)`,
  ADR 0076 — custom fields like `"von_mises_stress/250"` that show in the
  viewer picker), the **web viewers** (`show_web` / `serve_web`,
  kernel-safe), **headless video/GIF export** (`results.export_animation`),
  and the desktop-viewer **concurrent geometries** API
  (`director.geometries`, ADR 0058 — multiple deform states / offsets / stage
  pins side-by-side). Read for anything reading back solver results or plotting.
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
- **`references/workflows.md`** — end-to-end patterns: single-session,
  multi-part assembly, solid–frame coupling, pushover, staged SSI. Read when
  the user asks for a complete example or a workflow they haven't built.
- **`references/gotchas.md`** — the ❌→✅ anti-patterns list plus the subtle
  pitfalls that aren't obvious from the API (unit-dependent `remove_duplicates`
  tolerance, half-open `in_box`, Selection-v2 ADR-0017 gaps). Read when a build
  "should work" but doesn't, or before writing constraint/selection/Results
  code from memory.
- **`references/ladruno.md`** — targeting the **Ladruno fork** of OpenSees
  (`nmorabowen/OpenSees@ladruno`): `OpenSeesTarget` (pin which build runs)
  vs `ops.capabilities()` (what it can do), the live backend resolver (now
  fork-preferring), fork-only Bézier elements, ExplicitBathe + SMS integrators,
  the Ladruno material / beam-column / analysis clusters, contact/embed handlers,
  EnergyBalance + `.ladruno` recorder, the `≥33000` class-tag band. Read **only**
  when wiring fork-specific emit/read or pinning a build; stock `openseespy`
  stays first-class.

If the user asks to modify the library itself (not just use it), also skim
`internal_docs/guide_*.md` in the project — they are the authoritative
user-facing docs and (mostly) match current source. When in doubt, the
`CHANGELOG.md` v2.0.0 section + Unreleased block is the source of truth.

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
> → `.add(...).couple(...).materialize()` (shipped v2.0.0, PR #433). It's a
> thin wrapper that *produces* a composed session; see `references/compose.md`.
> For everything else, build multi-part models with `g.compose(...)` /
> `apeGmsh.from_h5`.

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
these** — it double-constrains the model. (Shipped v2.0.0, ADR 0022.)

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
7. **`results.viewer()` kills the Jupyter kernel** — blocking VTK+Qt is the
   default. In notebooks use `results.show_web()` or
   `results.viewer(blocking=False)`. See `references/results.md`.

Which reference covers a given failure: `BridgeError` / staged / ndf →
`opensees-bridge.md`; `MalformedH5Error` / `SchemaVersionError` →
`fem-broker.md`; `LineageError` / viewer crash → `results.md`;
`MeshRecipeError` / selection → `api-cheatsheet.md`; `GeometryValidationError`
→ `gotchas.md`.

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
- **Two standalone modules** (separate imports, NOT session composites):
  `from apeGmsh.hpc import Cluster, Job` (remote SLURM, pairs with
  `ops.run_remote`) and `from apeGmsh.sensitivity import Sensitivity` (FD
  gradient/calibration). Details in `api-cheatsheet.md`.
- **Schema constants** live in `fem-broker.md` (neutral + bridge zones, ADR
  0023) and the viewer session schema in `results.md` — they drift, so read the
  number from there, not from memory.

Before claiming a method or signature exists, confirm it in `src/apeGmsh/`;
`references/api-cheatsheet.md` indexes the public surface.
