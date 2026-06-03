---
name: apegmsh-helper
description: Use whenever the user is working with apeGmsh — the structural-FEM wrapper around Gmsh with OpenSees integration. Triggers on building FEM models from CAD/STEP imports, Part-based assembly workflows, composite-based geometry/mesh/constraint APIs (g.model, g.mesh, g.physical, g.constraints, etc.), the apeSees(fem) OpenSees bridge with typed primitives and automatic MP-constraint emission, staged analysis (ops.stage), loads/masses/constraints resolution into the FEMData broker, native model.h5 persistence (FEMData.to_h5/from_h5, save_to=/g.save(), apeGmsh.from_h5), model composition (g.compose), post-processing OpenSees output via Results (from_native/from_mpco/from_recorders) with the interactive and web viewers (results.viewer / results.show_web), and exporting models to OpenSees Tcl or openseespy scripts. Covers apeGmsh's own abstractions on top of Gmsh and OpenSees. For raw gmsh API questions see the gmsh-structural skill; for raw OpenSees analysis commands see opensees-expert; for FEM theory first principles see fem-mechanics-expert.
---

# apeGmsh — structural FEM wrapper around Gmsh

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

## Before writing code: read the right reference

The library is big. Don't try to remember every composite — read the
reference file that matches the task, *then* write the code. The
references are tight; reading them is cheap.

- **`references/api-cheatsheet.md`** — one-page map of every session
  composite (`g.model.*`, `g.mesh.*`, `g.parts`, `g.loads`, `g.masses`,
  `g.constraints`, `g.node_ndf`, `g.physical`, `g.labels`,
  `g.mesh_selection`) plus the post-session `apeSees(fem)` bridge, and the
  methods on each. **Read this first** for any non-trivial apeGmsh task —
  it saves you from guessing signatures.
- **`references/fem-broker.md`** — deep dive on `FEMData`, the broker
  returned by `g.mesh.queries.get_fem_data(dim=...)`, **plus native
  persistence** (`FEMData.to_h5` / `from_h5`, `save_to=` / `g.save()`,
  schema constants, integrity checks). Read when the task touches nodes,
  elements, iteration, solver hand-off, or saving/reloading a model.
- **`references/opensees-bridge.md`** — the `apeSees(fem)` bridge:
  typed-primitive materials/sections/elements, explicit `ops.fix`/`ops.mass`/
  `ops.pattern`, **automatic MP-constraint emission**, **staged analysis**
  (`ops.stage(...)` + `s.*` verbs), **per-node ndf** wiring,
  `ops.tcl/py/h5/run`, and **which OpenSees runs** (`OpenSeesTarget` /
  `ops.capabilities()`). Read this for any OpenSees generation task.
- **`references/results.md`** — `Results` post-processing of OpenSees output
  (`from_native` / `from_mpco` / `from_recorders`, all of which now
  **require `model=` / `model_h5=`**), the `results.model.fem` broker chain,
  `results.lineage`, and the **web viewers** (`show_web` / `serve_web`,
  kernel-safe). Read for anything reading back solver results or plotting.
- **`references/compose.md`** — model composition: `g.compose(...)`,
  `apeGmsh.from_h5(...)` chain-phase sessions, anchors vs translate, nested
  compose, and the string-keyed `'Module'` viewer color modes. Read when
  assembling several saved `model.h5` modules into one model.
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
  vs `ops.capabilities()` (what it can do), fork-only BezierTri6,
  ExplicitBathe integrators, EnergyBalance + `.ladruno` recorder, the
  `≥33000` class-tag band. Read **only** when wiring fork-specific
  emit/read or pinning a build; stock `openseespy` stays first-class.

If the user asks to modify the library itself (not just use it), also skim
`internal_docs/guide_*.md` in the project — they are the authoritative
user-facing docs and (mostly) match current source. When in doubt, the
`CHANGELOG.md` v2.0.0 section + Unreleased block is the source of truth.

## Mental model

Four concepts, in this order:

**1. A session (`g`) owns a single Gmsh kernel.** Open it with
`g.begin()` / `g.end()` or a `with apeGmsh(...) as g:` block. Every
composite — `g.model`, `g.mesh`, `g.loads`, `g.masses`, `g.constraints`,
`g.node_ndf`, etc. — is a thin namespace that talks to that shared kernel.
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
larger assemblies, and every solver bridge consumes it.

**4. MP constraints + per-node ndf now emit automatically.** When the
snapshot carries multi-point constraints (`fem.nodes.constraints`,
`fem.elements.constraints`) the `apeSees(fem)` bridge auto-emits the matching
`equalDOF` / `rigidLink` / `rigidDiaphragm` / `ASDEmbeddedNodeElement` deck
lines (and an `ops.constraints.Transformation()` handler when present), and
per-node ndf set via `g.node_ndf` is wired into the deck. **Do not hand-emit
these** — that double-constrains the model. *(This reverses the old
"MP constraints are deferred" claim, which is stale as of v2.0.0.)*

## Core workflow

Every apeGmsh script follows the same skeleton. Learn this, and everything
else is filling in blanks:

```python
# verified: tests/test_femdata_from_h5.py::test_session_save_then_from_h5
#           tests/opensees/integration/test_runnable_deck.py::test_tcl_deck_contains_constraint_lines
from apeGmsh import apeGmsh

with apeGmsh(model_name="my_model", save_to="my_model.h5") as g:
    # 1. GEOMETRY — occ kernel, via g.model sub-composites
    g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")

    # 2. PHYSICAL GROUPS — bridge labels to the solver (dim-unique names)
    g.physical.add_volume("body", name="Body")
    # ...surface PGs come from labels/queries, never raw entity tags.

    # 3. LOADS / MASSES / CONSTRAINTS — pre-mesh, reference labels/PGs
    with g.loads.case("dead"):
        g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)
    g.masses.volume("Body", density=2400)

    # 4. MESH
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)

    # 5. SNAPSHOT — the solver contract (also autosaved to my_model.h5 on exit)
    fem = g.mesh.queries.get_fem_data(dim=3)
    print(fem.info)        # "N nodes, M elements, bandwidth=..."

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
   `fem.elements.get(element_type="tet4").resolve()`.
5. **Labels without physical groups** — labels on the main session don't
   auto-promote to PGs (only `Part` sessions do). Call
   `g.labels.promote_to_physical("name")` or add `g.physical.add(...)`.
6. **`Results.from_native(...)` raises `TypeError`** — the constructor now
   *requires* `model=` (`from_mpco` requires `model_h5=`). See
   `references/results.md`.
7. **`results.viewer()` kills the Jupyter kernel** — blocking VTK+Qt is the
   default. In notebooks use `results.show_web()` or
   `results.viewer(blocking=False)`. See `references/results.md`.

## Version & layout facts you can rely on

- **Current version is v2.0.0** — `pyproject.toml` and the latest tagged
  `CHANGELOG.md` section agree. (A stale editable install may still print
  `v1.6.0` in the import banner until it is reinstalled — trust the source,
  not the banner.) v2.0.0 shipped the three-broker chain
  `FEMData ⊂ OpenSeesModel ⊂ Results`, automatic MP-constraint emission,
  per-zone schema versioning, and deleted `BindError`. Anything claiming
  "v1.0" is stale.
- **`g.masses`, not `g.mass`.** The `g.mass → g.masses` rename shipped long
  ago. Legacy v0.x forms (`g.model.add_point`, `g.model.fuse`,
  `g.initialize()`/`g.finalize()`, `g.opensees`) are **removed** — use the
  sub-composite forms (`g.model.geometry.add_point`, `g.model.boolean.fuse`,
  `g.begin`/`g.end`) and the post-session `apeSees(fem)` bridge.
- **`g.node_ndf`** is a real top-level composite (`set_default(ndf=)`,
  `set(target, ndf=)`, `list()`, `clear()`) for mixed-ndf models
  (shell-on-solid); `fem.nodes.ndf_for(nid)` fails loud on an unset node.
- **Schema constants** (two independent zones, ADR 0023): neutral
  `NEUTRAL_SCHEMA_VERSION = "2.10.0"`; bridge `SCHEMA_VERSION = "2.12.0"`.
  Readers accept only their own minor and one below.
- The source is strictly typed under `src/apeGmsh/`. The standalone viewer
  under `apeGmshViewer/` is a Qt/PyVista app and is deliberately **not**
  strictly typed — don't apply library-level type discipline there.

Before claiming a method or signature exists, confirm it in `src/apeGmsh/`;
`references/api-cheatsheet.md` indexes the public surface.
