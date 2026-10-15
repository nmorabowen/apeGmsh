---
name: apegmsh-helper
description: >
  How to use apeGmsh, the Gmsh wrapper for structural FEM: labelled geometry
  and meshes, a FEMData snapshot saved to model.h5, a typed OpenSees bridge
  (apeSees), and Results readers and viewers. Use it to write, run, debug or
  review a user model script that imports apeGmsh. TRIGGER: from apeGmsh
  import apeGmsh / Part / FEMData / Results; g.model, g.mesh, g.physical,
  labels and selections; loads, masses and constraints; Part and Assembly
  multi-part models; get_fem_data and model.h5; apeSees(fem), stages and the
  emitted Tcl or openseespy deck; reading OpenSees output with Results.from_*;
  the apeGmsh Studio CLI or MCP. SKIP: changing apeGmsh itself (its source,
  tests, ADRs, docs or this skill; use the AGENTS.md task guides:
  apegmsh-bridge-feature, apegmsh-viewer-results, apegmsh-adr-docs); the raw
  gmsh API or .geo files (gmsh-structural); OpenSees Tcl, openseespy or C++
  work without apeGmsh (opensees-expert, opensees-ladruno); design-code
  checks.
---

# apeGmsh: structural FEM on top of Gmsh

**Humans:** start with the docs learning path
(<https://nmorabowen.github.io/apeGmsh/tutorials/learning-path/>), then read
`references/model-scripts.md`, whose reference scripts run in CI.

## Mental model

1. A session `g` owns one Gmsh kernel; its composites (`g.model`, `g.mesh`, `g.physical`, `g.loads`, `g.masses`, `g.constraints`, ...) split by concern into sub-composites.
2. You name geometry with labels and physical groups, and declare loads, masses and constraints on those names before meshing.
3. `g.mesh.queries.get_fem_data(dim=...)` returns `FEMData`: an immutable, solver-neutral snapshot that saves to `model.h5` and needs no live Gmsh.
4. `apeSees(fem)` is the OpenSees bridge, used after the session: typed primitives, with MP constraints and per-node ndf emitted for you; it writes a Tcl, Python or H5 deck, or runs one.
5. `Results.from_*` reads the output back against the model for checks, stills and viewers.

## The skeleton

Every script has this shape. It runs as written (CI executes it); copy it and
change the data, the geometry and the names.

<!-- reference-script: skeleton.py -->
```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# DATA (SI: m, N, kg, s)
LENGTH = 10.0  # m, cantilever span along x
WIDTH = 5.0  # m
DEPTH = 2.0  # m
DENSITY = 2400.0  # kg/m^3
GRAVITY = (0.0, 0.0, -9.81)  # m/s^2
YOUNG = 30e9  # Pa
POISSON = 0.2  # -
MESH_SIZE = 0.5  # m
X_AXIS = (1.0, 0.0, 0.0)  # normal of both end faces
FACE_TOL = 1e-6  # m
TIP_NODE_FORCE = (0.0, 0.0, -5e4)  # N, at every Tip node

with apeGmsh(model_name="my_model", save_to="my_model.h5") as g:
    # GEOMETRY (occ) → PHYSICAL GROUPS (dim-unique names) → pre-mesh
    # LOADS/MASSES/CONSTRAINTS (on labels/PGs) → MESH → SNAPSHOT (workflows.md §1).
    g.model.geometry.add_box(0.0, 0.0, 0.0, LENGTH, WIDTH, DEPTH, label="body")
    g.physical.add_volume("body", name="Body")
    g.model.select(None, dim=2).on_plane((0.0, 0.0, 0.0), X_AXIS, tol=FACE_TOL).to_physical("Base")
    g.model.select(None, dim=2).on_plane((LENGTH, 0.0, 0.0), X_AXIS, tol=FACE_TOL).to_physical("Tip")
    with g.loads.case("dead"):
        g.loads.gravity("Body", g=GRAVITY, density=DENSITY)
    g.masses.volume("Body", density=DENSITY)
    g.mesh.sizing.set_global_size(MESH_SIZE)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)   # the solver contract; autosaved on exit
    print(fem.info)

assert fem.nodes.select(pg="Base").ids and fem.nodes.select(pg="Tip").ids

# OPENSEES (optional): post-session bridge. Loads are OPT-IN via p.from_model
# (ADR 0051); masses and fixities are restated; MP constraints + ndf auto-emit.
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=YOUNG, nu=POISSON)   # no rho: g.masses carries it
ops.element.FourNodeTetrahedron(pg="Body", material=conc)
ops.fix(pg="Base", dofs=(1, 1, 1))
ops.mass_from_model()
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("dead")
    p.load(pg="Tip", forces=TIP_NODE_FORCE)
ops.py("model.py")     # or ops.tcl / ops.h5 / ops.run
```

## Rules, each with its reason

1. **Open the session with `with apeGmsh(...) as g:`.** Each session holds a
   reference on the shared Gmsh process, and the block calls `g.end()` even
   when the run fails, so that reference is released and Gmsh can finalize.
2. **Name what you will refer to: a `label=` at creation, and physical groups
   with dimension-unique names.** OCC renumbers tags on every boolean, and the
   bridge's `pg=` selectors resolve by name, so a tag goes stale and a name
   shared by two dimensions is ambiguous. With no label to hand, promote a
   selection in one call, as the skeleton does for `Base` and `Tip`.
3. **Declare loads, masses and constraints on names before meshing, then take
   the snapshot.** They resolve onto the mesh inside `get_fem_data`; the
   snapshot is what every solver bridge reads, and it saves to `model.h5`.
4. **Call the full sub-composite path** (`g.mesh.generation.generate`,
   `g.model.geometry.add_box`). Parent composites carry no shortcut methods,
   so a shortened path raises `AttributeError`.
5. **On the bridge, restate supports, masses and loads**: `ops.fix(pg=...)`,
   `ops.mass_from_model()`, and `p.from_model(case)` inside a pattern. The
   bridge consumes model records on request only (ADR 0051), and emit warns
   `UnconsumedModelDefinitionWarning` when a support or mass record never
   reached the deck.
6. **Leave MP constraints and per-node ndf to the bridge.** `apeSees(fem)`
   emits the `equalDOF` / `rigidLink` / `rigidDiaphragm` / embedded-node lines
   and each node's ndf from the snapshot, and the contact and interface
   elements too; a hand-written copy constrains the model twice.
7. **Check a mesh or a solve with `assess()`, and look at stills, not
   windows.** `fem.assess()` and `results.assess(figures=True)` return a
   verdict text and PNG paths you can read; act on `severity="error"`. Qt and
   web viewers are for a human who asked to look, and an agent never captures
   the human's screen.
8. **Read output through a `Results.from_*` constructor that is given the
   model.** The reader binds each record to the model's names, so a check can
   ask for `pg="Tip"` instead of node numbers.
9. **Keep the script flat and its numbers named**, with units in the data
   block (`references/model-scripts.md` has the full rules). A reviewer reads
   it as an engineering procedure, and the lint below checks most of it.

## Before you return a script

- **Signatures:** run `python -m apeGmsh.studio.lookup SYMBOL` (or the MCP
  `lookup` tool) for any call this skill does not show. The index behind it
  is checked against the live code; prose and memory drift.
- **Lint:** run `python scripts/lint_model_script.py <script.py>` (beside this
  file; standard library only) and fix every finding. Rule R1 flags each
  symbol listed in `scripts/removed_symbols.json` and names its replacement.
- **Run it** and keep its asserts: exit code 0 with a hand check is the proof.

Reading `src/apeGmsh/` to write a model script is an index miss: use
`lookup` and the references. The source is for changing apeGmsh itself.

## Routing: one task, one file

Pick one row and read only that file (or its matching heading).

| Task | Read |
|---|---|
| Writing a model script: shape, rules, nearest reference script | `references/model-scripts.md` |
| An end-to-end pattern beyond the skeleton | `references/workflows.md` |
| A signature | `python -m apeGmsh.studio.lookup SYMBOL` |
| Map of the public `g.*` / `apeSees` surface | `references/api-cheatsheet.md` |
| `FEMData`, `model.h5`, schema numbers; `MalformedH5Error` | `references/fem-broker.md` |
| `apeSees`, stages, emit, ndf; `BridgeError` | `references/opensees-bridge.md` |
| Checking a mesh or a solve; finding codes | `references/assess.md` |
| `Results`, plots, stills, viewers; `LineageError` | `references/results.md` |
| Multi-part models, `Assembly`, `from_h5` | `references/compose.md` |
| Rebar cages (`g.rebar`) | `references/rebar.md` |
| ETABS or analytical-model import | `references/interop.md` |
| Section properties, fiber sections; `SectionMeshError` | `references/section-properties.md` |
| Ladruno fork primitives through `apeSees` | `references/ladruno.md` |
| An error, a trap, an anti-pattern, version and layout facts | `references/gotchas.md` |
| Studio CLI, MCP tools, habitats, the example library | `references/studio.md` |

Published docs: <https://nmorabowen.github.io/apeGmsh/>. `CHANGELOG.md`'s
Unreleased section is the moving truth between releases.
