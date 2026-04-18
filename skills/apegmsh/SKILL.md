---
name: apegmsh
description: >
  apeGmsh is a Gmsh wrapper for structural FEM with first-class OpenSees
  integration. Use this skill any time the user is writing, reading,
  debugging, or reviewing code that imports apeGmsh (``from apeGmsh
  import apeGmsh`` / ``Part``) or touches the apeGmsh project at
  ``C:\Users\nmora\Github\apeGmsh``. Trigger for: building an ``apeGmsh``
  session, composing geometry via ``g.model.geometry/boolean/transforms/
  io/queries``, meshing via ``g.mesh.generation/sizing/field/structured/
  editing/queries/partitioning``, labels, physical groups, FEMData
  broker (``g.mesh.queries.get_fem_data``), multi-part assemblies
  (``Part`` + ``g.parts``), loads / masses / constraints (``g.loads``,
  ``g.masses``, ``g.constraints``), the OpenSees bridge
  (``g.opensees.materials/elements/ingest/inspect/export``), and the
  ``apeGmshViewer`` post-processing app. Also use it when the user says
  "meshing", "FEA mesh", "structural mesh", or "OpenSees from gmsh"
  in a context where apeGmsh is the wrapper of choice — the user has
  built apeGmsh specifically to replace hand-rolled Gmsh + OpenSees
  glue code and always prefers the apeGmsh API over raw gmsh calls.
---

# apeGmsh — structural FEM wrapper around Gmsh

apeGmsh is the user's in-house Gmsh wrapper. It lives at
`C:\Users\nmora\Github\apeGmsh`, and the *core idea* is:

> Describe a model **once** — geometry, labels, loads, constraints, masses —
> then hand a solver-agnostic **snapshot** (`FEMData`) to any FEM solver.
> OpenSees gets a first-class bridge; anyone else reads the snapshot.

Whenever the user is working in this project or writing code against
apeGmsh, prefer the apeGmsh API over raw `gmsh.*` calls. Raw `gmsh`
calls still work (you're holding the same session), but the whole point
of the wrapper is that you don't have to write them.

## Before writing code: read the right reference

The library is big. Don't try to remember every composite — read the
reference file that matches the task, *then* write the code. The
references are tight; reading them is cheap.

- **`references/api-cheatsheet.md`** — one-page map of every composite
  (`g.model.*`, `g.mesh.*`, `g.opensees.*`, `g.parts`, `g.loads`,
  `g.masses`, `g.constraints`, `g.physical`, `g.labels`,
  `g.mesh_selection`) and the methods on each. Read this first for
  any non-trivial apeGmsh task — it will save you from guessing
  signatures.
- **`references/fem-broker.md`** — deep dive on `FEMData`, the broker
  object returned by `g.mesh.queries.get_fem_data(dim=...)`. Read this
  when the task touches nodes, elements, iteration, or solver hand-off.
- **`references/opensees-bridge.md`** — the `g.opensees.*` pipeline:
  materials, element assignment, `ingest.loads/masses`, `build()`,
  `export.tcl/py`. Read this for any OpenSees generation task.
- **`references/workflows.md`** — end-to-end patterns: single-session,
  multi-part assembly, solid-frame coupling, pushover. Read when the
  user asks for a complete example or a workflow they haven't built
  before.

If the user asks to modify the library itself (not just use it), also
skim `docs/guide_*.md` in the project — they are the authoritative
user-facing docs and match the current source.

## Mental model

Three concepts, in this order:

**1. A session (`g`) owns a single Gmsh kernel.** Open it with
`g.begin()` / `g.end()` or a `with apeGmsh(...) as g:` block. Every
composite — `g.model`, `g.mesh`, `g.opensees`, `g.loads`, etc. — is
a thin namespace that talks to that shared kernel. There is **no
separate Assembly class**. The session *is* the assembly.

**2. Composites split by concern.** `g.model` is further split into
`geometry / boolean / transforms / io / queries`. `g.mesh` splits
into `generation / sizing / field / structured / editing / queries /
partitioning`. `g.opensees` splits into `materials / elements /
ingest / inspect / export`. Every method you want has a home — you
rarely need to reach into `gmsh.*` directly.

**3. `FEMData` is the solver contract.** When the mesh is ready, call
`fem = g.mesh.queries.get_fem_data(dim=3)` (or `dim=2`, or `None` for
all dims). The returned `FEMData` is an immutable snapshot with
`fem.nodes`, `fem.elements`, `fem.info`, `fem.inspect`. It works
without a live Gmsh session, and solver bridges consume it.

## Core workflow

Every apeGmsh script follows the same skeleton. Learn this, and
everything else is filling in blanks:

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="my_model", verbose=True) as g:
    # 1. GEOMETRY — occ kernel, via g.model sub-composites
    box = g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="body")
    # ... or assemble from points / lines / surfaces

    # 2. PHYSICAL GROUPS — bridge to solver
    g.physical.add(3, [box], name="Body")
    # ...and a surface PG from tags you got from the model:
    # g.physical.add_surface([base_tag], name="Base")

    # 3. LOADS / MASSES / CONSTRAINTS — pre-mesh, reference labels/PGs
    with g.loads.pattern("dead"):
        g.loads.gravity("Body", g=(0, 0, -9.81), density=2400)
    g.masses.volume("Body", density=2400)

    # 4. MESH
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)

    # 5. SNAPSHOT — the solver contract
    fem = g.mesh.queries.get_fem_data(dim=3)
    print(fem.info)        # "N nodes, M elements, bandwidth=..."

    # 6. OPENSEES (optional)
    g.opensees.set_model(ndm=3, ndf=3)
    g.opensees.materials.add_nd_material("Conc", "ElasticIsotropic",
                                          E=30e9, nu=0.2, rho=2400)
    g.opensees.elements.assign("Body", "FourNodeTetrahedron",
                                material="Conc")
    g.opensees.elements.fix("Base", dofs=[1, 1, 1])
    g.opensees.ingest.loads(fem).masses(fem)
    g.opensees.build()
    g.opensees.export.py("model.py")
```

This is the **happy path**. When the user asks for something more
involved (multi-part assembly, coupled shell + frame, pushover, cyclic),
read `references/workflows.md` for the pattern — don't improvise.

## When things go wrong: the usual suspects

Most apeGmsh bugs are in one of these categories. Check here before
assuming a deeper problem:

1. **`RuntimeError: session is already open`** — you called `begin()`
   twice, or a previous notebook cell crashed between `begin()` and
   `end()` and left Gmsh initialized. Fix: call `gmsh.finalize()`
   manually, or restart the kernel. Prefer the `with` form to avoid
   this entirely.

2. **Physical group resolves to wrong dimension** — `g.physical.add(1,
   tags, name="Fixed")` and `g.physical.add(2, tags, name="Fixed")`
   are different groups. If the OpenSees bridge later complains about
   ambiguity, pass `dim=` to `elements.assign(...)` / `elements.fix(...)`.

3. **"No label, physical group, or part named X"** — target resolution
   order is `label → physical group → part label`. The name you used
   doesn't exist in any of the three. Inspect what's available with
   `g.labels.get_all()`, `g.physical.summary()`, `g.parts.labels()`.

4. **`fem.elements.connectivity` raises `TypeError`** — the mesh has
   multiple element types (e.g. tets + shells). Either iterate
   `for group in fem.elements:`, or filter with
   `fem.elements.get(element_type="tet4").resolve()`.

5. **Labels without physical groups on the assembly session** — labels
   created on the main `apeGmsh` session don't auto-promote to physical
   groups (only `Part` sessions do that). To make a label solver-visible,
   call `g.labels.promote_to_physical("name")` or add an explicit
   `g.physical.add(...)`.

6. **Orphan nodes in `FEMData`** — nodes on lower-dim entities (e.g.
   points, curves) show up in `fem.nodes.ids` even if your target is
   `dim=3`. Pass `remove_orphans=True` to `get_fem_data(...)` to drop
   them — nodes referenced by constraints/loads/masses are always kept.

7. **Mixing `begin()` / context-manager on the same `g`** — pick one.
   Context-manager is preferred in scripts; `begin()` / `end()` is for
   notebooks where the session must span cells.

## Version and layout facts you can rely on

The current version is **v1.0** (see `CHANGELOG.md`). The v1.0 refactor
split the Model and Mesh mixins into explicit sub-composites and
renamed `g.mass → g.masses`. Anything you see in older code —
`g.model.add_point(...)`, `g.model.fuse(...)`, `g.mass.*`,
`g.initialize()` / `g.finalize()` — is v0.x and has been removed. Use
the sub-composite forms (`g.model.geometry.add_point`,
`g.model.boolean.fuse`, `g.masses`, `g.begin` / `g.end`).

The source is strictly typed under `src/apeGmsh/`. The standalone
viewer under `apeGmshViewer/` is a Qt/PyVista app and is deliberately
not strictly typed — don't apply library-level type discipline there.

Before claiming that a specific method or signature exists, confirm it
in the source. The code is under `src/apeGmsh/`; the cheatsheet in
`references/api-cheatsheet.md` indexes the public surface.
