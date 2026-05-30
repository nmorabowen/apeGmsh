# Importing CAD and meshes — STEP, IGES, and `.msh`

A practical guide to getting external geometry and external meshes into
apeGmsh. Three sources are covered:

- **STEP** (`.step`, `.stp`) — the industry-standard CAD exchange format
- **IGES** (`.iges`, `.igs`) — the older CAD exchange format
- **MSH** (`.msh`) — Gmsh's native format, for *already-meshed* models

STEP and IGES go through the OpenCASCADE kernel and land in the session
as **geometry** — you still have to define physical groups and mesh
them. MSH is different: it already contains nodes, elements, and
(usually) physical groups, and the apeGmsh pipeline treats it as a
direct path into the **FEM broker**, skipping geometry and meshing
entirely.

All snippets assume `from apeGmsh import apeGmsh, MshLoader`.

## Tasks on this page

- [Import geometry from STEP / IGES](#11-the-one-line-call) · [Heal broken CAD](#14-heal-broken-cad) · [Diagnose import health](#15-diagnosing-import-health-diagnose) · [Mesh imported CAD end to end](#16-the-full-cad-mesh-solver-path)
- [Export back to STEP / IGES](#2-exporting-back-to-step-iges) · [Round-trip a `.msh` in a live session](#31-round-tripping-modelload_msh-modelsave_msh) · [Extract a broker from `.msh` headlessly](#32-headless-extraction-mshloaderload)

For the focused recipe, see [How-to: import a STEP file](../how-to/import-step.md).


## 1. STEP and IGES — importing geometry

### 1.1 The one-line call

Both formats share a single code path inside the model
(`_model_io._import_shapes`), which wraps
`gmsh.model.occ.importShapes`, synchronises the OCC kernel, and
registers every imported entity in `Model._metadata` so that later
boolean ops, transforms, and queries can address it.

```python
g = apeGmsh(model_name="bracket")
g.begin()

imported = g.model.io.load_step("bracket.step")
# imported == {3: [1], 2: [6, 7, 8, ...], 1: [...], 0: [...]}   (if highest_dim_only=False)
# imported == {3: [1]}                                          (default)

bodies = imported[3]      # all imported volume tags
```

`load_iges` has the exact same signature and return shape. Everything
said in this section about STEP applies equally to IGES — only the file
format differs.

### 1.2 `highest_dim_only` — what actually comes back

The OCC importer returns every sub-entity (faces, edges, vertices) of
the imported shape by default, which is almost never what you want for
structural work — it pollutes the registry with hundreds of low-dim tags
you will never reference. apeGmsh flips this on its head: the default is
`highest_dim_only=True`, meaning the returned dict only contains the
top-dimensional entities.

```python
# Default — clean: only volumes for a solid model
imported = g.model.io.load_step("part.step")
assert list(imported.keys()) == [3]

# Everything — useful if you need to tag specific faces / edges
imported = g.model.io.load_step("part.step", highest_dim_only=False)
volumes  = imported[3]
faces    = imported[2]     # now addressable for physical groups
edges    = imported[1]
```

Use `highest_dim_only=False` when the next step is *"assign a boundary
condition to that specific face of the imported CAD"*. Otherwise keep
the default and discover faces with `getBoundary` / `getEntitiesInBoundingBox`
when you actually need them.

### 1.3 `sync` — when to skip synchronisation

Every apeGmsh load method takes `sync=True` as the default and calls
`gmsh.model.occ.synchronize()` before returning. You can turn this off
when you are about to chain several kernel-level calls and want to pay
the synchronisation cost only once:

```python
imp1 = g.model.io.load_step("plate.step",  sync=False)
imp2 = g.model.io.load_step("rib.step",    sync=False)
g.model.transforms.translate(imp2[3], 0, 0, 50, sync=False)
g.model.boolean.fuse(imp1[3] + imp2[3])         # this call synchronises
```

For single imports, leave `sync=True`. The cost of an extra
`synchronize()` on a clean CAD load is negligible compared to the
import itself.

### 1.4 Heal broken CAD

STEP and IGES files from the wild are rarely clean. Legacy CAD
exporters produce:

- Faces that don't quite meet at edges (small gaps)
- Edges shorter than the meshing tolerance
- Open shells that should be closed solids
- Degenerate triangular faces collapsed to a line

Gmsh's OCC kernel exposes `occ.healShapes` for exactly this, and apeGmsh
wraps it as `Model.heal_shapes`:

```python
imported = g.model.io.load_step("legacy_part.step")

g.model.io.heal_shapes(
    tolerance       = 1e-3,     # aggressive — tune to your model size
    fix_degenerated = True,
    fix_small_edges = True,
    fix_small_faces = True,
    sew_faces       = True,     # close gaps between adjacent faces
    make_solids     = True,     # promote healed shells into solids
)
```

When called without a `tags` argument, `heal_shapes` heals **every**
entity in the model. Passing explicit tags restricts the healing to a
subset — useful when only one imported part is broken and you don't
want to touch the rest.

**The tolerance is the knob that matters.** Start with the default
(`1e-8`), and only increase it if meshing fails at the broken-edge
locations. A tolerance much larger than the smallest genuine feature
in the part will merge features you wanted to keep — always verify
visually with `g.inspect.show()` after aggressive healing.

#### Healing at import time — `heal=` and `dedupe=`

For the common case of "import and clean in one shot", `load_step`,
`load_iges`, and `g.parts.import_step` all accept a `heal=` keyword that
runs `heal_shapes` on the freshly-imported entities before returning:

```python
# scale-aware: derives a tolerance from the model bbox (≈ 1e-6 · diagonal)
imported = g.model.io.load_step("legacy_part.step", heal="auto")

# heal=True is identical to heal="auto"
imported = g.model.io.load_step("legacy_part.step", heal=True)

# a float pins an absolute tolerance, exactly like heal_shapes(tolerance=...)
imported = g.model.io.load_step("legacy_part.step", heal=1e-3)
```

`heal="auto"` (and the equivalent `heal=True`) is the recommended
default for messy CAD: a fixed absolute tolerance is meaningless across
unit systems (a `1e-3` mm gap and a `1e-3` m gap are wildly different),
so apeGmsh derives `≈ 1e-6 · bbox_diagonal` from the loaded geometry.
For the non-tolerance knobs (`sew_faces`, `make_solids`, …), call
`heal_shapes()` directly after a raw import.

`dedupe=` runs `g.model.queries.remove_duplicates` after the import (and
after healing, when both are set) to merge coincident entities that came
in as separate copies — common when a STEP assembly repeats shared
faces across bodies:

```python
imported = g.model.io.load_step("assembly.step", heal="auto", dedupe=True)
```

`dedupe=True` uses the current Gmsh tolerance; a float overrides it for
the call. (`g.parts.import_step` exposes the same `heal=` / `dedupe=`
pair.)

### 1.5 Diagnosing import health — `diagnose()`

When you import *raw* (no `heal=`) and the result contains slivers —
edges or faces far below the model scale — apeGmsh emits a non-mutating
`WarnGeomImportHealth` advisory so the problem doesn't surface later as
a cryptic meshing failure. To inspect the geometry on demand without
re-importing, call `diagnose()`:

```python
g.model.io.load_step("messy.step")          # raw — advisory may fire

report = g.model.io.diagnose()               # -> ImportHealth (non-mutating)
print(report)
# ImportHealth(solids=1, dims={0: 24, 1: 38, 2: 14, 3: 1},
#              highest_dim=3, short_edges=3, tiny_faces=1,
#              bbox_diag=412.3, suggested_tolerance=4.12e-04)

if report.is_suspect:                        # True iff slivers present
    # re-import with the suggested scale-aware healing
    imported = g.model.io.load_step("messy.step", heal="auto", dedupe=True)
    body = imported[3][0]
```

`diagnose()` scans the live OCC geometry and returns a frozen
`ImportHealth` carrying:

| Field / property        | Meaning                                                    |
|-------------------------|------------------------------------------------------------|
| `dim_counts`            | `{dim: count}` of entities currently in the model          |
| `highest_dim`           | top dimension present (`3` solids, `2` shells, …)          |
| `bbox_diag`             | model bounding-box diagonal (the scale)                    |
| `short_edges`           | tuple of edge tags shorter than `1e-4 · diag`              |
| `tiny_faces`            | tuple of face tags far below the model scale               |
| `suggested_tolerance`   | the `heal="auto"` tolerance for this model (`≈ 1e-6 · diag`)|
| `n_solids` *(property)* | shortcut for `dim_counts.get(3, 0)`                        |
| `is_suspect` *(property)* | `True` when `short_edges` or `tiny_faces` is non-empty   |

`is_suspect` keys off **slivers only** — a surface-only import
(`highest_dim < 3`) is *not* flagged on its own, because shell models
import that way on purpose and shouldn't trip a false advisory.

`diagnose()` never heals, dedupes, or renumbers — it is the
look-before-you-leap counterpart to `heal_shapes` (which *does* mutate).
Pass `warn=True` to have it emit the `WarnGeomImportHealth` advisory
itself when the report is suspect; `load_step` / `load_iges` use that
internally on a raw import. To silence or catch the advisory, filter on
the category:

```python
import warnings
from apeGmsh.core._geometry_errors import WarnGeomImportHealth

with warnings.catch_warnings():
    warnings.simplefilter("ignore", WarnGeomImportHealth)
    g.model.io.load_step("known_messy.step")
```

### 1.6 The full CAD → mesh → solver path

Putting it all together, a typical import-and-mesh workflow for a STEP
file looks like this:

```python
g = apeGmsh(model_name="bracket")
g.begin()

# 1. Import and heal
imported = g.model.io.load_step("bracket.step")
g.model.io.heal_shapes(tolerance=1e-4)

bodies = imported[3]

# 2. Name the things the solver will care about
#    (load_step doesn't create physical groups — they're yours to design)
g.physical.add(3, bodies, name="Steel")

# Discover faces by bounding-box query (see _model_queries.py)
base_faces = [t for (d, t) in g.model.queries.entities_in_bounding_box(
    -1e3, -1e3, -1e-3, 1e3, 1e3, 1e-3, dim=2)]
top_faces  = [t for (d, t) in g.model.queries.entities_in_bounding_box(
    -1e3, -1e3, 99.999, 1e3, 1e3, 100.001, dim=2)]

g.physical.add(2, base_faces, name="Fixed_Support")
g.physical.add(2, top_faces,  name="Top_Load")

# 3. Size and mesh
g.mesh.sizing.set_size_global(5.0)
g.mesh.generation.generate(dim=3)

# 4. Hand the snapshot to the solver
fem = g.mesh.queries.get_fem_data(dim=3)

g.end()
```

The important distinction: `load_step` only gives you **geometry and a
registry of tags**. Physical groups, mesh sizing, and meshing are still
your job, exactly as if you had built the geometry inside apeGmsh. STEP
import is not a shortcut to a solver-ready model — it is a shortcut to
a solver-ready *geometry*.

### 1.7 Multi-part STEP assemblies

A STEP file can contain multiple bodies (assemblies). `load_step`
returns all of them under the same `imported[3]` list:

```python
imported = g.model.io.load_step("assembly.step")
parts    = imported[3]                        # [1, 2, 3, 4, ...]

# Assign each body a distinct material physical group
for i, tag in enumerate(parts, start=1):
    g.physical.add(3, [tag], name=f"Body_{i}")
```

If the bodies share faces and you need a *conformal* mesh across the
interface, you must still call `g.model.boolean.fragment(parts)` (or `fuse`,
depending on whether you want the interface preserved). STEP import
alone does not imply conformality — bodies come in as independent
solids.


## 2. Exporting back to STEP / IGES

The reverse direction is trivial because Gmsh handles it natively:

```python
g.model.io.save_step("rebuilt.step")   # .step added if missing
g.model.io.save_iges("rebuilt.iges")
```

These write whatever geometry is currently in the model. They are
useful for checkpointing a mid-construction state, or for sending a
cleaned-up part back to a CAD tool after healing and boolean operations
in apeGmsh.

Note: STEP/IGES export does **not** carry mesh, physical groups, or
constraints. If you want the whole session state on disk, use
`save_msh` (Section 3.1).


## 3. MSH — the mesh-as-first-class-citizen path

MSH is Gmsh's native format, and unlike STEP/IGES it preserves
**everything**: geometry, mesh nodes and elements, physical groups,
element tags, and partition information. For apeGmsh this means an
imported `.msh` can go straight to the FEM broker without any
remeshing — the geometry is just there to keep Gmsh happy.

There are two entry points, and they serve genuinely different needs.

### 3.1 Round-tripping: `Model.load_msh` / `Model.save_msh`

When you already hold an active apeGmsh session and want to bring in a
previously-saved mesh, use the model-level helpers:

```python
g = apeGmsh(model_name="imported")
g.begin()

g.model.io.load_msh("previous_run.msh")
# Geometry + mesh + physical groups are now live in the session.
# Everything the composites (g.physical, g.inspect, g.mesh) offer works.

fem = g.mesh.queries.get_fem_data(dim=3)
g.end()
```

Under the hood `load_msh` calls `gmsh.merge()`, which is how Gmsh
imports any file it recognises. `save_msh` is symmetric: it sets
`Mesh.SaveAll = 1` and writes the file, so nothing gets silently
dropped.

Use this path when you want a **live** session: you plan to plot the
imported mesh, inspect physical groups, or re-mesh parts of it. The
composites stay available because the session is still open.

### 3.2 Headless extraction: `MshLoader.load`

The second path — and the more common one for *production* pipelines —
is `MshLoader.load`. It is a classmethod, takes a path, and returns a
`FEMData` object directly. No session, no `begin()`/`end()`, no Gmsh
state left behind:

```python
from apeGmsh import MshLoader

fem = MshLoader.load("bridge.msh", dim=2)

print(fem.info)                 # n_nodes, n_elems, bandwidth
print(fem.physical.summary())   # what's in the file
```

Internally it initialises Gmsh, creates an anonymous model, merges the
file, builds a `FEMData` snapshot via `_fem_extract.build_fem_data`,
and finalises Gmsh — all inside a `try/finally`. The returned object
is **completely decoupled** from Gmsh: you can pickle it, pass it to
another process, close the Python interpreter and reopen it later —
none of that requires Gmsh to be running.

This is the path to use when:

- You only care about the mesh data, not the geometry or the live
  Gmsh model
- You are chaining `.msh` files into a solver pipeline from a script
  or notebook that doesn't need apeGmsh's other composites
- You want to run a study over many `.msh` files and the session
  boilerplate gets in the way

### 3.3 The `dim` parameter — what it selects

Both `MshLoader.load` and `g.loader.from_msh` take a `dim` argument
with a default of `2`. This is **the element dimension to extract into
the broker** — not the dimension of the model as a whole.

| `dim` | What ends up in `fem.connectivity`                  |
|-------|-----------------------------------------------------|
| `1`   | Lines (truss, beam, frame)                          |
| `2`   | Triangles and quads (plane stress, plane strain, shells) |
| `3`   | Tetrahedra and hexahedra (3D solids)                |

The broker is homogeneous in element dimension: one `FEMData` holds
elements of one dimension only. If your `.msh` file contains both 2D
shells and 3D solids (common in structural models with a shell cladding
on a solid core), call the loader twice — once with `dim=2`, once with
`dim=3` — and keep both snapshots:

```python
shell_fem = MshLoader.load("mixed.msh", dim=2)
solid_fem = MshLoader.load("mixed.msh", dim=3)
```

Each broker is then responsible for its own solver emission. This is
deliberate and follows the same *source-agnostic broker* principle the
project stands on: `FEMData` describes a uniform FE discretisation, not
a Gmsh model.

### 3.4 Composite loading into a live session

The third option is `g.loader.from_msh`, which exists when you want
both worlds: load a `.msh` into an active session *and* also get the
`FEMData` back in one call.

```python
g = apeGmsh(model_name="imported")
g.begin()

fem = g.loader.from_msh("bridge.msh", dim=2)

# After the call:
#   - the mesh is merged into the active session, so
#     g.physical.get_all(), g.inspect.show(), etc. still work
#   - `fem` is a ready-to-use broker snapshot

g.end()
```

The difference from `Model.load_msh` is that `g.loader.from_msh` calls
`build_fem_data` eagerly and returns the broker in the same call,
whereas `Model.load_msh` just merges and leaves the broker extraction
to a later `g.mesh.queries.get_fem_data(dim=...)`. Functionally they are
equivalent; pick whichever reads better at the call site.

### 3.5 Verbose output — what the loader tells you

Both loaders accept `verbose=True` (standalone) or inherit it from the
session (`g._verbose`). When on, they print a one-line summary plus a
list of physical groups found in the file:

```text
[MshLoader] load('bridge.msh') → 12847 nodes, 24116 elements, bw=431
[MshLoader] physical groups (4): (2,1) 'Deck', (2,2) 'Piers',
                                  (1,3) 'Edge_Restraint', (1,4) 'Traffic_Load'
```

The bandwidth reported here comes straight from `FEMData.info`; it is
not a freshly computed RCM bandwidth. If you care about the optimal
bandwidth, build a `Numberer(fem)` and renumber — see
`guide_fem_broker.md` for that workflow.


## 4. Picking the right entry point

Three import paths sound like a lot. The decision tree is actually
short:

- **Geometry-only CAD file (STEP / IGES)** → `Model.load_step` /
  `Model.load_iges`. You still have to define physical groups and mesh.
- **Previously meshed file, live session wanted** (plot, inspect,
  re-mesh parts) → `Model.load_msh` or `g.loader.from_msh`.
- **Previously meshed file, broker only** (headless pipeline,
  notebook, pickle-and-ship) → `MshLoader.load`.

A useful mental model: STEP/IGES hand you geometry that is still
*mutable* inside Gmsh. MSH hands you a *snapshot* that is ready to
become a solver model, and the only question is whether you also want
the live Gmsh session around it.


## 5. Common pitfalls

**Forgetting `heal_shapes` on legacy STEP.** Symptom: meshing fails
with cryptic errors about degenerate edges, or the mesh has holes at
the seams between faces. Fix: call `heal_shapes` with a tolerance
matched to the smallest genuine feature in the part.

**Relying on STEP physical groups.** STEP and IGES don't have
physical groups — they only carry geometry. After `load_step` you
always have to define your own. If you need named regions to carry
across from the CAD tool, use DXF (via `load_dxf`, which maps layers
→ physical groups) or export to MSH from Gmsh itself.

**Mixing the OCC and geo kernels after a CAD import.** `load_step`
and `load_iges` use the OCC kernel. If the model also has geo-kernel
entities (e.g., from an earlier `addPoint` call that forgot to
specify `kernel="occ"`), boolean operations between the two will
fail. Stick to one kernel per session.

**Calling `MshLoader.load` from inside an active session.** It
initialises and finalises Gmsh internally, which will tear down your
session. Use `g.loader.from_msh` instead when a session is live.

**Extracting the wrong `dim` from a mixed mesh.** Asking for `dim=3`
on a file that only contains shells returns a broker with zero
elements — the empty result is not an error, because a file can
legitimately contain only lower-dimensional elements. Verify
`fem.info.n_elems > 0` before handing the broker to a solver, or
check `fem.physical.summary()` to see what is actually in the file.


## 6. Reference: the load/save surface area

All import/export lives in `core/_model_io.py` and
`mesh/MshLoader.py`. The public surface is intentionally small:

| Call                            | Direction | Returns                        |
|---------------------------------|-----------|--------------------------------|
| `g.model.io.load_step(path)`       | in        | `{dim: [tag,...]}`             |
| `g.model.io.load_iges(path)`       | in        | `{dim: [tag,...]}`             |
| `g.model.io.heal_shapes(...)`      | —         | `self` (chainable)             |
| `g.model.io.diagnose(warn=False)`  | —         | `ImportHealth` (non-mutating)  |
| `g.model.io.save_step(path)`       | out       | `None`                         |
| `g.model.io.save_iges(path)`       | out       | `None`                         |
| `g.model.io.load_msh(path)`        | in        | `{dim: [tag,...]}` (entities)  |
| `g.model.io.save_msh(path)`        | out       | `None`                         |
| `MshLoader.load(path, dim=...)` | in        | `FEMData`                      |
| `g.loader.from_msh(path, dim=)` | in        | `FEMData` (+ live session)     |

Everything else is composition: once geometry is in the session you
use the rest of apeGmsh the same way you would with natively-built
geometry; once a broker is in hand you use it the same way as any
other `FEMData` — see `guide_fem_broker.md`.


??? note "For maintainers — source map"

    - `src/apeGmsh/core/_model_io.py` — `load_iges`, `load_step`,
      `heal_shapes`, `diagnose` / `ImportHealth`, `load_msh`, `save_*`
    - `src/apeGmsh/mesh/MshLoader.py` — standalone and composite `.msh`
      loader
    - `src/apeGmsh/mesh/_fem_extract.py` — the broker builder used by both
      paths
