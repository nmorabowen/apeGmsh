# apeGmsh

[![Docs](https://github.com/nmorabowen/apeGmsh/actions/workflows/docs.yml/badge.svg)](https://nmorabowen.github.io/apeGmsh/)

Structural-FEM wrapper around [Gmsh](https://gmsh.info) with a composition-based API and a
snapshot FEM broker. Designed to make it cheap to describe a model
once (geometry + physical groups + loads + constraints) and feed it
to any solver. OpenSees has first-class support; other solvers can be
plugged in through the same `FEMData` contract.

**New to the library?** Start with the [**First steps guide**](internal_docs/first_steps.md) —
a conversational walk through the session model, naming system
(tags / labels / physical groups), queries and selection, booleans,
and CAD import.

**Documentation:** <https://nmorabowen.github.io/apeGmsh/>

> [!NOTE]
> **Built on Gmsh.** apeGmsh is a wrapper built on top of the (awesome)
> [Gmsh](https://gmsh.info) Python API. It adds a set of abstractions over the
> main API to fit an intended structural-FEM workflow — parts, constraints,
> loads, masses, and an OpenSees bridge. You still have the full Gmsh API
> underneath whenever you need it.

## Installation

Not on PyPI yet — install directly from the repo:

```bash
# Pinned to a release
pip install "git+https://github.com/nmorabowen/apeGmsh.git@v1.0.4"

# With all optional dependencies
pip install "apeGmsh[all] @ git+https://github.com/nmorabowen/apeGmsh.git@v1.0.4"
```

Or clone for editable development:

```bash
git clone https://github.com/nmorabowen/apeGmsh.git
cd apeGmsh
pip install -e ".[all]"
```

Requires Gmsh (with Python bindings), NumPy, and Pandas. Optional
extras: `matplotlib` (plotting), `openseespy` (analysis),
`pyvista` + `PyQt6` (Qt viewers).

## Quick start

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="plate") as g:
    # 1. Geometry — sub-composites under g.model
    p1 = g.model.geometry.add_point(0,   0,  0, lc=10)
    p2 = g.model.geometry.add_point(100, 0,  0, lc=10)
    p3 = g.model.geometry.add_point(100, 50, 0, lc=10)
    p4 = g.model.geometry.add_point(0,   50, 0, lc=10)
    l1 = g.model.geometry.add_line(p1, p2)
    l2 = g.model.geometry.add_line(p2, p3)
    l3 = g.model.geometry.add_line(p3, p4)
    l4 = g.model.geometry.add_line(p4, p1)
    loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
    surf = g.model.geometry.add_plane_surface(loop)

    # 2. Mesh — sub-composites under g.mesh
    g.mesh.sizing.set_global_size(10)
    g.mesh.generation.generate(dim=2)
    g.mesh.partitioning.renumber_mesh(method="simple", base=1)

    # 3. Snapshot for the solver
    fem = g.mesh.queries.get_fem_data(dim=2)
    print(fem.info)
```

## Architecture

`apeGmsh` is a session object that owns a single `gmsh` kernel and a
set of focused composites. **There is no Assembly class** — the
session *is* the assembly. Parts are registered into
`g.parts`, meshed together, and queried by label.

### Session composites

| Access | Purpose |
|---|---|
| `g.model`        | OCC geometry (see sub-composites below) |
| `g.parts`        | Part instances & assembly-level bookkeeping |
| `g.physical`     | Named physical groups (pre-mesh, entity-driven) |
| `g.mesh`         | Meshing (see sub-composites below) |
| `g.mesh_selection` | Post-mesh node/element selection sets |
| `g.constraints`  | Solver-agnostic constraint definitions & resolver |
| `g.loads`        | Load patterns & definitions (resolved into `fem.loads`) |
| `g.masses`       | Mass definitions (resolved into `fem.masses`) |
| `g.opensees`     | OpenSees bridge (see sub-composites below) |
| `g.inspect`      | Session-level diagnostics |
| `g.plot`         | Matplotlib visualisations (optional) |
| `g.view`         | Gmsh post-processing scalar/vector views |

### `g.model` sub-composites

Five focused namespaces for OCC geometry:

| Access | Methods |
|---|---|
| `g.model.geometry`   | `add_point`, `add_line`, `add_box`, `add_sphere`, `add_cylinder`, ... |
| `g.model.boolean`    | `fuse`, `cut`, `intersect`, `fragment` |
| `g.model.transforms` | `translate`, `rotate`, `scale`, `mirror`, `copy`, `extrude`, `revolve`, `sweep`, `thru_sections` |
| `g.model.io`         | `load_step`, `load_iges`, `load_dxf`, `save_step`, `save_msh`, `heal_shapes` |
| `g.model.queries`    | `bounding_box`, `center_of_mass`, `mass`, `boundary`, `adjacencies`, `entities_in_bounding_box`, `registry`, `remove`, `remove_duplicates`, `make_conformal`, `fragment_all` |

Plus `g.model.selection` (entity selection) and flat `g.model.sync()`,
`g.model.viewer()`.

### `g.mesh` sub-composites

Seven focused namespaces for meshing:

| Access | Methods |
|---|---|
| `g.mesh.generation`   | `generate`, `set_order`, `refine`, `optimize`, `set_algorithm` |
| `g.mesh.sizing`       | `set_global_size`, `set_size`, `set_size_sources`, `set_size_callback`, `set_size_by_physical` |
| `g.mesh.field`        | `distance`, `threshold`, `box`, `math_eval`, `boundary_layer`, `minimum`, `set_background` |
| `g.mesh.structured`   | `set_transfinite_{curve,surface,volume,automatic}`, `set_recombine`, `recombine`, `set_smoothing`, `set_compound` |
| `g.mesh.editing`      | `embed`, `set_periodic`, `clear`, `reverse`, `relocate_nodes`, `remove_duplicate_{nodes,elements}`, `affine_transform`, `import_stl`, `classify_surfaces`, `create_geometry` |
| `g.mesh.queries`      | `get_nodes`, `get_elements`, `get_fem_data`, `get_element_properties`, `get_element_qualities`, `quality_report` |
| `g.mesh.partitioning` | `partition`, `unpartition`, `compute_renumbering`, `renumber_{nodes,elements,mesh}` |

Plus flat `g.mesh.viewer()` and `g.mesh.results_viewer()` for
interactive windows.

### `g.opensees` sub-composites

Five focused namespaces for the OpenSees bridge:

| Access | Methods |
|---|---|
| `g.opensees.materials` | `add_nd_material`, `add_uni_material`, `add_section` |
| `g.opensees.elements`  | `add_geom_transf`, `assign`, `fix` |
| `g.opensees.ingest`    | `loads(fem)`, `masses(fem)` — ingest resolved records from a FEMData snapshot |
| `g.opensees.inspect`   | `node_table`, `element_table`, `summary` |
| `g.opensees.export`    | `tcl(path)`, `py(path)` |

Plus two lifecycle entry points that stay flat on `g.opensees`:
`set_model(ndm, ndf)` and `build()`.

### The FEM broker

`get_fem_data(dim)` returns a `FEMData` snapshot — a solver-agnostic
container with node IDs, coordinates, element connectivity, physical
groups, mesh selection sets, and the resolved load/mass/constraint
records. It's the single contract between Gmsh and any downstream
solver.

```python
g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
fem = g.mesh.queries.get_fem_data(dim=3)

# fem.node_ids, fem.element_ids, fem.node_coords, fem.connectivity
# fem.info           → mesh statistics (n_nodes, n_elems, bandwidth)
# fem.physical       → physical group lookup
# fem.mesh_selection → post-mesh selection sets
# fem.loads          → resolved NodalLoadRecord / ElementLoadRecord
# fem.masses         → resolved MassRecord
# fem.constraints    → resolved ConstraintRecord
```

### Constraints, loads, masses

Two-stage pipeline:

1. **Define** (pre-mesh): `g.constraints.equal_dof(...)`,
   `g.loads.point("TopFace", force_xyz=...)`,
   `g.masses.volume("Concrete", density=2400)`, etc. Returns
   lightweight definition dataclasses that reference *labels*, not raw
   tags.
2. **Resolve** (post-mesh, automatic): `get_fem_data()` resolves every
   definition against the mesh and attaches the results to
   `fem.constraints`, `fem.loads`, `fem.masses`.

The resolver is pure NumPy math with no Gmsh dependency — solver
bridges consume the records directly.

## Parts

A `Part` owns an isolated Gmsh session, builds a shape, and exports it
to STEP. The session-level `g.parts` registry imports those STEPs back
into the assembly session, tracks which tags belong to which label,
and offers higher-level operations (`fragment_all`, `fuse_group`,
`build_node_map`, `build_face_map`) that keep working through
fragmentation and re-tagging.

```python
from apeGmsh import apeGmsh, Part

web = Part("web")
web.begin()
# ... web.model.geometry.add_... to build the shape
web.save("web.step")
web.end()

with apeGmsh(model_name="I_beam") as g:
    g.parts.import_step("web.step", label="web")
    g.parts.import_step("flange.step", label="top_flange",
                        translate=(0, 0, 200))
    g.parts.import_step("flange.step", label="bot_flange")

    g.parts.fragment_all()   # conformal interfaces

    g.mesh.generation.generate(dim=3)
    g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)
```

## Project layout

```
apeGmsh/
  pyproject.toml
  README.md
  CHANGELOG.md
  docs/                        # mkdocs site source (index, api/, changelog)
  internal_docs/               # authored guides, migration, plans
    MIGRATION_v1.md
    guide_basics.md
    guide_meshing.md
    guide_cad_import.md
    guide_fem_broker.md
    guide_parts_assembly.md
    guide_parts_vs_session.md
    guide_selection.md
  architecture/                # design notes (surfaced in docs site)
  examples/         # runnable notebooks and scripts
  tests/            # pytest suite (no Gmsh required)
  src/
    apeGmsh/
      __init__.py
      _core.py                 # apeGmsh session class
      _session.py              # _SessionBase + composite wiring
      core/
        Model.py               # g.model composite container
        _model_geometry.py     # g.model.geometry
        _model_boolean.py      # g.model.boolean
        _model_transforms.py   # g.model.transforms
        _model_io.py           # g.model.io
        _model_queries.py      # g.model.queries
        Part.py
        _parts_registry.py
        ConstraintsComposite.py
        LoadsComposite.py
        MassesComposite.py
      mesh/
        Mesh.py                # g.mesh composite container
        _mesh_generation.py    # g.mesh.generation
        _mesh_sizing.py        # g.mesh.sizing
        _mesh_field.py         # g.mesh.field (FieldHelper)
        _mesh_structured.py    # g.mesh.structured
        _mesh_editing.py       # g.mesh.editing
        _mesh_queries.py       # g.mesh.queries
        _mesh_partitioning.py  # g.mesh.partitioning
        _mesh_algorithms.py    # Algorithm2D/3D, OptimizeMethod
        FEMData.py
        PhysicalGroups.py
        MeshSelectionSet.py
        MshLoader.py
        Partition.py
      solvers/
        OpenSees.py                # g.opensees composite container
        _opensees_materials.py     # g.opensees.materials
        _opensees_elements.py      # g.opensees.elements
        _opensees_ingest.py        # g.opensees.ingest
        _opensees_inspect.py       # g.opensees.inspect
        _opensees_export.py        # g.opensees.export
        _opensees_build.py         # OpenSees.build() implementation
        _element_specs.py          # element type registry
        Constraints.py
        Loads.py
        Masses.py
        Numberer.py
      viewers/                  # PyQt/PyVista viewers
      results/                  # VTU export + Results container
      viz/                      # Selection, Inspect, Plot, VTKExport
```

## Viewers

apeGmsh ships two separate viewers because pre-solve inspection and
post-solve visualization have different requirements. Both use
PyVista + Qt, but they target different stages of the workflow.

| Viewer | Import | Use when | Data source |
|---|---|---|---|
| **Embedded viewers** | `g.model.viewer()`, `g.mesh.viewer()`, `g.mesh.results_viewer()` | Authoring a model — live inspection while you build geometry, mesh, BCs, loads. | In-process session state (`g`) + its `FEMData` broker. Overlays for labels, physical groups, constraints, loads, BCs. |
| **Standalone viewer** | `from apeGmshViewer import show; show(...)` — or `python -m apeGmshViewer file.vtu` | Reviewing solver output — VTU/PVD files from an OpenSees run, or a `MeshData` built from `Results.to_mesh_data()`. | On-disk files, or a `MeshData` object (in-process only). |

The standalone viewer auto-detects Jupyter and runs in a subprocess
so the notebook keeps its kernel; in scripts it runs blocking.

Source layout:

- `src/apeGmsh/viewers/` — embedded viewers (`ModelViewer`,
  `MeshViewer`, results viewer). Coupled to the `apeGmsh` session.
- `apeGmshViewer/` — standalone post-processing app (`MainWindow` +
  panels + VTU/PVD/MSH loaders). No session dependency.

## Examples

See the `examples/` directory — every notebook runs in-place after
`pip install -e .`. Recommended order for learning the API:

1. `examples/example_plate_basic.ipynb` — minimal plate workflow
2. `examples/example_ibeam_modal.ipynb` — I-beam modal analysis
3. `examples/01_embedded_rebars.ipynb` — embedded 1D rebars in a
   concrete volume
4. `examples/example_frame3D_slab.ipynb` — mixed 1D frame + 2D slab
5. `examples/example_gusset.ipynb` — imported CAD gusset plate

## Migrating from v0.x

See [`internal_docs/MIGRATION_v1.md`](internal_docs/MIGRATION_v1.md) for the full
checklist. In short: package rename (`pyGmsh → apeGmsh`), `g.model.*`
split into five sub-composites, `g.mesh.*` split into seven,
`g.opensees.*` split into five (with `assign_element → assign`,
`consume_*_from_fem → ingest.loads/masses`, `export_tcl/py → export.tcl/py`),
`g.mass → g.masses`, `g.initialize/finalize → g.begin/end`. The
migration guide ships a ~150-line Python script that handles every
mechanical rewrite on an existing project.

## Credits

**Developed by:** Nicolás Mora Bowen · Patricio Palacios · José Abell · Guppi

Part of José Abell's *El Ladruño Research Group*.
