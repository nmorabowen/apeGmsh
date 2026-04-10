# pyGmsh

Gmsh wrapper for structural FEM workflows with an Abaqus-style Part/Assembly architecture and OpenSees integration.

## Installation

```bash
cd pyGmsh
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

Requires Gmsh (with Python bindings), NumPy, and Pandas. Optional extras: matplotlib (plotting), openseespy (analysis), gmsh2opensees (one-liner mesh transfer).

## Quick start

### Standalone (single model)

```python
from apeGmsh import apeGmsh

g = apeGmsh(model_name="plate", verbose=True)
g.begin()

p1 = g.model.geometry.add_point(0, 0, 0, lc=10)
p2 = g.model.geometry.add_point(100, 0, 0, lc=10)
p3 = g.model.geometry.add_point(100, 50, 0, lc=10)
p4 = g.model.geometry.add_point(0, 50, 0, lc=10)

l1 = g.model.geometry.add_line(p1, p2)
l2 = g.model.geometry.add_line(p2, p3)
l3 = g.model.geometry.add_line(p3, p4)
l4 = g.model.geometry.add_line(p4, p1)

loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
surf = g.model.geometry.add_plane_surface(loop)

g.mesh.generate(2)
g.mesh.renumber_mesh(method="simple", base=1)
fem = g.mesh.get_fem_data(dim=2)
print(fem.summary())

g.end()
```

### Part / Assembly (multi-part)

```python
from apeGmsh import Part, Assembly

# Build geometry in isolated sessions
web = Part("web")
web.begin()
# ... build geometry with web.model.geometry.add_point(), add_line(), etc.
web.save("web.step")
web.end()

flange = Part("flange")
flange.begin()
# ... build geometry
flange.save("flange.step")
flange.end()

# Assemble, fragment, mesh
asm = Assembly("I_beam")
asm.begin()
asm.add_part(web)
asm.add_part(flange, translate=(0, 0, 200), label="top_flange")
asm.add_part(flange, label="bot_flange")

asm.fragment_all()
asm.mesh.generate(dim=2)
asm.mesh.renumber_mesh(method="rcm", base=1)
fem = asm.mesh.get_fem_data(dim=2)

# fem.node_ids, fem.node_coords, fem.connectivity → feed to any solver
asm.end()
```

## Architecture

Two main classes with a clean separation:

- **Part** -- geometry modeler. Creates points, lines, surfaces, volumes in an isolated Gmsh session. Exports to STEP. Has only `model` and `inspect` composites.

- **Assembly** -- full Gmsh wrapper. Imports Parts, positions them, fragments for conformal interfaces, meshes, assigns physical groups, extracts FEM data, resolves constraints. Has all composites: `mesh`, `model`, `physical`, `view`, `plot`, `inspect`, `partition`, `g2o`.

The STEP file is the only contract between Part and Assembly -- they are fully decoupled.

### Composites

| Composite        | Access           | Purpose                                  |
|:-----------------|:-----------------|:-----------------------------------------|
| `Model`          | `.model`         | Geometry building, sync, export          |
| `Mesh`           | `.mesh`          | Generation, size control, FEM extraction |
| `PhysicalGroups` | `.physical`      | Named entity groups for BCs/loads        |
| `View`           | `.view`          | Gmsh post-processing scalar/vector views |
| `Plot`           | `.plot`          | Matplotlib visualisations                |
| `Inspect`        | `.inspect`       | Geometry and mesh queries                |
| `Partition`      | `.partition`     | Mesh partitioning                        |
| `Gmsh2OpenSees`  | `.g2o`           | One-liner mesh transfer to OpenSees      |

### Renumbering & FEM Data

`renumber_mesh()` remaps non-contiguous Gmsh tags to contiguous 1-based solver IDs directly in the Gmsh model. After renumbering, `get_fem_data()` returns a `FEMData` object with solver-ready IDs, coordinates, and connectivity. Supports simple sequential numbering and RCM bandwidth optimisation.

```python
g.mesh.renumber_mesh(method="rcm", base=1)
fem = g.mesh.get_fem_data(dim=2)
# fem.node_ids, fem.element_ids, fem.node_coords, fem.connectivity
```

### Constraints

Solver-agnostic two-stage constraint pipeline on the Assembly:

1. **Define** (pre-mesh): `asm.equal_dof(...)`, `asm.tie(...)`, `asm.rigid_link(...)`, etc.
2. **Resolve** (post-mesh): `asm.resolve_constraints(...)` produces concrete node pairs, weights, and constraint matrices for any solver.

Four levels: node-to-node, node-to-group, node-to-surface, surface-to-surface.

## Project layout

```
pyGmsh/
  pyproject.toml
  README.md
  src/
    pyGmsh/
      __init__.py
      _core.py           # standalone pyGmsh class
      Part.py
      Assembly.py
      Mesh.py
      Model.py
      View.py
      Plot.py
      Inspect.py
      PhysicalGroups.py
      Partition.py
      Gmsh2OpenSees.py
      VTKExport.py
      Constraints.py
      Numberer.py
      OpenSees.py
  examples/
    example_plate_pyGmsh.ipynb
    example_assembly.ipynb
    ...
```

## Examples

See the `examples/` directory. After `pip install -e .`, all notebooks can be run directly without sys.path manipulation.
