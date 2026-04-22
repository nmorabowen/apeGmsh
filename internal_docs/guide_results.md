# Results — Post-processing Container

## What Results is

After the solver finishes, you have raw numpy arrays of displacements, stresses, and reactions sitting in memory next to a `FEMData` object that knows the mesh. The `Results` class is the bridge between those two things and the outside world: ParaView, PyVista, colleagues who want your VTU files.

`Results` is a **self-contained post-processing container**. It bundles mesh geometry (nodes, connectivity, cell types) with named field arrays (nodal or element-level) in a single, immutable object that requires no live Gmsh session, no live solver session, and no running event loop. Once you hold a `Results` object you can export to VTK XML, open an interactive viewer, or pass it to another script — the mesh and the fields travel together.

The class lives at `apeGmsh.results.Results` and is re-exported from the package root:

```python
from apeGmsh import Results
```

Internally it stores VTK-ready flat cell arrays and 0-based connectivity, so the numpy-to-VTK translation that usually eats half a post-processing script is already done for you at construction time.


## Construction

There are two entry points: one for results you just computed (arrays in memory), and one for results saved to disk.

### From FEMData + numpy arrays — `Results.from_fem`

This is the primary workflow. You run an analysis, extract the solution vectors as numpy arrays, and hand them to `Results` together with the `FEMData` that describes the mesh.

**Static analysis (single load case):**

```python
fem = g.mesh.queries.get_fem_data(dim=3)

# ... run OpenSees, extract arrays ...

results = Results.from_fem(
    fem,
    point_data={"Displacement": u_array},    # (N, 3) or (N,)
    cell_data={"Stress_xx": stress_array},   # (E,) or (E, 3)
    name="gravity_load",
)
```

The `point_data` dict maps field names to numpy arrays sized `(N,)` for scalars or `(N, 3)` for vectors, where `N` matches `len(fem.nodes.ids)`. The `cell_data` dict works the same way with `E` matching the primary element count.

**Time-series (transient, modal, pushover):**

For multi-step results, pass `steps` instead of `point_data` / `cell_data`. Each step is a dict with a `"time"` key and optional `"point_data"` / `"cell_data"` dicts:

```python
steps = []
for i, (freq, phi) in enumerate(zip(frequencies, mode_shapes)):
    steps.append({
        "time": freq,
        "point_data": {"ModeShape": phi},   # (N, 3)
    })

results = Results.from_fem(fem, steps=steps, name="modal_analysis")
```

You cannot mix the two styles — passing both `point_data` and `steps` raises a `ValueError`. This is intentional: a single static result set and a time-series are fundamentally different objects in ParaView, so the API forces you to choose up front.

The `"time"` value in each step dict can represent whatever the physics demands: time in seconds for transient analysis, frequency in Hz for modal analysis, or pseudo-time / drift ratio for pushover. ParaView will use it as the animation axis.

**Full signature:**

```python
Results.from_fem(
    fem: FEMData,
    *,
    point_data: dict[str, ndarray] | None = None,
    cell_data: dict[str, ndarray] | None = None,
    steps: list[dict] | None = None,
    name: str = "results",
) -> Results
```

**Automatic cell-data padding.** When your mesh includes elements from physical groups of different dimensions (e.g., 1-D column lines alongside 2-D slab triangles), the VTK grid will contain more cells than the primary connectivity. If your `cell_data` arrays match only the primary element count, `from_fem` automatically pads them with `NaN` for the extra cells. This means you never have to worry about sizing mismatches when your model has mixed-dimension physical groups.


### From a VTK file — `Results.from_file`

Load previously exported results back into memory:

```python
results = Results.from_file("output.vtu")
results = Results.from_file("modes.pvd")    # time-series collection
```

This delegates to `apeGmshViewer.loaders.vtu_loader.load_file`, so any format that loader supports (VTU, VTK, PVD) works here. The display name defaults to the filename stem but can be overridden:

```python
results = Results.from_file("run_042.vtu", name="final_design")
```

**Full signature:**

```python
Results.from_file(
    filepath: str | Path,
    *,
    name: str | None = None,
) -> Results
```

This is useful when you want to reload results from a previous run, compare two meshes side-by-side, or hand someone a VTU and have them pull it back into apeGmsh for further queries.


## Field access

Once a `Results` object exists, you retrieve fields by name.

### Nodal fields — `get_point_field`

```python
u = results.get_point_field("Displacement")         # ndarray (N, 3)
```

For time-series results, pass the step index:

```python
phi_3 = results.get_point_field("ModeShape", step=3)   # mode 4 (0-indexed)
```

Omitting `step` on a time-series field raises a `ValueError` that tells you the valid range. An out-of-range step raises `IndexError`. If the field name does not exist, a `KeyError` is raised listing all available point field names — useful for debugging typos.

### Element fields — `get_cell_field`

```python
stress = results.get_cell_field("Stress_xx")
stress_at_step = results.get_cell_field("Stress_xx", step=5)
```

Same error semantics as `get_point_field`.

### Discovering what fields exist

The `field_names` property returns a dict with two keys:

```python
results.field_names
# {"point": ["Displacement", "ModeShape"], "cell": ["Stress_xx"]}
```

This merges static and time-series fields into a single list per category.


## Properties

The `Results` object exposes several read-only properties for introspection:

| Property | Type | Description |
|---|---|---|
| `node_coords` | `ndarray (N, 3)` | Mesh node coordinates |
| `cells` | `ndarray` | Flat VTK cell array |
| `cell_types` | `ndarray (uint8)` | Per-cell VTK type codes |
| `point_fields` | `dict[str, ndarray]` | Static nodal fields |
| `cell_fields` | `dict[str, ndarray]` | Static element fields |
| `time_steps` | `list[float] \| None` | Time/frequency values |
| `has_time_series` | `bool` | True if two or more steps exist |
| `n_steps` | `int` | Number of steps (1 for static) |
| `n_primary_cells` | `int` | Primary-dim element count |
| `n_total_cells` | `int` | Total cells including extra PG elements |
| `physical_groups` | `PhysicalGroupSet \| None` | PG data carried from FEMData |
| `name` | `str` | Display name |
| `field_names` | `dict` | `{"point": [...], "cell": [...]}` |


## Export

### Single timestep — `to_vtu`

```python
path = results.to_vtu("output/gravity.vtu")
```

Writes a VTK XML Unstructured Grid file. Returns the `Path` of the written file. For time-series results, `to_vtu` writes **step 0 only** — use `to_pvd` for the full series.

The VTU file is a standard VTK XML format that ParaView, VisIt, and any VTK-based tool can open directly. All point and cell fields are included as named data arrays.

### Time-series — `to_pvd`

```python
paths = results.to_pvd("output/modes")
```

Pass a base path **without extension**. The method produces:

```
output/modes.pvd          # PVD collection file
output/modes_000.vtu      # step 0
output/modes_001.vtu      # step 1
output/modes_002.vtu      # step 2
...
```

Returns a list of all written `Path` objects, with the PVD file first. The PVD file is a lightweight XML that references the individual VTU files and maps each to its time value. Open the `.pvd` in ParaView and you get animation controls out of the box.

The parent directory is created automatically if it does not exist.


## Visualization

### Interactive viewer — `viewer`

```python
results.viewer()                    # non-blocking (subprocess)
results.viewer(blocking=True)       # in-process, blocks until closed
```

The non-blocking path writes temporary VTU/PVD files and launches `apeGmshViewer` as a subprocess, so your script or notebook continues executing. The temporary files are cleaned up when the Python process exits.

The blocking path transfers mesh data directly in memory via `to_mesh_data()` and opens the viewer in-process — no file I/O. This is faster for quick inspection but locks the interpreter until you close the viewer window.

No live Gmsh session is needed for either path.

### Text summary — `summary`

```python
print(results.summary())
```

Produces a multi-line human-readable summary:

```
Results: 'gravity_load'
  Mesh: 4521 nodes, 22840 elements (22400 primary + 440 extra)
  Point fields: Displacement
  Cell fields:  Stress_xx
  Physical groups: 3
```

The `repr` is more compact and suitable for REPL inspection:

```python
>>> results
<Results 'gravity_load', 4521 nodes, 22840 cells, 2 fields>
```


## Practical workflow: OpenSees to ParaView

The most common workflow is: build a model with apeGmsh, run an OpenSees analysis, wrap the solution in `Results`, and export to VTU for visualization.

### Static analysis

```python
from apeGmsh import apeGmsh, Results
import openseespy.opensees as ops
import numpy as np

# 1. Build geometry and mesh
g = apeGmsh("slab")
g.begin()
# ... geometry, physical groups, constraints, loads ...
g.mesh.generation.generate(dim=3)
fem = g.mesh.queries.get_fem_data(dim=3)

# 2. Build and run OpenSees model
# ... (build nodes, elements, loads from fem) ...
ops.analyze(1)

# 3. Extract displacement array
n_nodes = len(fem.nodes.ids)
u_array = np.zeros((n_nodes, 3))
for i, nid in enumerate(fem.nodes.ids):
    u_array[i, :] = ops.nodeDisp(int(nid))[:3]

# 4. Wrap in Results and export
results = Results.from_fem(
    fem,
    point_data={"Displacement": u_array},
    name="slab_gravity",
)
results.to_vtu("output/slab_gravity.vtu")
results.viewer()

g.end()
```

### Convergence study

When you run the same problem on multiple mesh refinements, create a separate `Results` for each and export them to the same output directory:

```python
for mesh_size in [0.5, 0.25, 0.125]:
    g = apeGmsh(f"plate_h{mesh_size}")
    g.begin()
    # ... geometry ...
    g.mesh.sizing.set_global(mesh_size)
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)

    # ... solve ...
    u_array = extract_displacements(fem, ops)

    results = Results.from_fem(
        fem,
        point_data={"Displacement": u_array},
        name=f"h_{mesh_size}",
    )
    results.to_vtu(f"output/convergence/h_{mesh_size}.vtu")
    g.end()
```

Open all three VTU files in ParaView, apply the same color map, and you can visually confirm convergence. Use `get_point_field` to pull arrays back for quantitative comparison (max displacement, L2 norm of error, etc.).

### Modal analysis

```python
# After eigenvalue analysis
eigenvalues = ops.eigen(10)
frequencies = [np.sqrt(ev) / (2 * np.pi) for ev in eigenvalues]

steps = []
for mode_idx in range(10):
    phi = np.zeros((n_nodes, 3))
    for i, nid in enumerate(fem.nodes.ids):
        phi[i, :] = ops.nodeEigenvector(int(nid), mode_idx + 1)[:3]
    steps.append({
        "time": frequencies[mode_idx],
        "point_data": {"ModeShape": phi},
    })

results = Results.from_fem(fem, steps=steps, name="modal_10modes")
results.to_pvd("output/modes")
```

In ParaView, open `modes.pvd`. Each "timestep" is a natural frequency, and the "ModeShape" field shows the mode shape at that frequency. Use the Warp by Vector filter to animate mode shapes.


## Field naming conventions

Field names are free-form strings — `Results` does not enforce a schema. That said, consistent naming pays off when you open files in ParaView months later or share them with colleagues. The following conventions are recommended:

| Quantity | Suggested name | Shape | Notes |
|---|---|---|---|
| Displacement | `"Displacement"` | `(N, 3)` | ParaView auto-detects as vector |
| Velocity | `"Velocity"` | `(N, 3)` | |
| Acceleration | `"Acceleration"` | `(N, 3)` | |
| Reaction force | `"ReactionForce"` | `(N, 3)` | |
| Mode shape | `"ModeShape"` | `(N, 3)` | Use with Warp by Vector |
| Stress components | `"Stress_xx"`, `"Stress_yy"`, `"Stress_xy"` | `(E,)` | One scalar per component |
| Von Mises stress | `"VonMises"` | `(E,)` | |
| Strain components | `"Strain_xx"`, `"Strain_yy"`, `"Strain_xy"` | `(E,)` | |
| Pressure | `"Pressure"` | `(E,)` | Cell field for element pressure |
| Temperature | `"Temperature"` | `(N,)` | Nodal scalar |

ParaView treats arrays with 3 columns as vectors automatically, which enables the Warp by Vector, Glyph, and Stream Tracer filters. Single-column arrays are treated as scalars and can be directly color-mapped.

If you store all six stress components, consider also storing them as a single `(E, 6)` array named `"Stress"` — ParaView can display individual components via the component selector.


## Physical groups in VTK

When you construct a `Results` via `from_fem`, the method does more than just copy the primary connectivity. It walks **all** physical groups in the `FEMData` and includes elements from every dimension. If your model has 2-D slab elements as the primary mesh and 1-D beam elements in a physical group for columns, both end up in the same VTK unstructured grid as a mixed-type cell set.

The dimension and VTK cell type are inferred automatically from the nodes-per-element count:

| Dimension | npe | VTK type |
|---|---|---|
| 0 | 1 | Vertex |
| 1 | 2 | Line |
| 2 | 3 | Triangle |
| 2 | 4 | Quad |
| 3 | 4 | Tetrahedron |
| 3 | 6 | Wedge |
| 3 | 8 | Hexahedron |

The `n_primary_cells` property tells you how many cells came from the primary connectivity, and `n_total_cells` gives the total including extras. This distinction matters when you provide `cell_data` arrays: if your stress array has only `n_primary_cells` entries (because stress is only defined on the primary elements), `from_fem` pads the extra cells with `NaN` automatically.

The `physical_groups` property on the `Results` object carries the `PhysicalGroupSet` reference from `FEMData`, so you can still query group membership after construction — though for VTK export, the group structure is embedded in the mixed-type cell layout rather than as explicit cell arrays.

In ParaView, you can use the Threshold filter on cell type to isolate specific element kinds (e.g., show only triangles, hide lines), which effectively recovers group-level visualization.


## Debugging tips

**Shape mismatches.** The most common error is an array whose first dimension does not match the expected node or element count. `from_fem` validates cell data against `n_primary_cells` and `n_total_cells`, but point data is validated later when PyVista builds the grid. If you get a cryptic VTK error, check shapes first:

```python
print(f"Nodes: {len(fem.nodes.ids)}, array: {u_array.shape}")
print(f"Elements: {len(fem.elements.ids)}, array: {stress.shape}")
```

**Field not found.** `get_point_field` and `get_cell_field` raise `KeyError` with the list of available names. If you suspect a typo, use `results.field_names` to inspect what was stored.

**Time-series step errors.** Forgetting to pass `step=` on a time-series field raises `ValueError` with the valid step range. Passing an out-of-range index raises `IndexError`. Both messages are descriptive.

**Empty mesh.** If `fem.elements.connectivity` is empty, the VTK grid will have zero cells. The `summary()` method will show `0 elements`, which is your signal that the mesh was not generated or the wrong dimension was requested in `get_fem_data`.

**Mixed steps and static data.** You cannot pass both `point_data` and `steps` to `from_fem`. If you need static cell data alongside time-varying point data, put the cell data inside each step dict:

```python
steps.append({
    "time": t,
    "point_data": {"Displacement": u},
    "cell_data": {"Stress_xx": s},
})
```

**Viewer not opening.** The non-blocking `viewer()` call writes to a temp directory and launches a subprocess. If apeGmshViewer is not installed, you will get an `ImportError`. Install it with `pip install apeGmshViewer`. The blocking path requires PyVista as well.

**Reloading edited results.** `from_file` reads everything into memory at construction time. If you re-export and want to reload, create a new `Results` object — the original is immutable.
