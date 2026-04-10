# Thick-Walled Cylinder with pyGmsh — A Cleaner Gmsh-to-OpenSees Workflow

## pyGmsh as Mesh Provider + OpenSeesPy as External Solver

---

## 1. Overview and Architecture

This example solves the same **Lame problem** (thick-walled cylinder under internal pressure) as the basic example, but uses **pyGmsh** -- a wrapper library that simplifies the Gmsh API and provides solver-ready mesh data.

### Separation of Concerns

The architecture enforces a clean boundary between meshing and solving:

| Layer       | Responsibility                                      | Imports          |
|-------------|-----------------------------------------------------|------------------|
| **pyGmsh**  | Geometry, meshing, physical groups, post-processing  | `gmsh` only      |
| **OpenSeesPy** | Model building, analysis, result extraction       | `openseespy` only |
| **User script** | Connects the two via plain numpy arrays          | Both              |

This pattern makes pyGmsh reusable with *any* solver (OpenSees, FEniCS, Abaqus input decks, etc.) -- the mesh provider never imports or calls any solver.

### Problem Parameters

Same as the basic example:

```python
inner_radius = 100.0    # [mm]
outer_radius = 200.0    # [mm]
lc           = 10.0     # [mm] characteristic mesh size

E   = 210.0e3   # [MPa]  Young's modulus (steel)
nu  = 0.3       # [-]    Poisson's ratio
p   = 100.0     # [MPa]  internal pressure
thk = 1.0       # [mm]   unit thickness (plane strain)
```

---

## 2. Geometry and Mesh with pyGmsh

### 2.1 Initialization

```python
from apeGmsh import apeGmsh

g = apeGmsh(model_name="Plate2D", verbose=True)
g.begin()
```

The `pyGmsh` object wraps Gmsh's state. Unlike a context manager, calling `initialize()` / `finalize()` manually gives more flexibility in notebooks.

### 2.2 Geometry Definition

pyGmsh mirrors Gmsh's bottom-up approach but with a cleaner API and **labels** for every entity:

```python
pc = g.model.geometry.add_point(0, 0, 0, lc=lc, label="center")
p1 = g.model.geometry.add_point(inner_radius, 0, 0, lc=lc, label="inner_x")
p2 = g.model.geometry.add_point(outer_radius, 0, 0, lc=lc, label="outer_x")
p3 = g.model.geometry.add_point(0, outer_radius, 0, lc=lc, label="outer_y")
p4 = g.model.geometry.add_point(0, inner_radius, 0, lc=lc, label="inner_y")

l1 = g.model.geometry.add_line(p1, p2, label="bottom")
l2 = g.model.geometry.add_arc(p2, pc, p3, label="outer_arc")
l3 = g.model.geometry.add_line(p3, p4, label="left")
l4 = g.model.geometry.add_arc(p4, pc, p1, label="inner_arc")

loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
surf = g.model.geometry.add_plane_surface(loop, label="plate")
```

**Comparison with raw Gmsh:**

| Raw Gmsh                              | pyGmsh                                    |
|---------------------------------------|-------------------------------------------|
| `gmsh.model.geo.addPoint(x, y, z, lc)` | `g.model.geometry.add_point(x, y, z, lc=lc, label=...)` |
| `gmsh.model.geo.addLine(p1, p2)`     | `g.model.geometry.add_line(p1, p2, label=...)`     |
| `gmsh.model.geo.addCircleArc(s, c, e)` | `g.model.geometry.add_arc(s, c, e, label=...)`   |
| `gmsh.model.geo.synchronize()`       | Handled internally                         |

The `label` parameter is optional but useful for debugging -- `g.model.queries.registry()` prints a summary of all labeled entities.

### 2.3 Physical Groups

```python
pg_symY = g.physical.add(1, [l1], name="Sym_Y")      # bottom edge: uy = 0
pg_symX = g.physical.add(1, [l3], name="Sym_X")      # left edge:   ux = 0
pg_pres = g.physical.add(1, [l4], name="Pressure")    # inner arc:   pressure
pg_plat = g.physical.add(2, [surf], name="Plate")     # domain
```

- First argument: **dimension** (1 = curves, 2 = surfaces)
- Returns a tag used later with `g.physical.get_nodes()` to retrieve boundary node tags
- `g.physical.summary()` prints all defined physical groups

### 2.4 Mesh Generation

```python
g.mesh.set_order(1)    # linear elements (3-node triangles)
g.mesh.generate(2)     # mesh the 2D surfaces
```

The mesh is stored inside Gmsh's internal state. We extract it in the next step.

---

## 3. Mesh Extraction: Two Paths

pyGmsh offers two complementary ways to extract mesh data:

### 3.1 Raw FEM Data -- `get_fem_data()`

```python
fem = g.mesh.get_fem_data(dim=2)
```

Returns a dictionary with Gmsh's native (non-contiguous) tags:

| Key             | Shape/Type     | Description                          |
|-----------------|----------------|--------------------------------------|
| `node_tags`     | `(N,)`         | Gmsh node IDs (may have gaps)        |
| `node_coords`   | `(N, 3)`       | XYZ coordinates                      |
| `tag_to_idx`    | `dict`         | Gmsh tag -> array index              |
| `connectivity`  | `(nElem, 3)`   | Element-node matrix (Gmsh tags)      |
| `elem_tags`     | `list[int]`    | Gmsh element IDs                     |
| `used_tags`     | `set[int]`     | Nodes actually used by elements      |

This data is needed for boundary queries, edge extraction, and plotting.

### 3.2 Solver-Ready Mesh -- `get_numbered_mesh()`

```python
from apeGmsh import Numberer, NumberedMesh

mesh = g.mesh.get_numbered_mesh(dim=2, method="simple")
```

Returns a `NumberedMesh` object with **contiguous 1-based IDs** ready for any solver:

| Attribute               | Description                                    |
|-------------------------|------------------------------------------------|
| `node_ids`              | Contiguous 1-based solver node IDs             |
| `node_coords`           | Coordinates aligned with `node_ids`            |
| `elem_ids`              | Contiguous 1-based solver element IDs          |
| `connectivity`          | Element-node matrix **in solver IDs**          |
| `gmsh_to_solver_node`   | Gmsh tag -> solver ID map                      |
| `solver_to_gmsh_node`   | Solver ID -> Gmsh tag map                      |
| `gmsh_to_solver_elem`   | Gmsh element tag -> solver element ID          |
| `bandwidth`             | Semi-bandwidth of the connectivity             |
| `n_nodes`, `n_elems`    | Counts                                         |

The Numberer **automatically filters orphan nodes** (`used_only=True` by default), so the center point at `(0,0)` is excluded -- no manual `used_tags` check needed.

### 3.3 Boundary Queries

```python
bottom_nodes_gmsh = g.physical.get_nodes(1, pg_symY)['tags']   # Gmsh tags
left_nodes_gmsh   = g.physical.get_nodes(1, pg_symX)['tags']
inner_nodes_gmsh  = g.physical.get_nodes(1, pg_pres)['tags']
```

These return **Gmsh tags**. To use them in the solver, translate via `mesh.gmsh_to_solver_node`.

### 3.4 Edge Elements for Pressure Loading

```python
inner_elems = g.mesh.get_elements(dim=1, tag=l4)
inner_edges = []
for etype, enodes in zip(inner_elems['types'], inner_elems['node_tags']):
    props = g.mesh.get_element_properties(etype)
    inner_edges = enodes.reshape(-1, props['n_nodes']).astype(int)
```

Passing `tag=l4` retrieves only 1D line elements on the inner arc curve.

---

## 4. Building the OpenSees Model

### 4.1 Nodes -- Direct from NumberedMesh

```python
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)

for i in range(mesh.n_nodes):
    nid = int(mesh.node_ids[i])
    x, y = float(mesh.node_coords[i, 0]), float(mesh.node_coords[i, 1])
    ops.node(nid, x, y)
```

No manual remapping loop needed. Compare with the basic example:

```python
# OLD (manual):
gmsh_to_ops = {}
new_id = 0
for gtag, coords in zip(node_tags, node_coords):
    if int(gtag) not in used_tags:
        continue
    new_id += 1
    gmsh_to_ops[int(gtag)] = new_id
    ops.node(new_id, float(coords[0]), float(coords[1]))

# NEW (Numberer):
for i in range(mesh.n_nodes):
    ops.node(int(mesh.node_ids[i]), *mesh.node_coords[i])
```

### 4.2 Elements -- Connectivity Already in Solver IDs

```python
ops.nDMaterial("ElasticIsotropic", 1, E, nu)

for i in range(mesh.n_elems):
    eid = int(mesh.element_ids[i])
    n1, n2, n3 = [int(n) for n in mesh.connectivity[i]]
    ops.element("tri31", eid, n1, n2, n3, thk, "PlaneStrain", 1)
```

Since `mesh.connectivity` is already expressed in solver IDs, no translation is needed.

### 4.3 Boundary Conditions -- Tag Translation

```python
for gtag in bottom_nodes_gmsh.astype(int):
    sid = mesh.gmsh_to_solver_node.get(int(gtag))
    if sid is not None:
        ops.fix(sid, 0, 1)   # free ux, fix uy

for gtag in left_nodes_gmsh.astype(int):
    sid = mesh.gmsh_to_solver_node.get(int(gtag))
    if sid is not None:
        ops.fix(sid, 1, 0)   # fix ux, free uy
```

The `.get()` with `None` check handles the rare case where a physical-group node is an orphan (filtered out by the Numberer).

### 4.4 Pressure Loading

The consistent nodal force calculation is identical to the basic example. The only difference is the tag translation:

```python
# Gmsh tags -> solver IDs for force application
o1 = mesh.gmsh_to_solver_node[n1g]
o2 = mesh.gmsh_to_solver_node[n2g]
```

The load formula remains the same -- for a 2-node edge of length L:

```
F_i = (L/6)(2*t_i + t_j)     F_j = (L/6)(t_i + 2*t_j)
```

where **t** = p * **n_hat** is the radially outward traction.

---

## 5. Analysis and Result Extraction

### 5.1 Solve

```python
ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1.0e-8, 10)
ops.algorithm("Newton")
ops.integrator("LoadControl", 1.0)
ops.analysis("Static")
ok = ops.analyze(1)
```

### 5.2 Extract Results into NumPy

After solving, we pull everything out of OpenSees into plain numpy arrays and then call `ops.wipe()` to release the solver. This keeps the solver window as small as possible.

**Displacements** (using reverse map: solver ID -> Gmsh tag -> array index):

```python
disp = np.zeros((nNode, 2))
for solver_id, gmsh_tag in mesh.solver_to_gmsh_node.items():
    idx = tag_to_idx[gmsh_tag]
    disp[idx, 0] = ops.nodeDisp(solver_id, 1)
    disp[idx, 1] = ops.nodeDisp(solver_id, 2)
```

**Element stresses** (using reverse map: solver elem ID -> Gmsh elem tag):

```python
sig_xx = np.zeros(nElem)
for solver_eid in range(1, mesh.n_elems + 1):
    gmsh_etag = mesh.solver_to_gmsh_elem[solver_eid]
    idx = elem_tags.index(gmsh_etag)
    s = ops.eleResponse(solver_eid, "stresses")
    sig_xx[idx] = s[0]
```

**Radial displacement:**

```python
ur = (node_coords[:, 0]*disp[:, 0] + node_coords[:, 1]*disp[:, 1]) / r_safe
```

**Nodal averaging** for smooth stress contours (same technique as the basic example).

After extraction: `ops.wipe()` -- OpenSees is done.

---

## 6. Post-Processing Views (Solver-Agnostic)

### 6.1 Matplotlib Quick Checks

```python
import matplotlib.tri as mtri

triang = mtri.Triangulation(node_coords[:, 0], node_coords[:, 1], conn_idx)

ax.tripcolor(triang, ur, shading='gouraud', cmap='coolwarm')      # smooth
ax.tripcolor(triang, sig_xx, shading='flat', cmap='RdYlBu_r')     # element-constant
```

The notebook produces three side-by-side plots: radial displacement, nodal-averaged stress, and element-constant stress (raw CST output).

### 6.2 Gmsh Views via pyGmsh

Pass numpy arrays back to Gmsh for interactive visualization in the GUI:

```python
# Element-constant fields (raw CST stress)
g.view.add_element_scalar("sig_xx (elem)", elem_tags, sig_xx)
g.view.add_element_scalar("sig_yy (elem)", elem_tags, sig_yy)
g.view.add_element_scalar("sig_xy (elem)", elem_tags, sig_xy)

# Nodal fields (smooth interpolation)
g.view.add_node_scalar("sig_xx (nodal avg)", node_tag_list, sig_xx_nodal)
g.view.add_node_scalar("u_r", node_tag_list, ur)

# Vector field
g.view.add_node_vector("displacement", node_tag_list, disp)
```

**pyGmsh view methods:**

| Method                    | Data lives at... | Visual result       |
|---------------------------|------------------|---------------------|
| `add_element_scalar()`    | Elements         | Piecewise-constant  |
| `add_element_vector()`    | Elements         | Piecewise-constant  |
| `add_node_scalar()`       | Nodes            | Gouraud-smooth      |
| `add_node_vector()`       | Nodes            | Gouraud-smooth      |

These methods are **solver-agnostic** -- they accept plain lists/arrays and Gmsh tags, never importing OpenSees.

### 6.3 Launch the GUI

```python
g.launch_gui()    # interactive exploration (close window to continue)
g.end()      # release Gmsh resources
```

---

## 7. Three Approaches Compared

pyGmsh supports three paths from mesh to solver, each suited to different needs:

### Path 1: Numberer (recommended, used in this notebook)

```python
mesh = g.mesh.get_numbered_mesh(dim=2, method="simple")
for i in range(mesh.n_nodes):
    ops.node(int(mesh.node_ids[i]), *mesh.node_coords[i])
# Use mesh.gmsh_to_solver_node / solver_to_gmsh_node for BC/result mapping
```

Best for: full control, custom loads, multi-material models, bandwidth optimization.

### Path 2: g2o wrapper (quick prototype)

```python
if g.g2o.is_available():
    g.g2o.transfer(verbose=True)
    # Nodes and elements created automatically using Gmsh tags as IDs
```

Best for: rapid prototyping with standard elements. Less control over numbering and orphan filtering. Uses Gmsh tags directly as OpenSees IDs.

### Path 3: Manual loop (basic example)

```python
gmsh_to_ops = {}
new_id = 0
for gtag, coords in zip(node_tags, node_coords):
    if int(gtag) not in used_tags:
        continue
    new_id += 1
    gmsh_to_ops[int(gtag)] = new_id
    ops.node(new_id, float(coords[0]), float(coords[1]))
```

Best for: understanding the flow, teaching.

### Decision Guide

| Scenario                               | Recommended path                          |
|----------------------------------------|-------------------------------------------|
| Standard workflow (most cases)         | `get_numbered_mesh()` + maps              |
| Quick prototype, standard elements     | `g.g2o.transfer()`                        |
| Custom edge loads (like pressure here) | Numberer + raw `get_fem_data`             |
| Bandwidth optimization needed          | `get_numbered_mesh(method="rcm")`         |
| Multi-material with different elements | Numberer + physical group maps            |
| Teaching / understanding the flow      | Manual loop (basic example)               |

---

## 8. Summary: pyGmsh vs. Raw Gmsh Workflow

| Step                  | Raw Gmsh (basic example)                          | pyGmsh (this example)                         |
|-----------------------|---------------------------------------------------|------------------------------------------------|
| Initialize            | `gmsh.initialize()`                               | `g = apeGmsh(...); g.begin()`              |
| Add point             | `gmsh.model.geo.addPoint(x,y,z,lc)`              | `g.model.geometry.add_point(x,y,z, lc=lc, label=...)`  |
| Add arc               | `gmsh.model.geo.addCircleArc(s,c,e)`             | `g.model.geometry.add_arc(s,c,e, label=...)`            |
| Synchronize           | `gmsh.model.geo.synchronize()` (manual, easy to forget) | Handled internally                       |
| Physical group        | `gmsh.model.addPhysicalGroup(dim, [tags])`        | `g.physical.add(dim, [tags], name=...)`        |
| Get boundary nodes    | `gmsh.model.mesh.getNodesForPhysicalGroup(...)`   | `g.physical.get_nodes(dim, tag)`               |
| Extract mesh          | Manual: `getNodes()` + `getElements()` + filter   | `g.mesh.get_fem_data()` (all-in-one dict)      |
| Renumber for solver   | Manual loop with `used_tags` check                | `g.mesh.get_numbered_mesh()` (automatic)       |
| Orphan filtering      | Manual `set(connectivity.flatten())`              | Built into Numberer (`used_only=True`)         |
| Post-processing views | `gmsh.view.add()` + `addModelData()` (verbose)   | `g.view.add_element_scalar(name, tags, data)`  |
| Launch GUI            | `gmsh.fltk.run()`                                | `g.launch_gui()`                               |
| Finalize              | `gmsh.finalize()`                                 | `g.end()`                                 |

The key improvements are:
- **Labels** on geometry entities for debugging
- **Automatic synchronization** (no forgetting `synchronize()`)
- **One-call mesh extraction** with orphan filtering
- **Numberer** handles contiguous renumbering and bidirectional maps
- **Simplified view API** for passing results back to Gmsh
