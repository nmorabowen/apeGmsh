# Gmsh Interface

## Framework and Intended Use

Gmsh is a **mesh generator**, not a solver. Its scope is precisely bounded: geometry construction → meshing → post-processing visualization. It does not solve PDEs, assemble stiffness matrices, or apply boundary conditions. Those responsibilities belong to the solver (OpenSees, in our case).

The framework has three pillars:

```
┌─────────────────────────────────────────────────┐
│                   Gmsh Framework                 │
├────────────────┬────────────────┬────────────────┤
│   Geometry     │    Meshing     │ Post-Processing│
│                │                │                │
│ BRep model     │ Node/element   │ View data      │
│ OCC or geo     │ generation     │ (scalar,       │
│ kernel         │ per dimension  │  vector,       │
│                │                │  tensor fields)│
│ CAD import     │ Size control   │                │
│ (STEP/IGES)    │ Optimization   │ Visualization  │
│                │ Partitioning   │ (FLTK GUI)     │
└────────────────┴────────────────┴────────────────┘
```

The design philosophy is **automation-first**. Gmsh was built to be driven programmatically — the GUI exists for inspection and debugging, not as the primary workflow. This is reflected in the architecture: every operation is available through the API, but only a subset is accessible through the GUI.

For apeGmsh, this means Gmsh is the **geometry and mesh broker**. It handles everything up to the point where we have nodes, elements, connectivity, and physical group assignments. From there, apeGmsh translates into the solver's domain.

## Gmsh API

### Generation from a Single Source

All Gmsh API bindings are **auto-generated** from a single master definition file (`api/gen.py`). Running `python gen.py` produces:

| Output | Language | Mechanism |
| --- | --- | --- |
| `gmsh.h` | C++ | Header with direct calls into Gmsh internals |
| `gmshc.h` + `gmshc.cpp` | C | C header + C-to-C++ wrapper |
| `gmsh.h_cwrap` | C++ | C++ API redefined through the C layer |
| `gmsh.py` | Python | ctypes bindings to the C API |
| `gmsh.jl` | Julia | ccall bindings to the C API |
| `gmsh.f90` | Fortran | iso_c_binding to the C API |
| `api.texi` | — | Texinfo documentation |

The chain for non-C++ languages is always:

```
gen.py  ──generates──►  Language binding  ──calls──►  C API (gmshc)  ──wraps──►  C++ internals
```

This means **every language has identical functionality**. There is no "Python-only" or "Julia-only" feature. If a function exists in one binding, it exists in all of them.

### Design Constraints

From the `gen.py` header:

> *"By design, the Gmsh API is purely functional, and only uses elementary types from the target language."*

This has consequences:

**Purely functional** — no objects, no classes, no state held in return values. All state lives inside Gmsh's global singleton. You call functions that mutate that state and query functions that read it. There is no `mesh = model.getMesh()` object — you call `gmsh.model.mesh.getNodes()` and get back flat arrays.

**Elementary types only** — the API uses only: integers, doubles, strings, and vectors/arrays of these. No structs, no enums, no custom types cross the API boundary. A `(dim, tag)` pair is passed as two integers or a vector of integer pairs — never as a named struct.

### Module Hierarchy

The API is organized as a tree of modules:

```
gmsh
├── option          — get/set global options
├── model
│   ├── mesh        — meshing operations
│   │   └── field   — mesh size fields
│   ├── geo         — built-in kernel
│   │   └── mesh    — geo kernel meshing constraints
│   └── occ         — OpenCASCADE kernel
│       └── mesh    — occ kernel meshing constraints
├── view            — post-processing views
│   └── option      — view-specific options
├── plugin          — plugin management
├── graphics        — rendering control
├── fltk            — GUI control
├── parser          — .geo file parser access
├── onelab          — ONELAB parameter server
├── logger          — logging control
└── algorithm       — raw algorithms (currently minimal)
```

In Python this maps directly to dotted module access: `gmsh.model.occ.addBox(...)`, `gmsh.model.mesh.generate(3)`.

### What Crosses the API Boundary

Every API call transmits only elementary types. Here is how Gmsh's internal data maps to what you receive:

| Internal concept | API representation |
| --- | --- |
| Entity | `(dim, tag)` — two ints |
| Entity list | `vector<pair<int,int>>` → flat list of int pairs |
| Node | `(tag, [x,y,z], [u,v,...])` — int + double arrays |
| Element | `(tag, type, [nodeTags])` — int + int array |
| Coordinates | Flat `double[]` array, stride 3 |
| Connectivity | Flat `size_t[]` array per element type |
| Options | String key + number/string/color value |

There are no callbacks, no event listeners, no observer patterns. The API is strictly **call → response**.

## Python API

The Python binding (`gmsh.py`) is generated alongside all others and wraps the C API through `ctypes`. It is not a hand-written Pythonic wrapper — it is a direct, mechanical translation.

### What Python Gains

**Scripting ecosystem integration.** Python is where the rest of the simulation pipeline lives. OpenSeesPy, numpy, scipy, matplotlib — all accessible in the same script. No file-based handoff between mesh generation and solver setup:

```python
import gmsh
import openseespy.opensees as ops
import numpy as np

gmsh.initialize()
# ... build geometry, mesh ...
node_tags, coords, _ = gmsh.model.mesh.getNodes()
coords = np.reshape(coords, (-1, 3))

ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)
for tag, xyz in zip(node_tags, coords):
    ops.node(int(tag), *xyz)
```

**Numpy integration.** When numpy is available, the Python binding returns numpy arrays instead of Python lists. This happens transparently — the generated code checks for numpy at import time:

```python
# From the generated gmsh.py:
try_numpy = True
use_numpy = False
if try_numpy:
    try:
        import numpy
        use_numpy = True
    except:
        pass
```

When `use_numpy = True`, output vectors are returned as `numpy.ndarray` via `numpy.ctypeslib.as_array()` — a **zero-copy view** of the C memory. This means `getNodes()` on a million-node mesh returns a numpy array without copying the data.

**Parametric studies.** Loops, conditionals, function definitions — all native Python. A parametric mesh convergence study is a for-loop, not a separate templating system:

```python
for mesh_size in [0.5, 0.25, 0.1, 0.05]:
    gmsh.clear()
    build_model()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(3)
    # ... extract and analyze ...
```

**Error handling.** Python exceptions propagate naturally. A failed meshing operation raises an exception you can catch, log, and recover from — rather than silently producing a partial mesh.

### What Python Loses

**Performance overhead on the call boundary.** Every API call crosses Python → ctypes → C → C++. For bulk operations (`getNodes`, `generate`) the overhead is negligible because the work happens in C++. But for tight loops of small calls, the overhead accumulates:

```python
# Slow — N API calls through ctypes:
for tag in entity_tags:
    nodes, coords, _ = gmsh.model.mesh.getNodes(dim, tag)

# Better — one API call, filter in numpy:
all_nodes, all_coords, _ = gmsh.model.mesh.getNodes()
```

**GIL constraint.** Python's Global Interpreter Lock means you cannot truly parallelize Gmsh API calls from multiple threads. The meshing itself runs in C++ and releases the GIL, but the API call/return marshalling is single-threaded. For parallel workflows, use multiprocessing (separate Gmsh instances) rather than threading.

**No direct memory access to internal structures.** You work through the API's elementary-type contract. You cannot, for example, access Gmsh's internal half-edge data structure, walk the BRep adjacency graph in-memory, or modify mesh node coordinates without a full get/set round-trip. Everything goes through the serialization boundary.

**Type looseness.** The C API uses strict integer/double types. The Python binding accepts Python's dynamic types and converts via ctypes, which can silently truncate or misinterpret values. A float where an int is expected won't raise an error at the Python level — it will be cast.

### The ctypes Architecture

Understanding the binding mechanism helps explain its behavior:

```
Python script
    │
    ▼
gmsh.py (generated)       ← Python-side marshalling
    │                        Converts Python types to ctypes
    ▼
libgmsh.so / gmsh.dll     ← Shared library
    │                        C API entry points (gmshc.cpp)
    ▼
Gmsh C++ internals         ← Actual implementation
```

The generated `gmsh.py` uses `ctypes.CDLL` to load the Gmsh shared library and calls functions by name. Input arguments are marshalled to C types; output arguments are allocated as ctypes pointers, passed by reference, and unmarshalled on return.

This means the Python API is **as stable as the C API** — both are generated from the same source. No version drift between languages.

## FLTK GUI

Gmsh includes a built-in graphical interface built on FLTK (Fast Light Toolkit). The GUI is controlled through the `gmsh.fltk` module.

### What the GUI Provides

**Visualization.** Real-time 3D rendering of geometry and mesh. Zoom, rotate, pan. Display element quality, physical groups, partitions. Overlay post-processing views (scalar/vector/tensor fields) on the mesh.

**Interactive entity selection.** The API exposes this:

```python
# Open GUI and let user click on entities
gmsh.fltk.initialize()
r, dimTags = gmsh.fltk.selectEntities()
# r = 1 if user selected, 0 if cancelled
# dimTags = list of (dim, tag) pairs the user clicked
```

This is useful for debugging — "which entity is that face?" — but not for production workflows.

**Option editing.** The GUI exposes Gmsh's option tree (`Tools > Options`), letting you tweak meshing parameters visually and see immediate results. Useful for exploring parameter sensitivity.

**Window management.** Split views, multiple windows, status messages:

```python
gmsh.fltk.splitCurrentWindow("h", 0.5)  # horizontal split
gmsh.fltk.setStatusMessage("Meshing complete")
```

### What the GUI Cannot Do

The GUI is a **viewer and tweaker**, not a workflow tool. Here is what it lacks compared to the API:

**No scripting.** The GUI cannot execute a sequence of operations. There is no macro recorder, no command history, no "replay this workflow." Every action is manual and one-shot.

**No batch processing.** You cannot queue multiple models, run parametric sweeps, or automate mesh generation through the GUI. It processes one model at a time, interactively.

**No programmatic mesh extraction.** The GUI can display mesh data but cannot export it to a running program's memory. To get nodes and elements into your solver, you need the API — the GUI can only write to files.

**No headless operation.** The FLTK module requires a display server. It cannot run on headless servers, in Docker containers without X forwarding, or in CI/CD pipelines. The API works everywhere; the GUI does not.

**Main thread constraint.** All `gmsh.fltk` calls must happen from the main thread. In a multi-threaded application, you must route GUI updates through `gmsh.fltk.awake()`:

```python
# From a worker thread — request GUI update:
gmsh.fltk.awake("update")

# From the main thread — process the update:
gmsh.fltk.wait(0.1)
```

**Limited automation.** The `gmsh.fltk` module has 15 functions total. Compare this to `gmsh.model.mesh` (50+ functions) or `gmsh.model.occ` (40+ functions). The GUI API is intentionally minimal — enough to display and interact, not enough to build workflows.

### When to Use the GUI

In the apeGmsh context, the GUI serves exactly two purposes:

1. **Debugging geometry.** When a boolean operation produces unexpected results or a mesh looks wrong, `gmsh.fltk.run()` lets you visually inspect the model. This is faster than writing diagnostic code.

2. **Exploring parameters.** When tuning mesh size fields or algorithm choices, the GUI's immediate visual feedback helps find good values before hardcoding them in the script.

For everything else — mesh generation, data extraction, solver integration, parametric studies — the Python API is the correct interface.

## ONELAB

Gmsh includes a parameter server called ONELAB (Open Numerical Engineering LABoratory). It is a **key-value store** shared between Gmsh and external solvers, designed for coupling simulations.

```python
# Set a parameter
gmsh.onelab.setNumber("MyParam/meshSize", [0.1])

# Read it back (possibly modified by another tool)
value = gmsh.onelab.getNumber("MyParam/meshSize")
```

ONELAB parameters can be typed (number, string), given ranges and choices, and exposed in the GUI as sliders, dropdowns, or text fields. The `gmsh.onelab.run()` function can launch external solvers that read these parameters.

For apeGmsh, ONELAB is largely irrelevant — we couple Gmsh and OpenSees directly through Python, not through a parameter server. But it is worth knowing it exists, because ONELAB parameters appear in `.geo` files and in the GUI's parameter panel.

## Cross-Reference

Related architecture documents:

- [[gmsh_basics]] — BRep model, session lifecycle, options, physical groups
- [[gmsh_geometry_basics]] — Geometry construction and (dim,tag) tracking
- [[gmsh_meshing_basics]] — BRep-mesh duality, size control, meshing workflow
- [[gmsh_meshing_advanced]] — Algorithms, fields, optimization, embedded entities
- [[gmsh_partitioning]] — METIS partitioning, partition entities, OpenSeesMP pipeline
