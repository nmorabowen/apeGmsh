# apeGmsh sections

A guide to the parametric section builders — creating structural
cross-sections directly in the session with automatic labeling of
flanges, webs, and end faces.

Grounded in the current source:

- `src/apeGmsh/sections/_builder.py` — `SectionsBuilder` composite
- `src/apeGmsh/sections/solid.py` — solid-element section geometry
- `src/apeGmsh/sections/shell.py` — shell-element section geometry

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="frame")
g.begin()
```


## 1. What sections are

A structural section is a prismatic member defined by its
cross-sectional shape (I-beam, rectangle, channel) and a length.
apeGmsh builds these as 3D solid or 2D shell geometry directly in
the session, with named sub-regions (flanges, web, end faces) that
constraints and loads can target by label.

Sections are accessed via `g.sections`:

```python
col = g.sections.W_solid(
    bf=150, tf=20, h=300, tw=10, length=2000,
    label="col",
)
```

This creates an I-beam cross-section extruded along Z with:
- `col.labels.top_flange` → `"col.top_flange"`
- `col.labels.bottom_flange` → `"col.bottom_flange"`
- `col.labels.web` → `"col.web"`
- `col.labels.start_face` → `"col.start_face"` (z=0 end)
- `col.labels.end_face` → `"col.end_face"` (z=length end)


## 2. Sections vs Parts

Both create geometry with labels. The difference:

| | Part | Section |
|---|---|---|
| **Session** | Own isolated Gmsh session | Built directly in assembly session |
| **Persistence** | Auto-saves to STEP + sidecar JSON | No file — exists only in session |
| **Reuse** | Import many times with transforms | Build once per instance |
| **Labels** | Via COM-matching after STEP import | Created natively in session |
| **Use when** | Same geometry in many locations | One-off parametric members |

Use **Parts** when you have a column design that repeats 20 times
at different locations. Use **sections** when you have one beam with
specific dimensions that appears once.

In practice, sections are convenient for quick prototyping. Parts
are better for production assemblies because they survive session
restarts (the STEP file is on disk).


## 3. Available section types

### W_solid — Wide-flange solid section

```python
col = g.sections.W_solid(
    bf=150,      # flange width (mm)
    tf=20,       # flange thickness (mm)
    h=300,       # web height (clear distance between flanges)
    tw=10,       # web thickness (mm)
    length=2000, # extrusion length (mm)
    label="col",
    lc=50,                          # target element size (optional)
    translate=(0, 0, 0),            # position (optional)
    rotate=(1.5708, 0, 0, 1),      # rotation (optional)
)
```

Creates a 3D solid I-beam extruded along Z. Cross-section built by
subtracting two rectangular voids from an outer rectangle, then
sliced at the flange-web boundaries to create labeled sub-volumes.

**Total section height:** `2*tf + h` (flanges + clear web height).

**Labels created:**
- `{label}.top_flange` — upper flange volume(s)
- `{label}.bottom_flange` — lower flange volume(s)
- `{label}.web` — web volume(s)
- `{label}.start_face` — surface at z=0
- `{label}.end_face` — surface at z=length

### rect_solid — Rectangular solid section

```python
beam = g.sections.rect_solid(
    b=200,       # width (mm)
    h=400,       # height (mm)
    length=3000, # extrusion length (mm)
    label="beam",
)
```

Creates a simple rectangular prism. Labels:
- `{label}.body` — the single volume
- `{label}.start_face` — z=0 end
- `{label}.end_face` — z=length end

### W_shell — Wide-flange shell section (mid-surfaces)

```python
col_shell = g.sections.W_shell(
    bf=150, tf=20, h=300, tw=10, length=2000,
    label="col_shell",
)
```

Same parameters as `W_solid` but creates **mid-surface** shell
geometry instead of solid volumes. The flanges and web are
represented as surfaces at their mid-plane locations.

**Labels created:**
- `{label}.top_flange` — upper flange surface
- `{label}.bottom_flange` — lower flange surface
- `{label}.web` — web surface
- Start/end edges rather than faces (1D boundaries)

Use this when meshing with shell elements (dim=2) instead of
solid elements (dim=3).


## 4. Positioning and orientation

Sections are built at the origin and can be positioned with
`translate` and `rotate`:

```python
# Column at (5, 0, 0), rotated 90 degrees about Z
col = g.sections.W_solid(
    bf=150, tf=20, h=300, tw=10, length=3000,
    label="col_A",
    translate=(5000, 0, 0),
    rotate=(1.5708, 0, 0, 1),  # 90 deg about Z axis
)
```

The rotation uses the same convention as `g.model.transforms.rotate`:
`(angle_rad, ax, ay, az)` for rotation about an axis through the
origin, or `(angle_rad, ax, ay, az, cx, cy, cz)` for rotation about
a point.


## 5. Element sizing

The `lc` parameter sets the target element size on the section's
BRep points. It works alongside `g.mesh.sizing.set_global_size()`:

```python
col = g.sections.W_solid(
    bf=150, tf=20, h=300, tw=10, length=2000,
    label="col", lc=30,  # fine mesh on column
)
# Global mesh is coarser
g.mesh.sizing.set_global_size(100)
g.mesh.generation.generate(3)
```

The default `lc=1e22` imposes no local constraint — the element
size is governed purely by the global size.


## 6. Using labels for constraints and loads

The real power of sections is that every sub-region has a name
you can target:

```python
# Column section
col = g.sections.W_solid(
    bf=150, tf=20, h=300, tw=10, length=3000,
    label="col",
)

# Fix the bottom face
g.physical.add_surface(
    g.labels.entities("col.start_face"), name="Base"
)

# Apply gravity
with g.loads.pattern("dead"):
    g.loads.gravity("col.web", density=7850)
    g.loads.gravity("col.top_flange", density=7850)
    g.loads.gravity("col.bottom_flange", density=7850)

# Constrain slab to column top
g.constraints.equal_dof("col.end_face", "slab.bottom", dofs=[1, 2, 3])
```


## 7. Complete example

```python
from apeGmsh import apeGmsh

with apeGmsh("portal_frame") as g:
    # Columns
    left_col = g.sections.W_solid(
        bf=150, tf=20, h=300, tw=10, length=3000,
        label="left_col", translate=(0, 0, 0),
    )
    right_col = g.sections.W_solid(
        bf=150, tf=20, h=300, tw=10, length=3000,
        label="right_col", translate=(6000, 0, 0),
    )

    # Beam
    beam = g.sections.rect_solid(
        b=200, h=400, length=6000,
        label="beam",
        translate=(0, 0, 3000),
        rotate=(1.5708, 0, 1, 0),  # rotate to span X direction
    )

    # Fragment for conformal mesh at joints
    g.parts.fragment_all()

    # Mesh
    g.mesh.sizing.set_global_size(50)
    g.mesh.generation.generate(3)

    # FEM data
    fem = g.mesh.queries.get_fem_data(dim=3)
    print(fem.inspect.summary())
```


## See also

- `guide_parts_assembly.md` — Part-based workflow for reusable geometry
- `guide_basics.md` — geometry primitives and boolean operations
- `guide_constraints.md` — constraining sections to each other
