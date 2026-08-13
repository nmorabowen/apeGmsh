> **Absorbed** into `docs/concepts/sections.md` (ADR 0079 P2-B). This guide is retained as working memory; the concepts page is canonical.

# apeGmsh sections

A guide to the parametric section builders — creating structural
cross-sections directly in the session with automatic labeling of
flanges, webs, and end faces.

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="frame")
g.begin()
```


## Tasks on this page

- [Position and orient a section](#4-position-and-orient-a-section) · [Set element size](#5-set-element-size) · [Target sub-regions with constraints and loads](#6-using-labels-for-constraints-and-loads) · [Build a portal frame end-to-end](#7-complete-example)


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

### rect_hollow — Hollow rectangular tube (HSS)

```python
hss = g.sections.rect_hollow(
    b=200,       # outer width (mm)
    h=300,       # outer height (mm)
    t=10,        # wall thickness (mm)
    length=3000,
    label="hss",
)
```

Builds an HSS by cutting a smaller box from a larger one. Labels:
- `{label}.body` — the single hollow volume
- `{label}.start_face`, `{label}.end_face` — z=0 / z=length

### pipe_solid — Solid circular bar

```python
bar = g.sections.pipe_solid(
    r=50,        # radius (mm)
    length=2000,
    label="bar",
)
```

Labels: `{label}.body`, `{label}.start_face`, `{label}.end_face`.

### pipe_hollow — Hollow circular pipe

```python
pipe = g.sections.pipe_hollow(
    r_outer=80,  # outer radius (mm)
    t=8,         # wall thickness (mm)
    length=2000,
    label="pipe",
)
```

Built by cutting an inner cylinder from an outer cylinder.
Labels: `{label}.body`, `{label}.start_face`, `{label}.end_face`.

### angle_solid — L-shape (angle)

```python
angle = g.sections.angle_solid(
    b=100,       # horizontal leg width (mm)
    h=150,       # vertical leg height (mm)
    t=10,        # thickness of both legs (mm)
    length=3000,
    label="L",
)
```

Built by fusing two rectangles, extruding, and slicing along the
inside corner so each leg is its own labeled volume.

**Labels created:**
- `{label}.horizontal_leg` — the X-direction leg
- `{label}.vertical_leg` — the Y-direction leg
- `{label}.horizontal_leg_face`, `{label}.vertical_leg_face` — outer-skin
  faces from `classify_angle_outer_faces`
- `{label}.start_face`, `{label}.end_face`

### channel_solid — C-shape (channel)

```python
chan = g.sections.channel_solid(
    bf=80,       # flange width (mm)
    tf=10,       # flange thickness (mm)
    h=180,       # clear web height (mm)
    tw=8,        # web thickness (mm)
    length=3000,
    label="C",
)
```

Same `bf / tf / h / tw` parameter set as `W_solid`, sliced into
flange and web sub-volumes.

**Labels created:**
- `{label}.top_flange`, `{label}.bottom_flange`, `{label}.web`
- Outer-face labels from `classify_w_outer_faces`
- `{label}.start_face`, `{label}.end_face`

### tee_solid — T-shape (tee)

```python
tee = g.sections.tee_solid(
    bf=120,      # flange width (mm)
    tf=12,       # flange thickness (mm)
    h=180,       # stem height (mm)
    tw=10,       # stem thickness (mm)
    length=3000,
    label="T",
)
```

Built by fusing a flange rectangle with a stem rectangle and slicing
at the stem boundaries.

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

Use this when meshing with shell elements (dim=2) instead of
solid elements (dim=3).


## 4. Position and orient a section

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


## 5. Set element size

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
with g.loads.case("dead"):
    g.loads.gravity("col.web", density=7850)
    g.loads.gravity("col.top_flange", density=7850)
    g.loads.gravity("col.bottom_flange", density=7850)

# Constrain slab to column top
g.constraints.equal_dof("col.end_face", "slab.bottom", dofs=[1, 2, 3])
```

For the full recipes behind these snippets, see
[Fix supports and BCs](../how-to/supports-bcs.md) and
[Apply gravity](../how-to/gravity.md).


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

## Cross-section property analyzer (`SectionProperties`, ADR 0078)

The builders above produce *geometry*; the analyzer computes the
*numbers* — the full `sectionproperties`-class capability set, natively
in-process on any meshed 2-D face.

### Flat-face builders

Every solid recipe has a flat-face sibling (same shape parameters minus
`length`/`anchor`/`align`; in-plane `translate=(dx, dy)` and scalar
`rotate` in degrees; auto-PG named after `label`):
`W_face`, `rect_face`, `rect_hollow_face`, `pipe_face`,
`pipe_hollow_face`, `angle_face`, `channel_face`, `tee_face`.
Any meshed face works — raw OCC, `load_dxf`, STEP imports are equal
routes; the builders are convenience, not a requirement.

### Analyzing

```python
g.sections.W_face(bf=400.0, tf=25.0, h=1200.0, tw=12.0, label="girder")
g.mesh.sizing.set_global_size(15.0)
g.mesh.generation.generate(dim=2)
g.mesh.generation.set_order(2)          # tri6 — warping-grade
fem = g.mesh.queries.get_fem_data(dim=2)

from apeGmsh import SectionProperties
from apeGmsh.sections import SectionMaterial
sec = SectionProperties(fem, materials={"girder": SectionMaterial(E=200e3, nu=0.3, fy=345.0)},
                        name="PG1200x400")
geo, warp, plas = sec.geometric(), sec.warping(), sec.plastic()
sec.stress(N=-800e3, Vy=350e3, Mxx=1.9e9).plot("von_mises")
sec.viewer(blocking=False)              # Qt inspector (notebooks: blocking=False)
```

Plotting (matplotlib, headless-safe): `sec.plot()` one-call overview
(glyph view + summary panel), `sec.plot_mesh()` / `sec.plot_section()`,
`sec.plot_warping(shear_flow=True)` (ω contour + unit-torsion shear
flow), `stress(...).plot(component)` contours,
`stress(...).plot_vector(action=)` τ quiver,
`stress(...).plot_mohrs_circle(at=(x, y), pg=)`, and the pre-mesh
`g.sections.plot_faces()` geometry preview.

Naming law: rigidity-form fields (`EA`, `EIxx_c`, `GJ`, …) are always
valid; unprefixed accessors (`Ixx_c`, `J`, `Sxx`, …) divide by the
single modulus and raise `CompositeSectionError` on composites — pick a
reference with `transformed(e_ref=...)`. Omit `materials=` for
geometric-only mode (classic numbers).

Composite faces must **partition** the section into disjoint PGs: carve
the inner shape out of the outer (`g.model.boolean.cut(...,
remove_tool=False)`) before `g.parts.fragment_pair(...)` — see the SRC
worked example in ADR 0078.

### OpenSees handoff

```python
girder = p.section.ComputedSection(analysis=sec)   # lazy; resolves at emit
integ  = p.beamIntegration.Lobatto(section=girder, n_ip=5)
p.element.forceBeamColumn(pg="girders", transf=transf, integration=integ)
# or eager: sec.to_elastic_section(E=..., G=..., ndm=3)
```

One shared lowering owns the axis mapping (authoring x ≡ local z,
y ≡ local y → `Ixx_c→Iz`, `Iyy_c→Iy`, `As_y/A→alphaY`,
`As_x/A→alphaZ`); composites require explicit reference `E=`/`G=`
(transformed-section constants, fail-loud at emit otherwise).

## Section documents and the builder GUI (`SectionDocument`, ADR 0080)

A `SectionDocument` is versioned JSON describing one cross-section, and it
is the source of truth: the headless API below and the Qt builder are both
clients of it (the **parity law** — every GUI action is a document
mutation). Two lanes, one per document:

```python
from apeGmsh.sections import SectionDocument, launch_builder

launch_builder()                          # blank continuum document
launch_builder("col500.section.json")     # notebooks: blocking=False
doc = SectionDocument.open("col500.section.json")
```

### Continuum lane

Parametric shapes (the eight `*_face` builders 1:1), freehand polygons,
booleans, per-region materials, mesh prefs, `disconnected` policy;
`build()` runs a private session and returns a `SectionProperties`.

```python
doc = SectionDocument.new(name="src600")
doc.set_material("concrete", E=25e3, nu=0.2)
doc.set_material("steel", E=200e3, nu=0.3, fy=345.0)
doc.add_shape("rect_face", id="concrete", b=600.0, h=600.0)
doc.add_shape("W_face", id="steel", bf=300.0, tf=20.0, h=360.0, tw=12.0)
doc.add_embed("concrete", "steel")     # cut-then-fragment, one step
doc.set_mesh(lc=20.0, order=2)
sec = doc.build()
```

`add_embed` is the composite-partition primitive: the double-cover trap
documented above is **not expressible** through it. Raw `add_cut` /
`add_fragment_pair` stay available for non-overlapping compositions.
`add_bar` / `add_bar_line` attach a discrete rebar overlay resolved at
handoff through the `kind="fiber"` lowering (ADR 0080 B3).

### Fiber lane

Patches / layers / points plus parametric RC templates —
`rc_rect_column`, `rc_circ_column`, `rc_beam` — stored as parameters and
re-expanded on every build, so editing `cover` moves every bar.
`core_split=True` partitions the concrete exactly into confined core +
cover shell (you assign the materials; the template never computes
confinement parameters). `build()` returns a `FiberRecipe`;
`doc.to_section(ops)` registers it as an `ops.section.Fiber`.

Traps: `cover` is to the **bar centre**; `bars_x`/`bars_y` **include
corners** (4/4 is 12 bars, not 16).

### Moment–curvature

```python
from apeGmsh.sections import moment_curvature

mc = moment_curvature(doc, axis="z", kappa_max=8e-5, n_steps=60, axial=-1500e3)
mc.EI0, mc.M_max, mc.complete
```

Fiber lane only. `axis="z"` is DOF 6 (`EIxx_c`), `"y"` is DOF 5
(`EIyy_c`); `kappa_max` is signed; `axial` is OpenSees-signed
(compression negative), applied first and held. **It wipes the global
OpenSees domain** and runs synchronously on the calling thread — do not
move it to a worker (openseespy is a non-reentrant process-global
runtime; the B6 properties worker is threaded only because it drives
Gmsh).

### Export and handoff

`doc.export_script(path)` writes the equivalent plain apeGmsh script
(one-way). `handoff_snippet(doc, path=...)` writes the few paste-ready
lines that put the section on a bridge — the GUI's **Copy handoff**
button. Both render fiber sections through the same helpers, so a
snippet and an export can never disagree.

### Optional dependencies (both fail-soft, never required headless)

- **apeSteel** → the catalog picker prefills the `W_face` form from AISC
  v16 / EN designations, in millimetres. Absent → the picker is hidden.
  Enumeration reads a private apeSteel table and degrades to a free-text
  field if that shape ever changes.
- **openseespy / the Ladruno fork** → the M–κ button. Absent → greyed
  with guidance.

??? note "For maintainers — source map"
    - `src/apeGmsh/sections/_builder.py` — `SectionsBuilder` composite
    - `src/apeGmsh/sections/solid.py` — solid-element section geometry
    - `src/apeGmsh/sections/shell.py` — shell-element section geometry
    - `src/apeGmsh/sections/_analysis.py` — `SectionProperties` broker (ADR 0078)
    - `src/apeGmsh/sections/_lowering.py` — the single authoring→OpenSees mapping
    - `src/apeGmsh/opensees/section/computed.py` — `ComputedSection` primitive
    - `src/apeGmsh/sections/_inspector.py` — Qt section inspector (`sec.viewer()`)
    - `src/apeGmsh/sections/_document.py` — `SectionDocument` (ADR 0080 B1/B2)
    - `src/apeGmsh/sections/_rc_templates.py` — the three RC generators
    - `src/apeGmsh/sections/_script_export.py` — `export_script` + the shared
      fiber-rendering helpers
    - `src/apeGmsh/sections/_handoff.py` — `handoff_snippet` (B7)
    - `src/apeGmsh/sections/_mc.py` — `moment_curvature` harness (B7)
    - `src/apeGmsh/sections/_catalog.py` — apeSteel catalog bridge (B7)
    - `src/apeGmsh/sections/_builder_gui.py` — the Qt builder (B5/B6/B7)
    - `src/apeGmsh/sections/_drafting.py` — Qt-free snap / ortho / typed input
    - `src/apeGmsh/sections/_properties.py` — B6 build controller (solves
      on a worker thread; meshes out of process)
    - `src/apeGmsh/sections/_mesh_proc.py` / `_mesh_worker.py` — the
      child-process meshing seam. Gmsh is one process-global,
      non-reentrant C++ runtime: driving it from a background thread of
      a process that also drives it from the main thread aborts the
      interpreter with no traceback. `_session.py`'s runtime lock
      serializes every other threaded use; the worker meshes elsewhere
      instead, because a session left open would hold that lock for its
      whole lifetime and the panel would never refresh.
