# apeGmsh transforms

A guide to the transform and sweep operations in apeGmsh — translating,
rotating, scaling, mirroring, copying, extruding, revolving, sweeping
along paths, and lofting through sections. These operations live on the
`g.model.transforms` composite and act directly on Gmsh OCC entities.
This document is written for structural engineers who work with FEM
daily but may not be familiar with Gmsh's OpenCASCADE kernel; it
explains each operation in terms of the structural modeling problems it
solves.

The guide is grounded in the current source:

- `src/apeGmsh/core/_model_transforms.py` — the `_Transforms` class
  that exposes every method described here

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
# ... geometry created ...
```


## 1. How transforms work in apeGmsh

Every transform method lives on `g.model.transforms`. That object is a
thin, ergonomic wrapper around Gmsh's OCC kernel calls
(`gmsh.model.occ.translate`, `gmsh.model.occ.rotate`, etc.) with three
additions that matter for day-to-day use:

1. **Flexible tag input.** You can pass a bare integer, a list of
   integers, or a list of `(dim, tag)` tuples. The library normalizes
   them internally through `_as_dimtags`, so you never have to think
   about the `(dim, tag)` pair format that raw Gmsh demands.

2. **Automatic synchronization.** Every method accepts `sync=True` (the
   default). When set, it calls `gmsh.model.occ.synchronize()` after
   the operation, which makes the change visible to Gmsh immediately.
   You can pass `sync=False` to defer synchronization when you are
   chaining several transforms and want to pay the sync cost only once
   at the end — useful in tight loops over hundreds of entities.

3. **Method chaining.** The five rigid transforms (`translate`,
   `rotate`, `scale`, `mirror`, `copy`) return `self`, so you can
   chain calls: `g.model.transforms.translate(...).rotate(...)`. The
   generative operations (`extrude`, `revolve`, `sweep`,
   `thru_sections`) return the list of newly created `(dim, tag)`
   pairs instead, because you almost always need those tags for
   subsequent operations.

There are two families of operations on the composite:

- **Rigid transforms** — `translate`, `rotate`, `scale`, `mirror`,
  `copy`. These reposition or duplicate existing entities without
  creating new topological dimensions. A volume stays a volume; a
  curve stays a curve.

- **Generative sweeps** — `extrude`, `revolve`, `sweep`,
  `thru_sections`. These create geometry one dimension up: a point
  becomes a curve, a curve becomes a surface, a surface becomes a
  volume. They are the primary way to build 3-D structural volumes
  from 2-D profiles.


## 2. Shared parameters

Several parameters appear on every transform method. Understanding them
once saves you from re-reading each signature:

- **`tags`** — the entities to act on. Accepts a single integer tag, a
  list of integer tags, a single `(dim, tag)` tuple, or a list of
  such tuples. When you pass bare integers, the `dim` parameter
  decides which dimension they refer to.

- **`dim`** — default dimension for bare integer tags. The rigid
  transforms default to `dim=3` (volumes); `extrude` and `revolve`
  default to `dim=2` (surfaces), because you most often extrude a
  surface into a volume. If your tags are already `(dim, tag)` tuples,
  this parameter is ignored.

- **`sync`** — whether to synchronize the OCC kernel after the call.
  Default `True`. Set to `False` when batching many transforms, then
  call `gmsh.model.occ.synchronize()` yourself at the end.

Two important behavioral notes apply to all transforms:

- **In-place modification.** Every transform except `copy` modifies the
  entities in place. `translate(box, 5, 0, 0)` moves `box` — it does
  not create a second box at the new position. The original tag
  remains valid and refers to the entity at its new location.

- **Labels survive.** Because the entity tag does not change,
  any label or physical group membership you assigned before the
  transform remains intact. You can label a column, translate it into
  position, and the label still resolves to the same volume.

- **Sub-entity propagation.** Transforms affect the entire topological
  hierarchy. Translating a volume moves its bounding surfaces, their
  bounding curves, and their endpoints. You never need to translate
  sub-entities separately.


## 3. translate — positioning parts in space

`translate` shifts entities by a displacement vector `(dx, dy, dz)`.
This is the workhorse for assembly workflows: you create a part at the
origin, then translate copies to their grid positions.

```python
g.model.transforms.translate(tags, dx, dy, dz, *, dim=3, sync=True)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tags` | `TagsLike` | — | Entities to move |
| `dx`, `dy`, `dz` | `float` | — | Translation vector components |
| `dim` | `int` | `3` | Default dimension for bare integer tags |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `self` (for chaining).

The simplest structural use case is placing a column. You model the
column once at the origin and then translate it to each grid
intersection:

```python
col = g.model.geometry.add_box(0, 0, 0, 0.4, 0.4, 3.5, label="col_A1")

# move it to grid position (6.0, 0.0, 0.0)
g.model.transforms.translate(col, 6.0, 0.0, 0.0)
```

For a column grid, combine `copy` and `translate` (see section 7 for
the full pattern). The key insight is that `translate` is an in-place
operation — if you forget to copy first, you will move the original
instead of creating a second instance.


## 4. rotate — orienting structural members

`rotate` turns entities around an arbitrary axis passing through a
point. You specify the rotation center `(cx, cy, cz)`, the axis
direction `(ax, ay, az)`, and the sweep angle in radians.

```python
g.model.transforms.rotate(
    tags, angle, *,
    ax=0.0, ay=0.0, az=1.0,
    cx=0.0, cy=0.0, cz=0.0,
    dim=3, sync=True,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tags` | `TagsLike` | — | Entities to rotate |
| `angle` | `float` | — | Rotation angle in **radians** |
| `ax`, `ay`, `az` | `float` | `0, 0, 1` | Axis direction vector |
| `cx`, `cy`, `cz` | `float` | `0, 0, 0` | Point on the rotation axis |
| `dim` | `int` | `3` | Default dimension for bare integer tags |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `self` (for chaining).

The axis defaults point are worth memorizing. With the defaults
`(ax, ay, az) = (0, 0, 1)` and `(cx, cy, cz) = (0, 0, 0)`, you get
a rotation around the global Z axis through the origin — the most
common case for plan-view rotations of structural elements.

**The 4-argument form** uses the defaults and is the concise version
for simple rotations:

```python
import math

# Rotate a brace 45 degrees about global Z at the origin
g.model.transforms.rotate(brace, math.pi / 4)
```

**The 7-argument form** specifies an arbitrary axis and center, which
you need when rotating around a member's own axis or around a point
that is not the origin:

```python
import math

# Rotate a beam 90 degrees around its own longitudinal axis (X)
# passing through the beam's start point at (3.0, 0.0, 0.0)
g.model.transforms.rotate(
    beam, math.pi / 2,
    ax=1.0, ay=0.0, az=0.0,
    cx=3.0, cy=0.0, cz=0.0,
)
```

Angles are always in radians. Use `math.radians(45)` or
`math.pi / 4` — never pass `45` and wonder why your column ended up
at an impossible orientation. The library logs the angle in degrees
for readability, but the input is strictly radians.


## 5. scale — parametric studies and unit conversion

`scale` applies a dilation (uniform or non-uniform) centered at a
point. Each axis can have a different scale factor.

```python
g.model.transforms.scale(
    tags, sx, sy, sz, *,
    cx=0.0, cy=0.0, cz=0.0,
    dim=3, sync=True,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tags` | `TagsLike` | — | Entities to scale |
| `sx`, `sy`, `sz` | `float` | — | Scale factors per axis |
| `cx`, `cy`, `cz` | `float` | `0, 0, 0` | Center of dilation |
| `dim` | `int` | `3` | Default dimension for bare integer tags |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `self` (for chaining).

Uniform scaling (`sx == sy == sz`) preserves shape and is the safe
choice for parametric studies and unit conversion. Non-uniform scaling
distorts the geometry and should be used with care — it will turn
circles into ellipses and squares into rectangles.

Two practical uses come up regularly in structural work:

**Unit conversion.** You imported a STEP file modeled in millimeters
but your analysis runs in meters:

```python
# Scale everything from mm to m (all entities in the model)
all_vols = [t for _, t in gmsh.model.getEntities(3)]
g.model.transforms.scale(all_vols, 0.001, 0.001, 0.001)
```

**Parametric column study.** You want to run the same model with column
sections scaled by a factor:

```python
factor = 1.25  # 25% larger cross-section
g.model.transforms.scale(column, factor, factor, 1.0,
                          cx=col_x, cy=col_y, cz=0.0)
```

Here `sz=1.0` preserves the column height while `sx` and `sy` enlarge
the cross-section. The center `(col_x, col_y, 0.0)` ensures the
column scales outward from its own axis rather than from the origin.

Note that `scale` wraps `gmsh.model.occ.dilate`, which is the OCC
term for this operation. The apeGmsh name `scale` is more familiar to
engineers.


## 6. mirror — exploiting structural symmetry

`mirror` reflects entities through a plane defined by the equation
`ax + by + cz + d = 0`. This is the operation that turns a half-model
into a full model.

```python
g.model.transforms.mirror(tags, a, b, c, d, *, dim=3, sync=True)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tags` | `TagsLike` | — | Entities to mirror |
| `a`, `b`, `c`, `d` | `float` | — | Plane equation coefficients |
| `dim` | `int` | `3` | Default dimension for bare integer tags |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `self` (for chaining).

The plane equation can feel abstract, but the common cases are simple:

| Plane | Equation | `(a, b, c, d)` |
|-------|----------|-----------------|
| YZ plane (x = 0) | `1x + 0y + 0z + 0 = 0` | `(1, 0, 0, 0)` |
| XZ plane (y = 0) | `0x + 1y + 0z + 0 = 0` | `(0, 1, 0, 0)` |
| XY plane (z = 0) | `0x + 0y + 1z + 0 = 0` | `(0, 0, 1, 0)` |
| Plane x = 5 | `1x + 0y + 0z - 5 = 0` | `(1, 0, 0, -5)` |

Mirror is an in-place operation: it moves the original entities to
their reflected positions. If you want to keep the original and create
a reflected copy, you must `copy` first:

```python
# Build half a bridge, then mirror to get the full model
half_tags = [deck, pier_left, cable_left]

mirrored = g.model.transforms.copy(half_tags)
g.model.transforms.mirror(mirrored, 1, 0, 0, -L/2)
# Original half is at x < L/2, mirror is at x > L/2
```

This copy-then-mirror pattern is the standard workflow for symmetric
structures: you model one half (or one quarter) of the structure, get
it right, then mirror to produce the rest. The d coefficient offsets
the plane from the origin — here, `d = -L/2` places the mirror plane
at `x = L/2`, the midspan of the bridge.

After mirroring, the original and the copy are separate entities. If
you need displacement continuity at the symmetry plane, add an
`equal_dof` constraint between the two halves (see `guide_constraints.md`).


## 7. copy — duplicating entities

`copy` creates duplicates of existing entities. Unlike every other
transform, it does not modify the originals — it returns a list of new
tags pointing to identical entities at the same location.

```python
new_tags = g.model.transforms.copy(tags, *, dim=3, sync=True)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tags` | `TagsLike` | — | Entities to duplicate |
| `dim` | `int` | `3` | Default dimension for bare integer tags |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `list[Tag]` — tags of the newly created copies.

`copy` is almost never used alone. You copy, then immediately
translate (or rotate, or mirror) the copies into position. This is the
fundamental pattern for repetitive geometry — column grids, floor
plates, truss panels:

```python
# Create a 3x3 column grid at 6 m spacing
col = g.model.geometry.add_box(0, 0, 0, 0.4, 0.4, 3.5, label="col_origin")

spacing = 6.0
for i in range(3):
    for j in range(3):
        if i == 0 and j == 0:
            # The original is already at (0, 0) — just translate it
            g.model.transforms.translate(col, 0, 0, 0)
            continue
        copy_tag = g.model.transforms.copy(col)
        g.model.transforms.translate(copy_tag, i * spacing, j * spacing, 0)
```

A cleaner version uses a list to collect all column tags:

```python
col = g.model.geometry.add_box(0, 0, 0, 0.4, 0.4, 3.5, label="col")
columns = [col]

for i in range(3):
    for j in range(3):
        if i == 0 and j == 0:
            continue
        [c] = g.model.transforms.copy(col)
        g.model.transforms.translate(c, i * spacing, j * spacing, 0)
        columns.append(c)
```

The destructured assignment `[c] = ...` is handy when you copy a single
entity and want a plain integer tag rather than a one-element list.


## 8. extrude — sweeping profiles into solids

`extrude` is the most commonly used generative operation in structural
modeling. It sweeps entities along a straight vector, creating geometry
one dimension higher: a point becomes a curve, a curve becomes a
surface, a surface becomes a volume. Extruding a floor-plan surface
along the Z axis to create a slab is the textbook case.

```python
result = g.model.transforms.extrude(
    tags, dx, dy, dz, *,
    dim=2,
    num_elements=None,
    heights=None,
    recombine=False,
    sync=True,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tags` | `TagsLike` | — | Entities to extrude |
| `dx`, `dy`, `dz` | `float` | — | Extrusion vector |
| `dim` | `int` | `2` | Default dimension for bare integer tags |
| `num_elements` | `list[int]` or `None` | `None` | Structured layer counts |
| `heights` | `list[float]` or `None` | `None` | Relative heights per layer |
| `recombine` | `bool` | `False` | Use hex/quad instead of tet/tri |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `list[DimTag]` — all generated `(dim, tag)` pairs.

The return value deserves attention. For a surface extruded into a
volume, the list contains several entries in a fixed order:

- `result[0]` — the **top face** (a surface, the extruded copy of the
  input at the far end of the vector)
- `result[1]` — the **volume** created by the sweep
- `result[2:]` — the **lateral faces** (one per edge of the input
  surface)

This ordering lets you grab the volume for meshing and the top face for
applying boundary conditions without searching:

```python
# Extrude a floor outline into a 250 mm slab
slab_out = g.model.transforms.extrude(floor_surf, 0, 0, 0.25,
                                       num_elements=[4])
top_face  = slab_out[0]   # (2, tag) — for applying roof loads
slab_vol  = slab_out[1]   # (3, tag) — for material assignment
```

**Structured layers.** The `num_elements` and `heights` parameters
control through-thickness meshing. `num_elements=[10]` creates 10
uniform layers of elements through the extrusion. For non-uniform
layering — denser near the top and bottom of a slab, coarser in the
middle — combine `num_elements` with `heights`:

```python
# 3 layers: 30% of thickness with 4 elements, 40% with 2, 30% with 4
out = g.model.transforms.extrude(
    surf, 0, 0, 0.3,
    num_elements=[4, 2, 4],
    heights=[0.3, 0.7, 1.0],   # cumulative fractions, must end at 1.0
)
```

Note that `heights` are cumulative fractions of the total extrusion
length, and the list must reach `1.0`. The number of entries in
`heights` must match `num_elements`.

**Recombination.** Pass `recombine=True` to produce hexahedral (brick)
elements instead of tetrahedra. This requires structured layers
(`num_elements` must be set) and works best when the base surface mesh
is also quadrilateral. Hex meshes are preferred for many solid
mechanics problems because they exhibit less volumetric locking and
need fewer elements for the same accuracy.


## 9. revolve — axisymmetric parts

`revolve` sweeps entities around an axis, creating solids of
revolution. This is the operation for pipes, circular foundations,
cylindrical tanks, and any geometry with rotational symmetry.

```python
result = g.model.transforms.revolve(
    tags, angle, *,
    x=0.0, y=0.0, z=0.0,
    ax=0.0, ay=0.0, az=1.0,
    dim=2,
    num_elements=None,
    heights=None,
    recombine=False,
    sync=True,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tags` | `TagsLike` | — | Entities to revolve |
| `angle` | `float` | — | Sweep angle in **radians** |
| `x`, `y`, `z` | `float` | `0, 0, 0` | Point on the rotation axis |
| `ax`, `ay`, `az` | `float` | `0, 0, 1` | Axis direction vector |
| `dim` | `int` | `2` | Default dimension for bare integer tags |
| `num_elements` | `list[int]` or `None` | `None` | Structured layers around the sweep |
| `heights` | `list[float]` or `None` | `None` | Relative arc lengths per layer |
| `recombine` | `bool` | `False` | Hex/quad elements |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `list[DimTag]` — all generated `(dim, tag)` pairs.

The `num_elements`, `heights`, and `recombine` parameters work the
same way as in `extrude`, but they control the circumferential
discretization rather than a linear one.

A full revolution uses `angle = 2 * math.pi`. A half revolution
(`math.pi`) gives you a half-pipe or a semicircular arch. Partial
sweeps are useful when you only need part of the ring and want to
save elements.

```python
import math

# Revolve a rectangular cross-section 360 degrees around Y
# to create a circular foundation pad
profile = g.model.geometry.add_rectangle(2.0, 0.0, 0.0, 0.5, 0.3)
out = g.model.transforms.revolve(
    profile, 2 * math.pi,
    ay=1.0,                   # revolve around Y axis
    num_elements=[24],        # 24 elements around the circumference
)
foundation_vol = out[1]       # the solid ring
```


## 10. sweep — profiles along arbitrary paths

`sweep` pushes a profile along an arbitrary wire (a curve built from
lines, arcs, splines, or combinations). Unlike `extrude`, the path
does not have to be straight — it can follow any trajectory you define.

```python
result = g.model.transforms.sweep(
    profiles, path, *,
    dim=2,
    trihedron="DiscreteTrihedron",
    label=None,
    sync=True,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `profiles` | `TagsLike` | — | Profile(s) to sweep |
| `path` | `Tag` | — | Tag of the wire to sweep along |
| `dim` | `int` | `2` | Default dimension for profile tags |
| `trihedron` | `str` | `"DiscreteTrihedron"` | Frame transport method |
| `label` | `str` or `None` | `None` | Label for the highest-dim result |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `list[DimTag]` — all generated `(dim, tag)` pairs.

The `path` must be a wire — build it with `g.model.geometry.add_wire`
from individual curve segments. A wire is an ordered sequence of
connected curves; OCC uses it to define the trajectory.

The `trihedron` parameter controls how the profile's orientation is
maintained as it travels along the path. The default
`"DiscreteTrihedron"` works well for most structural paths. Use
`"Frenet"` for smooth curves without inflection points (it follows
the natural curvature frame). Use `"Fixed"` to keep the profile's
orientation constant in world space, which is useful for straight-ish
paths where you want no twisting.

Sweep is the right tool for curved beams, ramps, helical staircases,
and any member whose centerline is not a straight line:

```python
# Sweep an I-beam cross-section along a curved beam path
section = g.model.geometry.add_plane_surface(i_loop, label="I_section")
path    = g.model.geometry.add_wire([arc1, line1, arc2],
                                     label="beam_path")
out     = g.model.transforms.sweep(section, path, label="curved_beam")
```

A practical consideration: sweep can produce degenerate geometry if
the profile is large relative to the curvature radius of the path.
If you get OCC errors during a sweep, check whether the profile
extends past the center of curvature at any point along the wire.
Shrinking the profile or increasing the curve radius usually fixes it.


## 11. thru_sections — variable cross-sections

`thru_sections` (also called "loft") creates a smooth solid that
interpolates between an ordered set of wire profiles. This is the
operation when the cross-section *changes* along the length — tapered
columns, flared pylons, transition pieces between different flange
sizes.

```python
result = g.model.transforms.thru_sections(
    wires, *,
    make_solid=True,
    make_ruled=False,
    max_degree=-1,
    continuity="",
    parametrization="",
    smoothing=False,
    label=None,
    sync=True,
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `wires` | `list[Tag]` | — | Ordered wire tags (at least 2) |
| `make_solid` | `bool` | `True` | Cap the ends to produce a solid |
| `make_ruled` | `bool` | `False` | Force ruled (linearly interpolated) faces |
| `max_degree` | `int` | `-1` | Maximum surface degree (-1 = OCC default) |
| `continuity` | `str` | `""` | `"C0"`, `"G1"`, `"C1"`, `"G2"`, `"C2"`, `"C3"`, or `"CN"` |
| `parametrization` | `str` | `""` | `"ChordLength"`, `"Centripetal"`, or `"IsoParametric"` |
| `smoothing` | `bool` | `False` | Apply a smoothing pass |
| `label` | `str` or `None` | `None` | Label for the highest-dim result |
| `sync` | `bool` | `True` | Synchronize OCC after the operation |

**Returns:** `list[DimTag]` — all generated `(dim, tag)` pairs.

Each wire defines one cross-section at its position in space. OCC
builds a surface (or solid) that passes through all of them in order.
The wires should be topologically similar — same number of edges in
the same order — for reliable results.

`make_solid=True` (the default) caps the two ends and returns a
volume. Set it to `False` if you only need the skinned surface shell,
for instance when modeling a thin-walled structure that you will
assign shell elements to.

`make_ruled=True` forces the lateral faces to be ruled surfaces
(straight-line interpolation between adjacent sections). This
produces flat panels between sections, which can be desirable for
fabricated steel structures where plates are cut flat and welded at
the section boundaries.

```python
# Tapered column: 500x500 mm base, 300x300 mm top, 4 m tall
w_base = g.model.geometry.add_wire([lb1, lb2, lb3, lb4])
w_top  = g.model.geometry.add_wire([lt1, lt2, lt3, lt4])

out = g.model.transforms.thru_sections(
    [w_base, w_top],
    make_solid=True,
    label="tapered_column",
)
```

You can add intermediate sections to control the taper profile:

```python
# Column with a mid-height bulge (400x400 at z=2.0)
w_base = g.model.geometry.add_wire(base_curves)
w_mid  = g.model.geometry.add_wire(mid_curves)
w_top  = g.model.geometry.add_wire(top_curves)

out = g.model.transforms.thru_sections(
    [w_base, w_mid, w_top],
    make_solid=True,
    label="bulged_column",
)
```

The minimum number of wires is two. Passing fewer raises a
`ValueError`.


## 12. Practical examples

The following examples combine multiple transforms to solve common
structural modeling tasks. They illustrate the typical workflow: create
geometry at the origin, copy and position it, then use sweeps to build
3-D volumes.

### 12.1 Column grid via copy + translate

A 3-bay by 2-bay frame with columns at 8 m and 6 m spacing:

```python
import math
from apeGmsh import apeGmsh

g = apeGmsh(model_name="column_grid")
g.begin()

# One column at the origin
col = g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 4.0, label="col")

nx, ny = 4, 3          # 4 columns in X, 3 in Y
sx, sy = 8.0, 6.0      # spacings

columns = [col]
for i in range(nx):
    for j in range(ny):
        if i == 0 and j == 0:
            continue
        [c] = g.model.transforms.copy(col)
        g.model.transforms.translate(c, i * sx, j * sy, 0.0)
        columns.append(c)

# 12 columns are now positioned on the grid
g.end()
```

The pattern is always the same: create once, copy N-1 times, translate
each copy. Collecting the tags into a list lets you assign materials
or labels to all columns in a single loop afterward.

### 12.2 Symmetric bridge via copy + mirror

Model the left half of a simply-supported bridge, then mirror across
the midspan plane:

```python
import math
from apeGmsh import apeGmsh

g = apeGmsh(model_name="bridge")
g.begin()

L = 30.0  # total span

# Build the left half: deck slab and one pier
deck_left = g.model.geometry.add_box(0, 0, 4.0, L/2, 12.0, 0.3,
                                      label="deck_left")
pier = g.model.geometry.add_box(L/4 - 0.5, 4.5, 0, 1.0, 3.0, 4.0,
                                 label="pier_left")

left_half = [deck_left, pier]

# Copy and mirror across the plane x = L/2
right_half = g.model.transforms.copy(left_half)
g.model.transforms.mirror(right_half, 1, 0, 0, -L/2)

# Now the full bridge exists from x=0 to x=30
# Use boolean union or constraints to join the halves at midspan

g.end()
```

The mirror plane `(1, 0, 0, -L/2)` corresponds to the equation
`x - L/2 = 0`, i.e., the vertical plane at `x = 15 m`. Everything to
the left stays in place; the copies are reflected to the right.

### 12.3 Extruding a floor plan into a slab

Start with a 2-D floor outline (perhaps imported from a DXF or built
from geometry primitives), then extrude it vertically to create a
concrete slab:

```python
from apeGmsh import apeGmsh

g = apeGmsh(model_name="slab")
g.begin()

# Floor outline as a rectangle (could be any planar surface)
floor = g.model.geometry.add_rectangle(0, 0, 3.5, 20.0, 10.0,
                                        label="floor_outline")

# Extrude 250 mm upward with 4 structured layers
slab_out = g.model.transforms.extrude(floor, 0, 0, 0.25,
                                       num_elements=[4],
                                       recombine=True)

top_face = slab_out[0]    # for applying live loads
slab_vol = slab_out[1]    # for assigning concrete material

g.end()
```

The `recombine=True` flag requests hexahedral elements, which are
preferred for thin slabs because they avoid the volumetric locking
that plagues linear tetrahedra in bending. Combined with
`num_elements=[4]`, you get four layers of hex elements through the
slab thickness — enough to capture the bending stress gradient.

### 12.4 Revolving a cross-section into a circular foundation

Create a rectangular cross-section in the XZ plane, then revolve it
360 degrees around the Y axis to produce a ring foundation:

```python
import math
from apeGmsh import apeGmsh

g = apeGmsh(model_name="ring_foundation")
g.begin()

# Cross-section: a rectangle in the XZ plane
# at radial distance 3.0 m from the Y axis
profile = g.model.geometry.add_rectangle(
    3.0, 0, -0.5,     # corner at (x=3, y=0, z=-0.5)
    0.6, 0.5,          # width=0.6 m radial, depth=0.5 m
    label="ring_section",
)

# Full revolution around Y axis with 32 circumferential elements
out = g.model.transforms.revolve(
    profile, 2 * math.pi,
    ay=1.0,
    num_elements=[32],
    recombine=True,
)

ring_vol = out[1]

g.end()
```

The cross-section is placed at `x = 3.0`, which becomes the inner
radius of the ring. The rectangle's width of 0.6 m means the ring
spans from radius 3.0 to 3.6 m, with a depth of 0.5 m below grade.
The 32 circumferential elements give roughly 11-degree increments,
which is fine for a foundation ring.


## 13. Transform chaining

The rigid transforms return `self`, so you can chain them in a single
expression. This is a stylistic convenience, not a performance
optimization — each call still synchronizes OCC by default. But it
reads well for compound positioning:

```python
# Copy a column, move it to grid (8, 6), then rotate it 45 degrees
[c] = g.model.transforms.copy(col)
g.model.transforms \
    .translate(c, 8.0, 6.0, 0.0) \
    .rotate(c, math.pi / 4)
```

If you are chaining many transforms in a loop, consider passing
`sync=False` to each call and synchronizing once at the end:

```python
for i in range(100):
    [c] = g.model.transforms.copy(col, sync=False)
    g.model.transforms.translate(c, i * 2.0, 0, 0, sync=False)

gmsh.model.occ.synchronize()   # one sync for all 100 copies
```

This can noticeably speed up models with hundreds of repeated parts.
The OCC synchronization is the expensive step — it rebuilds the
internal topology graph — and batching it avoids doing that work
100 times.


## 14. Common pitfalls

**Forgetting to copy before transforming.** `translate`, `rotate`,
`scale`, and `mirror` modify entities in place. If you translate a
column without copying it first, the original column moves and you
have lost the template. Always `copy` before positioning when you need
the original to remain.

**Degrees vs radians.** `rotate` and `revolve` expect radians.
Passing `90` instead of `math.pi / 2` gives you a roughly 25.8-full-
revolution rotation, which is almost certainly wrong. Use `math.radians`
as a guard.

**Extrude default dim.** `extrude` and `revolve` default to `dim=2`
(surfaces), while the rigid transforms default to `dim=3` (volumes).
If you pass a bare integer tag that is actually a volume to `extrude`,
it will be interpreted as a surface tag and Gmsh will either error or
extrude the wrong entity. Be explicit with `(dim, tag)` tuples when
mixing dimensions.

**Structured layer mismatch.** When using `num_elements` and
`heights` together in `extrude` or `revolve`, the two lists must have
the same length. `heights` values are cumulative fractions and must end
at `1.0`. Violating either condition produces confusing OCC errors.

**Sweep curvature.** In `sweep`, if your profile is wider than the
local radius of curvature of the path, OCC produces self-intersecting
geometry and raises an error. Reduce the profile size or increase the
curve radius.

**thru_sections wire order.** The wires passed to `thru_sections`
must be ordered along the loft direction. Swapping two wires creates a
self-intersecting surface. If you get unexpected geometry, check that
the wire positions progress monotonically along the intended axis.


## 15. Transforms and the FEM pipeline

Transforms happen at the geometry stage, before meshing. They modify
(or create) OCC entities, and those entities are what the mesher sees.
This means:

- Labels and physical groups assigned before a transform survive,
  because the entity tag does not change. You can label a column, copy
  it, translate the copy, and both the original and the copy carry their
  respective labels into the mesh.

- Mesh sizes set on entities before a transform also survive. If you
  set a characteristic length on a surface and then translate it, the
  mesh density travels with the surface.

- Generative operations (`extrude`, `revolve`, `sweep`,
  `thru_sections`) create new entities that have no labels yet. The
  `label` parameter on `sweep` and `thru_sections` is a convenience to
  assign a label to the highest-dimension result immediately. For
  `extrude` and `revolve`, you index into the returned list and label
  the entities yourself.

- After all transforms are done and the mesh is generated, the FEM
  broker (`get_fem_data()`) extracts nodes, elements, and connectivity
  from the final mesh. The broker does not know or care how the
  geometry was built — it sees only the mesh.


## See also

- `guide_parts_assembly.md` — part-based assembly workflows that use
  transforms for positioning
- `guide_selection.md` — building physical groups and mesh selections
  on the transformed geometry
- `guide_loads.md` — applying loads to entities after transforms
- `guide_constraints.md` — coupling parts at shared boundaries after
  positioning
