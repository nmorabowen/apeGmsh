# Reading & Filtering Results

Once you have a `Results` object — from `from_native`, `from_mpco`,
or `from_recorders` — the question is **how to ask it for what you
want**. This guide covers the full selection and filtering API:
named selectors, geometric helpers, time slicing, stage scoping,
and how everything composes.

The mental model in one sentence:

> **Read with the same vocabulary you wrote with**, plus a few
> geometric helpers for ad-hoc queries — and **every filter is
> additive** so they compose naturally.

---

## 1. The composite tree

`Results` mirrors the `FEMData` shape — every topology level has a
composite with a `.get(...)` method:

| Composite | Returns | Topology |
|---|---|---|
| `results.nodes` | `NodeSlab` `(T, N)` | per node |
| `results.elements` | `ElementSlab` `(T, E, npe)` | per element-node force |
| `results.elements.gauss` | `GaussSlab` `(T, sum_GP)` | per continuum integration point |
| `results.elements.line_stations` | `LineStationSlab` `(T, sum_S)` | per beam integration point |
| `results.elements.fibers` | `FiberSlab` `(T, sum_F)` | per fiber |
| `results.elements.layers` | `LayerSlab` `(T, sum_L)` | per shell layer |
| `results.elements.springs` | `SpringSlab` `(T, E)` | per ZeroLength spring direction |

Every composite supports the same filter vocabulary; only the slab
shape and per-row metadata differ.

---

## 2. Selectors — the canonical menu

All filters are passed as keyword arguments. Below is the full menu;
each section that follows shows them in action.

### 2.1  Named selectors (apply on every composite)

| Selector | Accepts | Resolves to | Notes |
|---|---|---|---|
| `pg=` | string or list of strings | physical-group node/element IDs | Tier-2 names, the most common |
| `label=` | string or list of strings | apeGmsh-label node/element IDs | Tier-1 names; survive boolean ops |
| `selection=` | string or list of strings | mesh-selection-set IDs | Built at session time via `g.mesh_selection.*` |
| `ids=` | iterable of int | exact ID list | Mutually exclusive with the named ones above |

Multiple named selectors **union** (`pg=["A","B"]` is `A ∪ B`;
`pg="A", label="L"` is `A ∪ L`). `ids=` is the surgical override and
cannot mix with named selectors.

### 2.2  Geometric helpers (separate methods)

Spatial queries live as **dedicated methods** rather than `.get()`
kwargs, because their semantics are different (intersection, not
union):

| Method | Args | Filter |
|---|---|---|
| `nearest_to(point)` | 3-element coordinate | Single closest entity |
| `in_box(box_min, box_max)` | two 3-coords | AABB containment, half-open `[lo, hi)` |
| `in_sphere(center, radius)` | center, scalar | Closed ball ‖x − c‖ ≤ r |
| `on_plane(point_on_plane, normal, tolerance)` | point, normal, scalar | `|((x − p) · n̂)| ≤ tol` |

For element-level composites, distances and containment use the
**centroid** of each element (mean of its node coordinates).

### 2.3  Element-type selector (element-level composites only)

| Selector | Accepts | Notes |
|---|---|---|
| `element_type=` | string (broker type name, e.g. `"Tet4"`, `"Hex8"`, `"Quad4"`) | Resolves via `fem.elements.types` and `fem.elements.resolve(element_type=...)` |

### 2.4  Time slicing

| Argument | Behaviour |
|---|---|
| `time=None` | Full time axis (default) |
| `time=N` (int) | Single step by index (`-1` = last) |
| `time=[i, j, k]` | Explicit step indices |
| `time=slice(a, b)` | Half-open over time *values* (numpy semantics) |
| `time=t_value` (float) | Reader picks nearest step |

### 2.5  Stage scoping

| Argument | Behaviour |
|---|---|
| `stage=None` | Auto-resolves when there's exactly one stage |
| `stage="name"` or `stage="<id>"` | Pick a specific stage |

For multiple stages, you can also scope the whole `Results` first
via `results.stage("name")` and then call without `stage=` on the
scoped instance. Same effect.

---

## 3. Geometric helpers — recipes

Every geometric helper exists on every composite. The signatures
look slightly different per topology (nodes vs elements), but the
behaviour is consistent.

### 3.1  `nearest_to(point)`

Returns the slab at the single entity closest to `point`. Distance
is 3D Euclidean against the bound FEMData's node coordinates (or
element centroids for element-level composites).

```python
# Globally nearest node to (1, 0, 0)
slab = results.nodes.nearest_to(
    point=(1.0, 0.0, 0.0),
    component="displacement_z",
)
# → NodeSlab with shape (T, 1)

# Globally nearest element (by centroid)
slab = results.elements.gauss.nearest_to(
    point=(0.0, 0.0, 1.5),
    component="stress_xx",
)
# → GaussSlab with rows for the matching element only
```

`nearest_to` returns **exactly one** entity. To find the K-nearest,
combine with `in_sphere` and post-process the slab values.

### 3.2  `in_box(box_min, box_max)`

Axis-aligned bounding box, half-open on the upper side
(`box_min ≤ xyz < box_max` per axis). The half-open semantic ensures
adjacent boxes don't double-count nodes on a shared face.

```python
# All nodes inside a story-level box
slab = results.nodes.in_box(
    box_min=(-np.inf, -np.inf, 3.0),
    box_max=(np.inf, np.inf, 6.0),
    component="displacement_z",
)

# All Gauss points whose element centroid is in a clipping box
slab = results.elements.gauss.in_box(
    box_min=(-1.0, -1.0, 0.0),
    box_max=(1.0, 1.0, 5.0),
    component="stress_xx",
)
```

Use `np.inf` / `-np.inf` to relax an axis (useful for story-level
or column-line cuts).

### 3.3  `in_sphere(center, radius)`

Closed ball — entity is included if its distance to `center` is
**less than or equal to** `radius`. Useful for blast-radius queries
and proximity searches.

```python
slab = results.nodes.in_sphere(
    center=(2.0, 2.0, 0.0),
    radius=1.5,
    component="acceleration_x",
)
```

`radius` must be non-negative. `radius=0` returns only entities
exactly at `center` (rarely useful — typically you want a small
non-zero tolerance).

### 3.4  `on_plane(point_on_plane, normal, tolerance)`

Slice-through-the-model query: returns entities within `tolerance`
of the plane defined by a point and a normal. The normal is
normalised internally so any non-zero vector works.

```python
# Mid-span vertical cut on a beam laid along x
slab = results.elements.line_stations.on_plane(
    point_on_plane=(L / 2, 0, 0),
    normal=(1, 0, 0),     # plane perpendicular to x
    tolerance=0.05,
    component="bending_moment_y",
)

# Roof slab at z = z_roof, ± 10 mm
slab = results.nodes.on_plane(
    point_on_plane=(0, 0, z_roof),
    normal=(0, 0, 1),
    tolerance=0.010,
    component="displacement_z",
)
```

Useful pairings:
- `point_on_plane=story_height_z, normal=(0,0,1)` → story-level cut.
- `point_on_plane=column_x, normal=(1,0,0)` → column-line cut.

---

## 4. Additive composition — the killer feature

**Every spatial helper accepts the named selectors and intersects
with them.** This is the design feature that makes the read-side
ergonomic: you don't pre-build sets, you compose at the call site.

```python
# Intersection: in box AND in PG TopFlange
slab = results.elements.gauss.in_box(
    box_min=(0, 0, 5),
    box_max=(10, 10, 6),
    component="von_mises_stress",
    pg="TopFlange",      # ← named selector restricts FIRST
)

# Intersection: in sphere AND of element type Tet4 AND in label "RC_zone"
slab = results.elements.gauss.in_sphere(
    center=(0, 0, 0),
    radius=2.0,
    component="stress_xx",
    label="RC_zone",
    element_type="Tet4",
)

# Nearest-element search restricted to a specific element type
slab = results.elements.line_stations.nearest_to(
    point=(L / 2, 0, 0),
    component="bending_moment_y",
    element_type="Line2",          # only beam-column elements
)

# Plain ID set + spatial narrow
slab = results.nodes.in_box(
    box_min=(-1, -1, 0),
    box_max=(1, 1, 5),
    component="...",
    ids=interesting_node_ids,
)
```

The mental rule: **named selectors define the candidate set; the
geometric helper narrows it.** Empty intersections are valid (return
zero-row slab, no error).

---

## 5. The five-shape slab dataclass

Every `.get()` (and every spatial helper) returns a frozen
dataclass with `values` plus location metadata. Shapes:

| Slab | `values` | Location fields |
|---|---|---|
| `NodeSlab` | `(T, N)` | `node_ids` |
| `ElementSlab` | `(T, E, npe)` | `element_ids` |
| `LineStationSlab` | `(T, sum_S)` | `element_index`, `station_natural_coord` |
| `GaussSlab` | `(T, sum_GP)` | `element_index`, `natural_coords`, optional `local_axes_quaternion` |
| `FiberSlab` | `(T, sum_F)` | `element_index`, `gp_index`, `y`, `z`, `area`, `material_tag` |
| `LayerSlab` | `(T, sum_L)` | `element_index`, `gp_index`, `layer_index`, `sub_gp_index`, `thickness`, `local_axes_quaternion` |
| `SpringSlab` | `(T, E)` | `element_index` |

Every slab also carries `time` (shape `(T,)`) and `component` (str).

For `GaussSlab` specifically, `slab.global_coords(fem)` maps each
row's natural coordinates to `(sum_GP, 3)` world coordinates — handy
for plotting Gauss-point fields without building a custom mapper.

---

## 6. Time slicing recipes

```python
# Full time axis
slab = results.nodes.get(component="displacement_z", pg="Top")
slab.values.shape       # (T, N)

# Last step
last = results.nodes.get(component="displacement_z", pg="Top", time=-1)
last.values.shape       # (1, N)

# Specific step indices
sample = results.nodes.get(
    component="displacement_z", pg="Top",
    time=[0, 50, 99],
)
sample.values.shape     # (3, N)

# Half-open slice over time *values* (numpy semantics)
window = results.nodes.get(
    component="displacement_z", pg="Top",
    time=slice(1.0, 2.0),
)

# Single time value — reader picks nearest step
near = results.nodes.get(
    component="displacement_z", pg="Top",
    time=1.5,
)
```

The `time=` parameter is honoured by **every** filter helper —
spatial included.

---

## 7. Stage scoping recipes

```python
# Single-stage file: stage auto-resolves
disp = results.nodes.get(component="displacement_z")

# Multi-stage file: pick explicitly
gravity = results.stage("gravity")
g_disp = gravity.nodes.get(component="displacement_z", pg="Top")

# Or pass stage= on each call
disp = results.nodes.get(
    component="displacement_z", pg="Top", stage="gravity",
)

# Modal stages — kind="mode" filter
for mode in results.modes:
    print(mode.mode_index, mode.frequency_hz, mode.period_s)
    shape = mode.nodes.get(component="displacement_z")
    # shape.values has shape (1, N) — one "step" per mode

# Stable iteration by mode index
for mode in sorted(results.modes, key=lambda m: m.mode_index):
    ...
```

---

## 8. Discovery — what's in this file?

Three layers of introspection:

```python
# Top-level
print(results)                          # __repr__ → inspect.summary()
print(results.inspect.summary())        # explicit
results.stages                          # list[StageInfo]

# Per topology
results.nodes.available_components()
# ['displacement_x', 'displacement_y', 'displacement_z', 'reaction_force_x', …]

results.elements.gauss.available_components()
# ['stress_xx', 'stress_yy', 'stress_zz', 'strain_xx', …]

results.elements.fibers.available_components(stage="dynamic")
```

Pair this with the declaration-side introspection (PR #50) to round-
trip the vocabulary discoverability:

```python
from apeGmsh.solvers.Recorders import Recorders
Recorders.categories()              # what you can declare
Recorders.components_for("nodes")   # what valid components exist
results.nodes.available_components()# what's actually in this file
```

---

## 9. Worked recipes

### 9.1  Time history at the closest node to a target point

```python
slab = results.nodes.nearest_to(
    point=(1.5, 0, 0),
    component="displacement_z",
)
# slab.node_ids is a 1-element array — the chosen node ID
# slab.values has shape (T, 1)

import matplotlib.pyplot as plt
plt.plot(slab.time, slab.values[:, 0])
plt.xlabel("time [s]"); plt.ylabel("u_z [m]")
```

### 9.2  Story-level horizontal cut

```python
slab = results.nodes.in_box(
    box_min=(-np.inf, -np.inf, 5.95),
    box_max=(np.inf, np.inf, 6.05),
    component="displacement_x",
)
# slab.values: (T, N_in_story)
# slab.node_ids: matching IDs
```

### 9.3  Yield zone — Gauss points in PG with high stress

```python
slab = results.elements.gauss.get(
    component="von_mises_stress",
    pg="ColumnFlange",
)
peak_per_gp = np.abs(slab.values).max(axis=0)   # (sum_GP,)
yielding = peak_per_gp > yield_stress
print(f"{yielding.sum()} / {yielding.size} GPs above yield")

# Map to element IDs
yielding_elements = np.unique(slab.element_index[yielding])

# Re-fetch only those elements' Gauss points if you want a tight slab
focus = results.elements.gauss.get(
    component="von_mises_stress",
    ids=yielding_elements,
)
```

### 9.4  Mid-span moment along a beam line

```python
slab = results.elements.line_stations.on_plane(
    point_on_plane=(L / 2, 0, 0),
    normal=(1, 0, 0),
    tolerance=0.05,
    component="bending_moment_y",
    label="frame.beam_BC",   # restrict to one beam
)
# slab.values: (T, N_stations_near_midspan)
```

### 9.5  Modal contour at z = z_roof

```python
mode = sorted(results.modes, key=lambda m: m.mode_index)[0]
slab = mode.nodes.on_plane(
    point_on_plane=(0, 0, z_roof),
    normal=(0, 0, 1),
    tolerance=0.01,
    component="displacement_z",
)
# slab.values: (1, N_roof) — one "step" because modal
```

### 9.6  Pushover — displacement vs reaction force

A canonical capacity-curve plot. Three flavours showing how the
filters compose.

**Basic — pg= for both top and base:**

```python
# Top node — applied displacement
top_disp = results.nodes.get(
    component="displacement_z", pg="Top",
)
u_z = top_disp.values[:, 0]   # single node → pick column 0

# Base reactions — sum over the whole base PG
base_react = results.nodes.get(
    component="reaction_force_z", pg="Base",
)
V = -base_react.values.sum(axis=1)   # opposite sign of applied force

plt.plot(u_z, V)
plt.xlabel("Top displacement u_z  [m]")
plt.ylabel("Base shear V  [N]")
```

**Geometric — no PGs needed:**

```python
# Top — pick the node nearest a target coordinate
u_z = results.nodes.nearest_to(
    point=(0, 0, story_height),
    component="displacement_z",
).values[:, 0]

# Base — slice through z = 0
V = -results.nodes.on_plane(
    point_on_plane=(0, 0, 0),
    normal=(0, 0, 1),
    tolerance=1e-3,
    component="reaction_force_z",
).values.sum(axis=1)

plt.plot(u_z, V)
```

**Compound — additive PG ∩ spatial slice for a specific column line:**

```python
# Top of column-line (5, 0) at story height
u_z = results.nodes.in_box(
    box_min=(4.95, -0.05, story_height - 0.05),
    box_max=(5.05,  0.05, story_height + 0.05),
    component="displacement_z",
    pg="ColumnTops",       # restrict first, then narrow with the box
).values.mean(axis=1)

# Base reaction at the same column line
V = -results.nodes.in_box(
    box_min=(4.95, -0.05, -0.05),
    box_max=(5.05,  0.05,  0.05),
    component="reaction_force_z",
    pg="Base",
).values.sum(axis=1)
```

**Multi-stage — gravity baseline subtracted from pushover:**

```python
gravity = results.stage("gravity")
pushover = results.stage("pushover")

# End-of-gravity reference state
u_baseline = gravity.nodes.get(
    component="displacement_z", pg="Top", time=-1,
).values[:, 0]

# Pushover deltas
top = pushover.nodes.get(component="displacement_z", pg="Top")
delta_u = top.values[:, 0] - u_baseline

V = -pushover.nodes.get(
    component="reaction_force_z", pg="Base",
).values.sum(axis=1)

plt.plot(delta_u, V)
```

### 9.7  Compare two stages at the same nodes

```python
gravity = results.stage("gravity")
dynamic = results.stage("dynamic")

g = gravity.nodes.get(component="displacement_z", pg="Top")
d = dynamic.nodes.get(component="displacement_z", pg="Top")

# g.node_ids and d.node_ids should be identical (same PG, same FEM)
relative = d.values - g.values[-1:]   # subtract last gravity step
```

---

## 10. Pitfalls

### 10.1  "I didn't bind a fem= so my pg= raises"

```python
results = Results.from_native("run.h5")        # no fem=
results.nodes.get(pg="Top", ...)
# RuntimeError: Cannot resolve pg= / label= / selection= without a bound FEMData.
```

Either pass `fem=` to `from_native` or call `results.bind(fem)`. The
embedded `FEMData` snapshot from native files works for IDs but not
always for labels — the binding contract is hash-checked.

### 10.2  "My in_box returns nothing"

Three common causes:

- **Wrong unit.** `box_min=(0,0,0), box_max=(1,1,1)` on a model in
  millimetres returns nothing if all coords are >> 1.
- **Half-open semantics.** Tight boxes that exactly match a node's
  coordinate on the upper side **exclude** that node. Pad by a
  small ε.
- **Empty intersection** with a named selector you also passed.

### 10.3  "nearest_to picked the wrong node"

Distances are computed in world coordinates from
`fem.nodes.coords`. If your model has scale issues or the FEM
snapshot doesn't match what you think, `fem.nodes.coords` is the
ground truth. Use `results.fem.nodes.coords` to spot-check.

### 10.4  "I want elements with a node in box, not centroid in box"

The geometric helpers on element composites filter by **centroid**.
For "any-node-in-box" semantics, do it manually:

```python
ids_with_node_in_box = []
for type_info in fem.elements.types:
    eids, conn = fem.elements.resolve(element_type=type_info.name)
    for i, eid in enumerate(eids):
        node_xyz = fem.nodes.coords[
            np.searchsorted(np.sort(fem.nodes.ids),
                            conn[i])
        ]
        if np.any(np.all((node_xyz >= box_min) & (node_xyz < box_max), axis=1)):
            ids_with_node_in_box.append(int(eid))

slab = results.elements.gauss.get(
    component="...", ids=ids_with_node_in_box,
)
```

We may add this as a built-in (`anchor="centroid"|"any_node"|"all_nodes"`)
in a future revision — open an issue if you need it.

---

## 11. What's queued (next iteration)

Two more filter families on the way:

### 11.1  Value-range filter (`where`)

```python
# Coming soon
results.nodes.where(
    component="displacement_z",
    greater_than=0.05,
    aggregate="any",       # "any" | "all" | "peak" | "trough"
    pg="Top",              # additive — same vocabulary as everywhere
    time=None,             # None | int | "peak"
)
```

Filters by the result *values* themselves: "show me entities whose
displacement exceeds X at any time". Composes additively with the
spatial helpers.

### 11.2  Slab aggregation methods

```python
# Coming soon — operations on the slab itself
slab = results.nodes.get(component="displacement_z", pg="Top")
slab.peak()           # max-abs over time per entity → (values, time_indices)
slab.envelope()       # (min_per_entity, max_per_entity) over time
slab.to_dataframe()   # pandas DataFrame, one row per (time, entity)
slab.relative_to(ref_id)  # subtract a reference node's history (drift)
```

These live on the **slab** dataclass, not the composite — they're
post-fetch operations on data you already have.

---

## 12. Cross-references

- [Results — Obtaining the database](guide_obtaining_results.md) —
  the five execution strategies.
- [Recorder reference](guide_recorders_reference.md) — what you can
  declare on the write side (mirror of this page).
- [Results container guide](guide_results.md) — `Results` as a
  data structure (slabs, snapshots, viewers).
- [Architecture — Obtaining the database](../architecture/apeGmsh_results_obtaining.md) —
  the spec-as-seam pattern.
