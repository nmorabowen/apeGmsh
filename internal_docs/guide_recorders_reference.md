# Recorders — What You Can Declare

The recorder vocabulary at a glance: **what categories exist, what
components each accepts, what shorthands expand to what, and what
selectors / cadence options apply.**

> **Programmatic discovery.** Everything below is also available at
> runtime through three static methods — call them in a notebook or
> wire them into validation:
>
> ```python
> from apeGmsh.solvers.Recorders import Recorders
>
> Recorders.categories()
> # ('nodes', 'elements', 'line_stations', 'gauss',
> #  'fibers', 'layers', 'modal')
>
> Recorders.components_for("nodes")[:3]
> # ('acceleration_x', 'acceleration_y', 'acceleration_z')
>
> Recorders.shorthands_for("nodes")["displacement"]
> # ('displacement_x', 'displacement_y', 'displacement_z')
> ```

---

## Categories at a glance

| Method | Topology | Slab returned | Strategy coverage |
|---|---|---|---|
| `recorders.nodes(...)` | per node | `NodeSlab` `(T, N)` | All five strategies |
| `recorders.elements(...)` | per element-node | `ElementSlab` `(T, E, npe)` | All five |
| `recorders.line_stations(...)` | per beam IP | `LineStationSlab` `(T, sum_S)` | All five |
| `recorders.gauss(...)` | per continuum GP | `GaussSlab` `(T, sum_GP)` | All five |
| `recorders.fibers(...)` | per fiber | `FiberSlab` `(T, sum_F)` | `capture` + `emit_mpco` only |
| `recorders.layers(...)` | per shell layer | `LayerSlab` `(T, sum_L)` | `capture` + `emit_mpco` only |
| `recorders.modal(n_modes=N)` | per mode | one stage per mode | `capture` + `emit_mpco` only |

> "Strategy coverage" refers to the five execution paths described in
> [Obtaining results](guide_obtaining_results.md). Fibers, layers,
> and modal records aren't expressible as classic OpenSees recorder
> commands cleanly, so the export.tcl/py and live-recorder paths
> can't carry them. Use `spec.capture(...)` or
> `spec.emit_mpco(...)` instead.

---

## Selectors (the same on every method)

Provide **at most one** of these per declaration — they're mutually
exclusive:

| Selector | What it accepts | When you'd use it |
|---|---|---|
| `pg=` | physical-group name(s) | Solver-facing groups (Tier 2) — the most common case |
| `label=` | apeGmsh label name(s) | Geometry-time labels (Tier 1) — survive boolean ops, useful inside Parts |
| `selection=` | mesh-selection set name(s) | Post-mesh sets defined via `g.mesh_selection` |
| `ids=` | raw int IDs | When you've computed IDs yourself; mutex with the named selectors |

Each named selector accepts a single string or an iterable of strings;
multiple names form a union.

## Cadence (the same on every method)

Provide **at most one**:

| Argument | Behaviour |
|---|---|
| `dt=0.01` | Wall-clock cadence — record every 0.01 time-units |
| `n_steps=10` | Step-count cadence — record every 10 analyse-steps |
| (neither) | Every analysis step (default) |

Modal records honour `dt=` / `n_steps=` only at the file-format level
(MPCO writes the modes once per cadence event).

## Disambiguation hint — `element_class_name=`

Every method except `modal(...)` accepts an optional
`element_class_name=` kwarg (verified at `Recorders.py:204, 256, 311,
389, 452`). It is a hint for the `.out` transcoder when the same
component name could come from more than one OpenSees element class
sharing a selector — the transcoder uses the hint to pick the right
element class without falling back to heuristics. Pass it when the
out-file decoder warns about an ambiguous component, otherwise leave
it `None`.

---

## `recorders.nodes(...)`

Per-node values across the analysis.

### Components

| Group | Names |
|---|---|
| Translational kinematics | `displacement_x/y/z`, `velocity_x/y/z`, `acceleration_x/y/z`, `displacement_increment_x/y/z` |
| Rotational kinematics | `rotation_x/y/z`, `angular_velocity_x/y/z`, `angular_acceleration_x/y/z` |
| Applied forces / moments | `force_x/y/z`, `moment_x/y/z` |
| Reactions | `reaction_force_x/y/z`, `reaction_moment_x/y/z` |
| Scalars | `pore_pressure`, `pore_pressure_rate` |

### Shorthands

| Shorthand | Expands to |
|---|---|
| `"displacement"` | `displacement_x`, `_y`, `_z` |
| `"velocity"` | `velocity_x`, `_y`, `_z` |
| `"acceleration"` | `acceleration_x`, `_y`, `_z` |
| `"displacement_increment"` | `displacement_increment_x`, `_y`, `_z` |
| `"rotation"` | `rotation_x`, `_y`, `_z` |
| `"angular_velocity"` | `angular_velocity_x`, `_y`, `_z` |
| `"angular_acceleration"` | `angular_acceleration_x`, `_y`, `_z` |
| `"force"` | `force_x`, `_y`, `_z` |
| `"moment"` | `moment_x`, `_y`, `_z` |
| `"reaction_force"` | `reaction_force_x`, `_y`, `_z` |
| `"reaction_moment"` | `reaction_moment_x`, `_y`, `_z` |
| `"reaction"` | both forces and moments above (six names) |

Shorthand expansions are clipped to the active `ndm` / `ndf` at
`resolve()` time (a 2D model omits `_z`; an `ndf=3` model omits
rotations).

```python
g.opensees.recorders.nodes(
    components=["displacement", "reaction_force"],
    pg="Top",
    dt=0.01,
)
```

---

## `recorders.elements(...)`

Per-element-node force vectors — the resisting force at each node of
each element.

### Components

| Frame | Names |
|---|---|
| Global | `nodal_resisting_force_x/y/z`, `nodal_resisting_moment_x/y/z` |
| Local | `nodal_resisting_force_local_x/y/z`, `nodal_resisting_moment_local_x/y/z` |

Components from different frames cannot mix in one record — split
into separate records (one for global, one for local).

### Shorthands

This category currently has no shorthands; pass canonical names.

```python
g.opensees.recorders.elements(
    components=["nodal_resisting_force_x"],
    pg="Frame",
)
```

---

## `recorders.line_stations(...)`

Section forces at integration points along beam-column elements.

### Components — section forces

In the section's local frame:

`axial_force`, `shear_y`, `shear_z`, `torsion`, `bending_moment_y`,
`bending_moment_z`.

`bending_moment_y/z` (line-station, local frame) are intentionally
distinct from `moment_x/y/z` on `nodes` (applied nodal moment, global
frame) — same physical units, different topology + frame.

### Shorthand

| Shorthand | Expands to |
|---|---|
| `"section_force"` | all six force / moment components above |

Section deformations (`axial_strain`, `curvature_y/z`, etc.) live in
the read-side vocabulary but are not currently accepted by this
declaration path. They emerge through MPCO's `section.deformation`
bucket on the read side.

```python
g.opensees.recorders.line_stations(
    components=["axial_force", "bending_moment_y"],
    label="frame",
)
```

---

## `recorders.gauss(...)`

Stress / strain / derived scalars at integration points of continuum
elements.

### Components — stress (Cauchy)

| Element type | Names |
|---|---|
| 3D continuum | `stress_xx/yy/zz`, `stress_xy/yz/xz` |
| 2D plane | `stress_xx`, `stress_yy`, `stress_xy` (three independent) |

### Components — strain (small-strain)

| Element type | Names |
|---|---|
| 3D continuum | `strain_xx/yy/zz`, `strain_xy/yz/xz` |
| 2D plane | `strain_xx`, `strain_yy`, `strain_xy` |

### Components — derived scalars

`von_mises_stress`, `pressure_hydrostatic`, `principal_stress_1/2/3`,
`equivalent_plastic_strain`.

### Components — material state

`damage` (single-component), `damage_tension`, `damage_compression`
(split-tension/compression damage models).
`equivalent_plastic_strain_tension`,
`equivalent_plastic_strain_compression`.
`state_variable_<n>` for any non-negative integer `<n>` — generic
material-state outputs.

### Shorthands

| Shorthand | Expands to |
|---|---|
| `"stress"` | six tensor components in 3D, three in 2D |
| `"strain"` | analogous |

Stress and strain components cannot mix in one record (they come
from different OpenSees recorder tokens). Split into two records.

```python
g.opensees.recorders.gauss(
    components=["stress", "von_mises_stress"],
    pg="Body",
)
```

---

## `recorders.fibers(...)`

Per-fiber stress / strain in fiber-section beam-columns
(`FiberSection2d` / `FiberSection3d`).

### Components

| Group | Names |
|---|---|
| Fiber | `fiber_stress`, `fiber_strain` (uniaxial along fiber axis) |
| Material state | `damage`, `equivalent_plastic_strain`, `state_variable_<n>`, etc. |

Indexed variants (`fiber_stress_<n>` / `fiber_strain_<n>`) are
handled at read-time for shell layered sections that emit a vector
per layer. They aren't enumerated here; pass them directly.

### Coverage caveat

**Not emittable via the classic recorder path** (`emit_recorders`,
`export.tcl`, `export.py`) — the per-fiber row count would explode
the `.out` file format. Use `spec.capture(...)` or
`spec.emit_mpco(...)` instead.

```python
g.opensees.recorders.fibers(
    components=["fiber_stress", "fiber_strain"],
    pg="RC_Columns",
)
```

---

## `recorders.layers(...)`

Per-layer stress / strain through the thickness of layered shells
(`LayeredShellFiberSection` on `ASDShellQ4` / `ShellMITC4` /
`ShellDKGQ` / etc.).

### Components

| Group | Names |
|---|---|
| Layer (scalar) | `fiber_stress`, `fiber_strain` |
| Layer (vector) | `fiber_stress_<n>`, `fiber_strain_<n>` (component count auto-discovered on first probe) |
| Material state | `damage_tension`, `damage_compression`, `state_variable_<n>`, etc. |

### Coverage caveat

Same as `fibers` — not emittable via classic recorders. Use
`spec.capture(...)` (needs an OpenSees back-reference for per-layer
thickness + material-tag metadata) or `spec.emit_mpco(...)`.

```python
g.opensees.recorders.layers(
    components=["fiber_stress", "fiber_strain"],
    pg="Slab",
)
```

---

## `recorders.modal(n_modes)`

Eigenvalue analysis. Each requested mode lands as its own stage with
`kind="mode"` in the resulting file, exposing `eigenvalue`,
`frequency_hz`, `period_s` as stage attributes.

### Signature

```python
g.opensees.recorders.modal(n_modes=10, dt=None, n_steps=None, name=None)
```

Modal records have **no `components=` and no selectors** — they
capture all node displacement DOFs of every mode shape.

### Coverage caveat

Modal records can only be executed via:

- `spec.capture(...)` — apeGmsh calls `ops.eigen()` itself.
- `spec.emit_mpco(...)` — MPCO records the `modesOfVibration` token
  natively.

The classic recorder path (`emit_recorders` / `export.tcl`) **cannot**
drive eigenvalue analysis and will raise on `__enter__` if your spec
contains modal records.

```python
g.opensees.recorders.modal(n_modes=10)
```

---

## Validation timing

| Check | When |
|---|---|
| Cadence (`dt` xor `n_steps`) | At declaration |
| Component name in vocabulary | At `resolve(fem)` |
| Component is allowed in this category | At `resolve(fem)` |
| Selector name resolves to IDs | At `resolve(fem)` |
| Element class supports the component | At execution (in `capture` / `emit_*`) |
| MPCO build is available | At `emit_mpco.__enter__` |

If `resolve()` errors, the message lists the valid components for
that category — same data this page documents. So when in doubt:
declare, call resolve, read the error.

---

## Registry inspection and reset

The `Recorders` registry supports the standard container protocols
(`Recorders.py:886-895`):

```python
recs = g.opensees.recorders

len(recs)          # number of declared records
list(recs)         # iterate -- yields RecorderRecord objects
recs.clear()       # drop all declarations and reset the auto-id counter
```

`clear()` is useful when re-running a parametric study in the same
session — call it before re-declaring recorders so the auto-id
counter starts back at zero.

---

## See also

- [Obtaining results](guide_obtaining_results.md) — five execution
  strategies.
- Architecture — *Obtaining the database* (`architecture/apeGmsh_results_obtaining.md`) —
  the spec-as-seam pattern. (Lives outside `internal_docs/`; navigate
  via the published site's *Architecture* section.)
- API reference — *OpenSees* (`docs/api/opensees.md`) — `Recorders`
  class with auto-rendered method signatures. (Lives under
  `docs_dir`; navigate via the published site's *API* section.)
