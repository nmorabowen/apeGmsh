# apeGmsh masses

A guide to defining lumped masses in an apeGmsh session — point masses,
distributed line/surface/volume masses, and the tributary vs consistent
reduction strategies. This document covers the apeGmsh abstraction; see
`guide_fem_broker.md` for how masses land in the broker.

Grounded in the current source:

- `src/apeGmsh/core/MassesComposite.py` — the user-facing composite
- `src/apeGmsh/solvers/Masses.py` — `MassDef`, `MassRecord`, and
  `MassResolver`
- `src/apeGmsh/mesh/_record_set.py` — `MassSet` that lands in the broker

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
# ... geometry, parts, mesh ...
```


## 1. The two-stage pipeline: define, then resolve

Masses follow the same define-then-resolve pattern as loads and
constraints. You define mass distributions before meshing, and they
resolve to per-node mass records after the mesh exists.

```python
# Stage 1 — define (pre-mesh)
g.masses.volume("concrete_block", density=2400)

# Stage 2 — resolve (happens inside get_fem_data)
fem = g.mesh.queries.get_fem_data(dim=3)

# Resolved records
fem.nodes.masses              # MassSet
fem.nodes.masses.total_mass() # sanity check
```

Multiple mass definitions that target overlapping nodes are
**accumulated** — the resolver sums contributions per node. This means
you can define mass for each part independently without worrying about
double-counting at shared boundaries.


## 2. Mass types

### Point mass

A concentrated mass at one or more nodes:

```python
g.masses.point("tip_node", mass=500.0)

# With rotational inertia
g.masses.point("flywheel", mass=200.0, rotational=(10.0, 10.0, 50.0))
```

`rotational=(Ixx, Iyy, Izz)` adds rotational inertia terms. These
only matter for 6-DOF models (ndf=6). For solids (ndf=3), rotational
terms are ignored during resolution.


### Line mass

Distributed mass per unit length along a curve:

```python
g.masses.line("beams", linear_density=50.0)  # 50 kg/m
```

Use this for beam/truss elements where the cross-section mass is
known as a linear density (mass per meter of member length).


### Surface mass

Distributed mass per unit area on a surface:

```python
g.masses.surface("floor_slab", areal_density=300.0)  # 300 kg/m^2
```

Use this for shell elements or non-structural mass on surfaces (e.g.,
floor finishes, cladding weight per m^2).


### Volume mass

Distributed mass per unit volume:

```python
g.masses.volume("concrete_body", density=2400)  # 2400 kg/m^3
```

This is the most common mass definition for solid models. The resolver
computes `mass_per_node = density * V_element / n_nodes` for each
element and accumulates across elements sharing a node.

**Important:** If you also define gravity loads via `g.loads.gravity()`
with a density, make sure you are not double-counting. Either:
- Use `g.masses.volume()` for mass + `g.loads.gravity()` with
  `density=None` (reads from mass), OR
- Set the material density in the solver and let it handle both mass
  and gravity


## 3. Tributary vs consistent reduction

Every distributed mass definition accepts a `reduction` parameter:

```python
g.masses.volume("body", density=2400, reduction="lumped")      # default
g.masses.volume("body", density=2400, reduction="consistent")
```

### Tributary (lumped) — the default

The total mass of each element is divided equally among its nodes.
For a tetrahedral element with volume V and density rho:

```
mass_per_node = rho * V / 4
```

This produces a **diagonal mass matrix** — the gold standard for
explicit dynamics (central difference, explicit Newmark) because the
mass matrix is trivially invertible. It is also correct for static
analysis where mass only matters for self-weight.

Advantages:
- Diagonal mass matrix (fast for explicit solvers)
- Easy to verify by hand (total mass = sum of all nodal masses)
- Exact for uniform density on regular meshes

### Consistent

The mass is distributed using the element shape functions:

```
M_ij = integral( rho * N_i * N_j dV )
```

This produces a **full (non-diagonal) mass matrix** that captures
the kinetic energy of the shape functions exactly. For linear
elements, the consistent mass matrix has the same eigenvalues as the
lumped matrix to leading order, so the difference is small.

When to use consistent:
- Quadratic or higher-order elements where the shape functions are
  not constant
- When matching a textbook/benchmark that specifies consistent mass
- When you need accurate higher-mode eigenfrequencies (consistent
  mass gives better upper-mode accuracy)

When to use lumped:
- Explicit dynamics (mandatory — consistent mass destroys the
  diagonal structure)
- Most practical engineering (indistinguishable results for the
  modes that matter)
- When the mass matrix needs to be diagonal for your solver

**Rule of thumb:** Use `"lumped"` for everything unless you have a
specific reason not to. It is simpler, faster, and produces the same
engineering answer for the first N modes that matter in design.


## 4. How masses land in the broker

After `get_fem_data()`, masses live under `fem.nodes.masses`:

```python
fem.nodes.masses                    # MassSet
len(fem.nodes.masses)               # number of nodes with mass
fem.nodes.masses.total_mass()       # sum of translational mass (mx)
```

Each record is a `MassRecord(node_id, mass)` where `mass` is a
length-6 tuple: `(mx, my, mz, Ixx, Iyy, Izz)`. The solver slices
to its DOF count when emitting commands.


## 5. Consuming masses in a solver

```python
for m in fem.nodes.masses:
    ops.mass(m.node_id, *m.mass[:3])  # 3-DOF model
    # or
    ops.mass(m.node_id, *m.mass)      # 6-DOF model

# Lookup by node
m = fem.nodes.masses.by_node(42)
if m is not None:
    print(f"Node 42: mx={m.mass[0]:.2f}")
```


## 6. Sanity checks

Three quick checks that catch most mass errors:

```python
fem = g.mesh.queries.get_fem_data(dim=3)

# (a) Total mass — should match hand calculation
print("Total mass:", fem.nodes.masses.total_mass())
# Expected: density * total_volume

# (b) Summary table
print(fem.nodes.masses.summary())
# DataFrame: node_id, mx, my, mz, Ixx, Iyy, Izz

# (c) Introspection
print(fem.inspect.mass_summary())
# Nodal masses (1500 nodes):
#   Total: 24000.0  (source: volume mass on pg 'Body', density=2400)
```

The most common bug is an order-of-magnitude error in density (e.g.,
using g/cm^3 instead of kg/m^3). The total mass check catches this
immediately.


## 7. Complete example

```python
from apeGmsh import apeGmsh

g = apeGmsh("building")
g.begin()

# Geometry...
g.model.geometry.add_box(0, 0, 0, 10, 10, 3, label="slab")
g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3, label="column")
g.physical.add_volume([...], name="Slab")
g.physical.add_volume([...], name="Column")

# Mass definitions
g.masses.volume("Slab", density=2400)      # concrete slab
g.masses.volume("Column", density=2400)    # concrete columns
g.masses.surface("Slab", areal_density=200)  # floor finishes (non-structural)

# Mesh and extract
g.mesh.generation.generate(dim=3)
fem = g.mesh.queries.get_fem_data(dim=3)

# Verify
print(f"Total mass: {fem.nodes.masses.total_mass():.0f} kg")
print(f"Nodes with mass: {len(fem.nodes.masses)}")
print(fem.inspect.mass_summary())

# Emit to solver
for m in fem.nodes.masses:
    ops.mass(m.node_id, *m.mass)

g.end()
```


## See also

- `guide_fem_broker.md` — how the broker organizes masses
- `guide_loads.md` — the analogous pipeline for loads (including gravity)
- `guide_constraints.md` — the constraint system
