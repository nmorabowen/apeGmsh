# Patterns, loads, and constraints — the OpenSees-driven decision

This document explains why `apeSees` makes load patterns **explicit
context managers** and why some commands are model-level while others
are pattern-scoped. The decision is forced by the OpenSees domain
layer — we are surfacing it, not inventing it.

## The OpenSees evidence

From `OpenSees/SRC/domain/domain/Domain.h`:

```cpp
virtual  bool addSP_Constraint(SP_Constraint *);                          // model-level
virtual  bool addSP_Constraint(SP_Constraint *, int loadPatternTag);      // pattern-scoped
virtual  bool addNodalLoad   (NodalLoad *,    int loadPatternTag);        // always pattern-scoped
```

`Domain` exposes **two** overloads of `addSP_Constraint`: one without
a pattern tag (model-level), one with. `addNodalLoad` has **no**
overload without a pattern tag — nodal loads are unconditionally
pattern-scoped.

From `SRC/domain/constraints/SP_Constraint.h`:

```cpp
virtual bool isHomogeneous(void) const;       // line 60
...
int  loadPatternTag;                          // line 82
```

The SP carries `loadPatternTag` as a member. `isHomogeneous()`
distinguishes the two flavors at runtime: `fix` produces homogeneous
SPs (no pattern); `sp` inside a pattern produces non-homogeneous
SPs.

`SRC/domain/load/NodalLoad.cpp` and `ElementalLoad.cpp` are owned by
the LoadPattern that holds them — they cannot exist without one.

## What this means in practice

| Command | Scope at the OpenSees level | Why |
|---|---|---|
| `fix` | model-level | homogeneous SP, no pattern |
| `sp` (homogeneous, value=0) | model-level (or pattern, depending on call site) | rare; we route to `fix` |
| `sp` (prescribed, non-zero) | **pattern-scoped** | the SP carries a `loadPatternTag` |
| `load` (NodalLoad) | **pattern-scoped** | pattern owns the load object |
| `eleLoad` (ElementalLoad) | **pattern-scoped** | same |
| `mass` | model-level | mass attaches to the Node |
| `pattern UniformExcitation` | the pattern itself | ground motion is a pattern |

## The API consequence

```python
# ✓ model-level operations are flat
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
ops.mass(pg="Floors", values=(50, 50, 50, 0, 0, 0))

# ✓ pattern-level operations REQUIRE a pattern context
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="RoofFloor", forces=(100e3, 0, 0))
    p.eleLoad.beamUniform(pg="Beams", wy=-2400)
    p.sp(pg="Bearing", dof=1, value=0.005)         # prescribed disp

# ✗ pattern-level operation outside a pattern raises
ops.load(pg="RoofFloor", forces=(100e3, 0, 0))
# RuntimeError: load() must be called inside a `with ops.pattern.X(...) as p:`
#               block. Loads are pattern-scoped at the OpenSees level
#               (see Domain.h:117). Use ops.fix(...) for homogeneous BCs,
#               or open a pattern for non-zero loads.
```

## Why explicit and not a "current pattern" pointer

openseespy itself uses an **implicit current pattern** — calling
`ops.pattern(...)` sets a global that subsequent `ops.load(...)` calls
attach to. This works but produces opaque bugs:

- A `load` after `wipe()` silently does nothing (no current pattern).
- Two pattern-defining functions called in sequence interleave their
  loads if the second forgets to issue its own `pattern(...)` first.
- Reading a script, you cannot tell where one pattern ends and the
  next begins without scanning.

A `with` block:

- Makes the pattern boundary visible textually.
- Errors at the call site if a `load` escapes outside.
- Maps cleanly onto Tcl `pattern Plain N Linear { ... }` block scoping.
- Passes the `Pattern` instance as `p`, so `p.load(...)` is a method
  call on a real object — typed, capability-bearing, inspectable.

## How `Pattern` aggregates its contents

`Pattern` follows the same aggregation pattern as `Node` (P1
relaxation): the typed instance carries the loads and prescribed
SPs that were added inside its block.

```python
with ops.pattern.Plain(series=gm) as p:
    p.load(pg="RoofFloor", forces=(100e3, 0, 0))
    p.load(pg="Floor3",    forces=(80e3, 0, 0))

# After the block:
p.tag                          # OpenSees pattern tag
p.series                       # the time series instance
p.loads                        # tuple of (target, forces) records
p.summary()                    # DataFrame
```

`UniformExcitation` for ground-motion patterns:

```python
gm = ops.timeSeries.Path(file="elcentro.txt", dt=0.01, factor=9.81)

with ops.pattern.UniformExcitation(direction=1, series=gm) as p:
    pass    # body is empty — uniformExcitation is the whole pattern
# OR equivalently:
ops.pattern.UniformExcitation(direction=1, series=gm)   # no with-block
# (zero-payload pattern — the syntax is permitted)
```

## What about `MultiSupportPattern`?

```python
gm_x = ops.timeSeries.Path(file="elcentro_x.txt", dt=0.01, factor=9.81)
gm_y = ops.timeSeries.Path(file="elcentro_y.txt", dt=0.01, factor=9.81)

with ops.pattern.MultiSupport() as p:
    p.groundMotion(tag=1, accel_series=gm_x)
    p.groundMotion(tag=2, accel_series=gm_y)
    p.imposedMotion(node=ops.nodes.get("FootingX"), dof=1, gm_tag=1)
    p.imposedMotion(node=ops.nodes.get("FootingY"), dof=2, gm_tag=2)
```

Same shape. `groundMotion` and `imposedMotion` are methods on the
multi-support pattern instance. The block-scoped vocabulary makes the
pattern's contents inspectable and the boundaries visible.

## Summary

- `fix` and `mass` are flat on `apeSees`.
- `load`, `eleLoad`, prescribed `sp`, ground-motion declarations are
  methods on a pattern context manager.
- The `Pattern` instance carries its contents for inspection.
- Errors raised at the call site, not at build time, when a
  pattern-scoped command escapes.

This matches OpenSees's domain layer one-for-one. We surface the
constraint, we do not invent it.
