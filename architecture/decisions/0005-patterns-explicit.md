# ADR 0005 — Patterns are explicit context managers

**Status:** Accepted

## Context

Loads, prescribed `sp` constraints, and ground-motion declarations
are pattern-scoped at the OpenSees C++ level (see
[patterns-and-loads.md](../patterns-and-loads.md) for the source
evidence in `Domain.h` and `SP_Constraint.h`).

openseespy uses an **implicit current pattern**: `ops.pattern(...)`
sets a global state; subsequent `ops.load(...)` calls attach to it.
This is fragile — the binding is invisible at the call site, errors
are silent, and reading a script you cannot tell where one pattern
ends and the next begins.

## Decision

Patterns are **explicit context managers**. `load`, `eleLoad`, and
prescribed (non-zero) `sp` are only callable inside a `with` block:

```python
with ops.pattern.Plain(series=gm) as p:
    p.load(pg="RoofFloor", forces=(100e3, 0, 0))
    p.eleLoad.beamUniform(pg="Beams", wy=-2400)
    p.sp(pg="Bearing", dof=1, value=0.005)
```

Calling `ops.load(...)` outside any pattern raises `RuntimeError` with
a clear message pointing at the pattern requirement.

`fix` and `mass` are model-level — flat methods on `apeSees`, no
context manager needed.

## Alternatives considered

1. **Mirror openseespy's implicit current pattern.** Rejected —
   silent failure modes, hard-to-read scripts, no way to validate
   "did this load attach to the right pattern?" at parse time.
2. **Pattern as a free function returning a registration object.**
   Rejected — `with` is the natural Python idiom for scoped
   resources, and Tcl already uses block scoping for patterns. The
   visual match between the wrapper and Tcl is a feature.
3. **Allow both styles (`with` block AND free `ops.load(...)`).**
   Rejected — two ways to do it; either fails silently when
   misused.

## Consequences

**Positive:**

- Pattern boundaries are textually visible.
- Errors at the call site, not at build time.
- Maps cleanly onto Tcl `pattern Plain N tsTag { ... }` block
  scoping — TclEmitter writes the braces; PyEmitter writes the
  explicit `ops.timeSeries(...) ; ops.pattern(...)` sequence.
- The `Pattern` instance aggregates its loads for inspection
  after the block (P1 relaxation).
- Composes with `Node` aggregation — `roof.load(forces=...)`
  inside a pattern delegates to the active pattern.

**Negative:**

- Slightly more typing than `ops.load(...)`. Acceptable — the
  context manager is one line and applies to many loads.
- Users porting openseespy scripts must restructure calls to
  fit inside `with` blocks. Documented; mechanical.

## Reference

- [patterns-and-loads.md](../patterns-and-loads.md)
- [charter.md P7, P14](../charter.md)
- `OpenSees/SRC/domain/domain/Domain.h:104-117`
- `OpenSees/SRC/domain/constraints/SP_Constraint.h:60,82`
