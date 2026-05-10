# Charter

## Mission

Translate a meshed structural model — represented as a `FEMData`
snapshot plus typed declarations — into one of three OpenSees emit
targets: a Tcl script, a Python (openseespy) script, or a live
`openseespy` domain. Nothing more: the bridge does not own analysis
strategies, recording strategies, or post-processing. It writes the
deck and stops at "model + recorders + analysis settings + analyze
loop are populated."

## Audience

OpenSees-fluent users who want a Python deck they can verify against
the OpenSees manual, with the safety of static type checking, the
automation of physical-group-aware mesh integration, and the
flexibility to switch between Tcl, Python, and live execution from one
model definition.

## Principles

These are the rules we judge new code against. A change that violates
a principle needs a justification on its PR.

### P1 — Single responsibility per primitive (relaxed)

Each OpenSees concept (Material, Section, GeomTransf, TimeSeries,
Element, Pattern, Recorder, analysis components) is a typed Python
class. The class owns its parameters, validation, `repr`, and optional
capabilities (plotting, testing, queries).

**Where OpenSees concepts have a natural aggregate** (a single Node
carries fixes, mass, and per-pattern loads), the typed Python class
**may** aggregate them. The aggregate exposes the equivalent
OpenSees commands as methods (`node.fix(...)`); the bridge
**still** exposes the flat verb (`ops.fix(pg=...)`). Both work; the
aggregate is the convenience layer.

### P2 — Primitives never touch `ops` directly

No `import openseespy.opensees` outside the emitter layer. Primitives
emit through an `Emitter` Protocol. This is what makes the same
primitive code drive Tcl, py, and live execution from one definition.

### P3 — The bridge takes a `FEMData`, not a session

Construction is `apeSees(fem)`. No `gmsh.*` imports, no `g.parent`,
no implicit "active model." Pickle-able. Testable without booting
gmsh.

### P4 — The bridge owns tag allocation

Primitives carry an *optional* tag (for round-trip reproducibility);
if absent, the bridge assigns one at build time via a `TagAllocator`.
Users never type tags.

### P5 — Dependencies are explicit and topologically resolved

Each primitive declares `dependencies() → list[Primitive]` (a section
returns its materials; an element returns its section + transform).
The bridge sorts the dependency graph, deduplicates, and emits in
correct order.

### P6 — Declared state and built state live in different objects

`apeSees` holds user declarations. `apeSees.build()` returns a
`BuiltModel` — read-only, immutable, the only thing the emitter sees.
Composites (`ops.materials`, `ops.elements`, `ops.nodes`) provide
read-only views over the declared state, mirroring the apeGmsh
`*Composite` / `*Set` convention.

### P7 — One way to declare each kind of thing

`fix` and `mass` are model-level. `load`, `eleLoad`, and prescribed
`sp` are pattern-scoped (see [patterns-and-loads.md](patterns-and-loads.md)).
Both shapes are inherent to OpenSees; we surface them, not invent
them.

### P8 — Adding an emit target is one new file

A new `Emitter` subclass implements the same Protocol; no primitive
code changes. That's the test of whether the abstraction is right.

### P9 — `apeGmsh.opensees` does not depend on `apeGmsh.core` or
`apeGmsh.mesh` internals

The bridge depends only on `apeGmsh.mesh.FEMData` (the snapshot type).
Moving the bridge to a separate package later should be a rename, not
a refactor.

### P10 — Session sugar is optional and reversible

If `g.opensees(fem)` exists, it is a one-line factory that returns an
`apeSees(fem)`. It does not own state. It does not provide
functionality the standalone bridge lacks.

### P11 — Standalone instances

Typed primitives constructed outside a bridge (`Steel02(fy=...)`) are
valid and useful for material studies, parametric sweeps, and
notebooks. Their `tag` is `None` until they are registered with a
bridge via `bridge.register(prim)`.

### P12 — Static typing first

Every user-facing signature is fully typed and visible to pyright /
mypy. **No `**kwargs` and no positional `*args` in user-facing code.**
Internal forwarding (typed class → emitter → openseespy) MAY use
varargs, since the boundary is internal and the openseespy vocabulary
requires it.

### P13 — apeGmsh-native return types

Aggregates returned to the user follow apeGmsh conventions:
`*Composite` / `*Set` / `*Group` shapes (see
`apeGmsh.mesh.FEMData.NodeComposite`, `ElementComposite`,
`_group_set.PhysicalGroupSet` for the precedent). Container protocol
(`__iter__`, `__len__`, `__contains__`, `__getitem__`),
`.summary() → DataFrame`, `.get(target=...)` for queries.

### P14 — Patterns are explicit

`load`, `eleLoad`, and prescribed (non-zero) `sp` are only callable
inside a pattern context manager. `fix`, `mass`, and homogeneous SPs
are model-level. This matches the OpenSees domain layer (see
`SRC/domain/domain/Domain.h:104-117`,
`SRC/domain/constraints/SP_Constraint.h:60,82`,
[patterns-and-loads.md](patterns-and-loads.md)).

## Non-goals

- **No model-of-the-solver-loop.** Recorders are part of the deck;
  per-step Python callbacks during live runs are not.
- **No automated convergence rescue.** Recipes can ship retry logic
  on top, but the core stays simple.
- **No multi-solver targets in v1.** The architecture leaves room for
  ANSYS / Code_Aster, but we ship OpenSees only and prove the
  abstraction is right before adding a second target.
- **No backward compatibility with `apeGmsh.solvers`.** Apps migrate.
- **No automatic re-meshing on change.** If the FEM snapshot changes,
  build a new bridge.
