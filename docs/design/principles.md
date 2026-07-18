# Design principles

This page states the commitments behind apeGmsh's design — the rules that
decide what a feature is allowed to look like — so that when you extend the
library you can tell a good change from a workaround. The
[mental model](../concepts/mental-model.md) teaches the same ideas from the
user's side; here they appear as obligations on the code. A change that
violates one of these principles needs an argument for amending the principle,
not a quiet exception in an adapter.

## One workflow, one boundary

apeGmsh is a broker between Gmsh and FEM solvers, built for structural
engineers working in notebooks. Everything it does sits on one chain:

```
Geometry  ──►  Names  ──►  Mesh  ──►  Broker  ──►  Solver
```

The library's shape mirrors that chain, and features that don't fit it are
rejected or kept as sidecars — they don't become core composites. The chain
ends at `FEMData`, and `FEMData` is the boundary: it is the single contract
between the mesh world and the solver world. No solver adapter calls Gmsh, and
nothing downstream of the broker needs a live session (ADR 0001 fixed this for
the OpenSees bridge; it holds for every consumer). When the broker can't
express a concept a solver needs, the broker grows — the concept does not
migrate into the adapter. That is also why solver-facing records live in the
mesh layer, not in the solver packages (ADR 0013).

The broker is solver-agnostic, designed with OpenSees in mind. Every record
shape and every resolver output is defined in solver-neutral vocabulary, but
OpenSees is the compass: a proposed broker shape is validated by asking
*could the OpenSees adapter translate this in a one-liner?* If the adapter
needs gymnastics, the broker is under-specified and the fix belongs upstream.
Solver-agnostic is the rule; OpenSees is the compass.

## Names before nodes

Gmsh identifies entities by integer tags, and tags are brittle — a boolean or
a re-mesh renumbers them. apeGmsh therefore commits to two things. First,
**names survive operations**: a label or physical group assigned once stays
valid through every boolean, import, and re-mesh for the rest of the session.
This is what makes multi-part assemblies usable at all, and any operation that
silently drops a name is a bug, not a limitation. Second, **define before
mesh, resolve after**: constraints, loads, and masses target names until the
mesh exists, and resolvers turn that intent into node and element records only
at snapshot time. Engineering intent is decoupled from mesh realization, so
the same declarations survive a re-mesh unchanged.

Wrapping Gmsh does not mean hiding it. Gmsh is a beautiful piece of software;
apeGmsh gives it a workflow, not a replacement. The composite tree echoes
Gmsh's own module tree, `(dim, tag)` identity and singleton kernel state are
real and stay visible, and reaching for `gmsh.*` directly is a supported
escape hatch — never an accident to be papered over.

## The shape of the code

The public API is a composition tree, not an inheritance hierarchy. The
session is a container; sub-composites take the session as a parent and expose
one focused surface each. Public composites are plain classes assembled by
composition — behavior never arrives through a base class the user has to
excavate, because that defeats the way people actually navigate the library:
typing `g.` and reading the autocomplete tree. Static typing is the reason
that works. Every public return is typed, and **types are the reference
documentation** — `Any`, untyped `**kwargs`, and bare tuples are avoided in
public signatures. Docstrings carry the intent (*why*, *when to use*); types
carry the shape.

Objects come in three flavors, each with a fixed class style. Composites
(`Model`, `Mesh`, the loads and constraints surfaces) do work and hold state:
regular classes, never dataclasses. Definition objects (`PointLoadDef`,
`EqualDofDef` — pre-mesh intent) are mutable dataclasses whose typed fields
double as documentation. Records — the pieces of `FEMData` and other query
outputs — are frozen dataclasses. The rule to remember: if it has methods
that do work, it is a regular class.

Beneath the composites sits a second, stricter layer: resolvers. Constraint
math, tributary mass, load reduction — these are pure functions over numpy
arrays, with no Gmsh imports and no session references, unit-tested against
synthetic meshes. Composites are impure, integration-tested dispatchers. This
split is how a several-hundred-line resolver stays maintainable.

Errors follow the same domain-first stance. Gmsh's native errors are terse
and often misleading, so apeGmsh catches them at the boundary and re-raises
in workflow vocabulary — which label, which part, which phase. A failing
method tells the user what *they* did wrong, never what the C library
complained about.

## Frozen means frozen

Before `get_fem_data()`, everything is mutable: add, edit, and remove loads,
constraints, and physical groups freely. After it, `FEMData` is a snapshot —
immutable, self-contained, no live Gmsh dependency. The broker is forked from
the session, not subscribed to it; re-meshing produces a *new* broker, never a
mutation of the old one. This is what makes results reproducible, snapshots
composable (ADR 0038), and downstream views cacheable.

Reproducibility itself is treated as a correctness property, not a nicety.
Node renumbering is an explicit, named step; tag assignment is deterministic;
there is no "whatever Gmsh hands us today." Two users running the same
notebook get the same `FEMData`, bit for bit.

## Notebooks are the native habitat

The target user is a structural engineer in Jupyter, not a library architect
and not a Gmsh expert. That biases every API decision toward verbosity,
tab-completion, and dot-paths over clever shortcuts — a newcomer should guess
what `g.mesh.queries.get_fem_data(dim=3)` does without opening the docs. It
also imposes concrete obligations. Sessions, composites, and result objects
carry meaningful `__repr__`s (and `_repr_html_` where it helps), so the
library is usable without ever calling `print`. The viewer is core, not an
add-on — 3D FEM is unreviewable without one — but heavyweight optional
dependencies (Qt, plotting backends, openseespy) are imported at call time,
never at module load. `from apeGmsh import apeGmsh` must succeed on a
headless Colab kernel with no Qt and no display; viewer entry points detect
the environment and route to a web backend when Qt is absent. This is the
difference between *works in notebooks* and *works in the notebook
environment half the engineering world uses*.

## What apeGmsh refuses to be

The non-goals are as binding as the goals. apeGmsh is not a mesh generator —
Gmsh is; it adds workflow, not algorithms. It is not a solver — it never
assembles a stiffness matrix. It is not a parametric CAD tool, and it does
not fight Gmsh's singleton kernel with multi-session or multi-threaded
ambitions. And "solver-agnostic" means the broker is neutral, not that it
aspires to be a universal IR for every FEM code ever written.

Two smaller commitments round this out. apeGmsh is units-agnostic: it
documents conventions and validates none of them — a user mixing kN and MPa
owns the consequences. And it does not freeze its API prematurely: semver is
honored, breaks land in major versions, and each one ships with a
[migration guide](../migration.md) rather than a compatibility shim.

## Holding the line

These principles are meant to be cited. In review, "this hides Gmsh" or
"this puts solver knowledge in the adapter" is a complete argument, and the
answer is a design that respects the principle or a proposal to amend it here.
Larger decisions — the ones with alternatives worth recording — live as ADRs
in the repository, referenced by number throughout these pages. If you find
yourself writing "well, this case is special," stop: amendments are welcome;
quiet exceptions are not.

---

*Next: [The broker](broker.md).*
