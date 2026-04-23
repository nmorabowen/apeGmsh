# apeGmsh Principles

*Written by nmb and guppi.*

This document is the manifesto of apeGmsh — the non-negotiable commitments
that shape every file, every composite, and every public API choice. A
pull request that violates one of these needs an explicit amendment to
this document, not a workaround in code.

---

## 1. What apeGmsh is

**apeGmsh is a Python broker between [Gmsh](https://gmsh.info) and FEM
solvers, written for structural engineers who build models in
notebooks.** It collapses geometry authoring, naming, constraints,
loads, masses, and meshing into a single session whose output is a
frozen, solver-ready snapshot.

It is **solver-agnostic, designed with OpenSees in mind.** The broker
owes nothing to any particular solver — every data structure, every
resolver, every name is defined in solver-neutral terms. But OpenSees
is always the compass: when we choose vocabulary, argument shapes, or
resolver outputs, we validate the choice by asking *could the OpenSees
adapter translate this in a one-liner?* If the answer is no, the broker
is wrong. Solver-agnostic is the rule; OpenSees is the compass.

## 2. Who it's for

A structural or mechanical engineer comfortable in Python and Jupyter,
who wants an object-oriented, discoverable API. Not a library architect.
Not a Gmsh expert. This biases every decision toward **verbosity,
tab-completion, typed returns, and dot-paths over clever shortcuts.** A
newcomer reading a notebook should guess what
`g.mesh.queries.get_fem_data(dim=3)` does without opening the docs.

## 3. The canonical workflow

There is one path, and the library's shape mirrors it:

```
    Geometry  ──►  Names  ──►  Mesh  ──►  Broker  ──►  FEM
```

Geometry is built live in the session or composed from `Part` instances
via STEP. Names (physical groups) are assigned in code or by picking in
the viewer. The mesh is generated and renumbered. The broker resolves
pre-mesh definitions against the mesh into a frozen `FEMData`. The
solver adapter consumes the broker and does nothing else.

Features that don't fit this chain are rejected or demoted to sidecars.
They don't become core composites.

---

## 4. Design tenets

**(i) Composition over inheritance. No mixins.**
The session is a container, not a base class. Sub-composites accept the
session as a parent and expose a focused surface. Mixins are banned —
they obscure method origin and defeat static navigation.

**(ii) Static typing is a user aid, not a correctness tool.**
Every public return is typed. Users navigate the library by typing
`g.` and reading the IDE's autocomplete tree, then following types
into sub-composites. **Types are the reference documentation.** `Any`,
untyped `**kwargs`, and raw tuples are avoided in public APIs.

**(iii) Names survive operations.**
Labels and physical groups are preserved through every boolean, every
import, every re-mesh. A name assigned once is valid for the rest of
the session. This is what makes multi-part assemblies usable at all.

**(iv) Define before mesh, resolve after.**
Constraints, loads, and masses target names (label or PG), not nodes,
until the mesh exists. The resolver turns intent into node/element
records post-mesh. This decouples engineering intent from mesh
realization and survives re-meshing.

**(v) The broker is the boundary.**
`FEMData` is the single contract between the mesh world and the solver
world. No solver adapter calls Gmsh. No viewer calls Gmsh after the
broker exists. If the broker can't express a concept, the broker grows
— the concept does not migrate into the adapter.

**(vi) Solver-agnostic in code, OpenSees in mind.**
Broker vocabulary, resolver outputs, and record shapes are
solver-neutral. But the correctness bar is *"the OpenSees adapter is
thin and obvious."* If adding an OpenSees feature forces gymnastics in
the adapter, the broker is under-specified and needs an upstream change.

**(vii) Gmsh is a beautiful piece of software. We never hide it.**
apeGmsh wraps Gmsh to give it a workflow, not to replace it. The
composite tree echoes Gmsh's own module tree (`model`, `mesh`, `occ`)
so users who know Gmsh aren't disoriented. Reaching for `gmsh.*`
directly is a supported escape hatch, documented where relevant. We
don't abstract over singleton state, `(dim, tag)` identity, or BRep
adjacency — those are real, and the library reflects them. See
[[gmsh_basics]] and [[gmsh_interface]].

**(viii) The viewer is core and environment-aware.**
3D FEM is unreviewable without visualization. apeGmsh ships **two**
entry surfaces with distinct environment coverage:

- **Interactive Qt viewers** (`g.model.viewer()`, `g.mesh.viewer()`) —
  full authoring and review, picking, physical-group editing,
  overlays. Qt + PyVista. Works on **Desktop** and **local Jupyter
  with a Qt backend**; requires a display.
- **Inline WebGL preview** (`g.model.preview()`, `g.mesh.preview()`,
  top-level `apeGmsh.preview()`) — review-only, hover tooltips for
  `dim / tag / pg / label`, works anywhere plotly renders. This is the
  **Colab / headless** path. No Qt, no display required. Plotly is an
  optional dependency (pre-installed on Colab; `pip install plotly`
  locally).

Environment coverage today:

| Environment                      | Qt viewers | `preview()` |
|----------------------------------|:-:|:-:|
| Desktop                          | ✓ | ✓ |
| Local Jupyter (Qt available)     | ✓ | ✓ |
| **Colab / remote notebooks**     | ✗ | **✓** |
| Headless CI (no display)         | ✗ | ✓ (`browser=False` only) |

External results scale-up lives in `apeGmshViewer` (Rust/WebGL
subprocess); see `results/Results.py`. PyVista is a hard dependency;
PyQt6 and plotly are soft dependencies (installed per environment).
`gmsh.fltk.run()` is a desktop debugging aid — never part of a
documented workflow.

---

## 5. Style tenets

**(ix) Three object flavors, three class styles.**

- **Composite classes** — `Model`, `Mesh`, `LoadsComposite`, everything
  under `_COMPOSITES`. Behavior-bearing, stateful, hold a session
  reference. **Regular classes, always.** No `@dataclass`.
- **Definition classes** — `PointLoadDef`, `EqualDofDef`, all pre-mesh
  intent objects. Typed data bags the user constructs and hands to the
  resolver. **Dataclasses (mutable, with optional `__post_init__`
  validation).** The typed fields double as documentation.
- **Record / result classes** — `MassRecord`, `ConstraintRecord`, the
  pieces under `FEMData`. Outputs of the broker or of query methods.
  **Frozen dataclasses.** Frozen matches tenet (xi); you get `__eq__`,
  `__repr__`, and `__hash__` for free.

The rule to remember: **if an object has methods that do work, it is a
regular class.** Everything else leans on dataclasses exactly where they
pay off.

**(x) Errors are domain-level, not Gmsh-level.**
Gmsh's native errors are terse and often misleading. apeGmsh catches
them at the boundary and re-raises with domain vocabulary: which label,
which part, which phase of the workflow. Users should never have to
read Gmsh internals to debug their script. A failing method says what
the user did wrong, not what the C library complained about.

**(xi) Pre-mesh is mutable; the broker is frozen.**
Before `get_fem_data()`, you can add, remove, and edit constraints,
loads, masses, and physical groups freely. After, `FEMData` is a
snapshot — immutable, self-contained, no live Gmsh dependency. The
broker doesn't observe the session; it is forked from it. Re-meshing
produces a new broker, not a mutation of the old one. This is what
makes results reproducible and views cacheable.

**(xii) Pure resolvers, impure composites.**
Two layers, cleanly separated:

- **Composites** hold state, call Gmsh, manage the session. They are
  integration-tested.
- **Resolvers** (constraint math, tributary mass, load reduction) are
  pure functions of numpy arrays. No Gmsh imports. No session
  references. Unit-testable with synthetic meshes.

This is how a 700-line resolver stays maintainable: the math is pure,
the composite is a thin dispatcher.

**(xiii) Reproducibility is a correctness property.**
Given the same inputs and the same seed (where randomness matters —
e.g., Hilbert renumbering), the broker produces bit-identical output.
No implicit node ordering. No "whatever Gmsh hands us today."
Renumbering is an explicit step with a named method. Tag assignment is
deterministic. Two users running the same notebook get the same
`FEMData`.

**(xiv) Notebook-ergonomic by construction.**
Every session, every composite, every major result object has a
meaningful `__repr__` and — where it helps — a `_repr_html_`. Running
`g` in a cell shows the model name, state, and composite tree. Running
`fem` shows mesh stats. Running `fem.nodes` shows the first few IDs and
a count. The library is usable without ever calling `print(...)`.

---

## 6. Non-goals

- **Not a mesh generator.** Gmsh is. apeGmsh adds workflow, not
  algorithms.
- **Not a FEM solver.** OpenSees (and others) are. We don't assemble
  stiffness matrices.
- **Not a CAD tool.** Parametric sections exist; full parametric CAD
  does not.
- **Not multi-threaded or multi-session.** Gmsh is a singleton; we
  accept that.
- **Not a universal solver IR.** Solver-agnostic means the broker is
  neutral, not that it tries to be a lingua franca for every FEM code
  ever written.

---

## 7. Explicit commitments

**Units-agnostic.**
apeGmsh does not enforce units. A user mixing kN and MPa is the user's
problem. We document conventions; we don't validate them. Choice of
units is outside the broker's contract.

**No backward compatibility.**
Semver is honored. Between major versions, breaks are acceptable and
documented in `MIGRATION.md`. The library is young — freezing the API
prematurely would cost more than it would save. Users get a clean
migration script; they do not get a frozen v0.

**Docstrings carry intent; types carry shape.**
Every public method has a docstring. Types say *what*; the docstring
says *why* and *when to use*. Inline comments explain non-obvious *why
here* — never *what* (the code shows that).

**Lazy imports, Colab-safe.**
Google Colab is a first-class runtime. The import `from apeGmsh import
apeGmsh` must succeed on a headless Colab kernel with no Qt, no
display, no FLTK. This is not optional polish — it is the difference
between *works in notebooks* and *works in the notebook environment
half the engineering world uses.*

Concrete rules:

- Optional heavyweight dependencies (Qt, matplotlib, openseespy)
  are imported at call time, not at module load.
- Viewer modules detect the environment and route to an HTML backend
  when Qt is absent.
- No `gmsh.fltk.*` call appears in any import-time code path.
- CI includes a headless smoke test that imports every public symbol
  without a display.

---

## 8. How to use this document

This file is the charter. Before adding a feature, check it against
tenets (i)–(xiv). Before refactoring, check it against the commitments
in §7. If you find yourself writing "well, this case is special," stop
and either (a) find a design that respects the tenet, or (b) propose an
amendment to this document. Amendments are welcome. Quiet exceptions
are not.

The tenets are numbered so they can be cited. A PR comment that says
"this violates (vii) — we're hiding Gmsh here" is a complete argument.

---

*Cross-references:*
[[gmsh_basics]] · [[gmsh_interface]] · [[gmsh_geometry_basics]] ·
[[gmsh_meshing_basics]] · [[gmsh_selection]]
