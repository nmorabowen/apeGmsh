---
title: apeGmsh Hinting
aliases: [hinting, apeGmsh-hinting, typing, type-hints, static-checking, linting]
tags: [apeGmsh, architecture, typing, mypy, pyright, ruff, PEP-484, PEP-612, PEP-695]
---

# apeGmsh Hinting

> [!note] Companion document
> This file defines the **static typing and linting contract** for
> apeGmsh. It is load-bearing architecture, not cosmetics: every
> decision here is chosen so that the IDE, the type checker, and the
> reader see the same thing. The library is a *broker* — its job is to
> hand cleanly-shaped data to a solver. The API must therefore be
> self-describing at the call site. Read alongside
> [[apeGmsh_principles]] (especially tenets **(iii)** "names survive
> operations", **(ix)** "three class flavours", **(xii)** "pure
> resolvers, impure composites").

> [!important] The code must talk.
> A caller should know the shape of a method by *reading its
> signature* — not by reading the docstring, not by reading the body,
> not by running it. This document enumerates what that means in
> practice: no silent `**kwargs`, keyword-only parameters everywhere
> it matters, `Literal` types for string enums, and a shared vocabulary
> of type aliases so `DimTag`, `NodeId`, and `PartLabel` are never
> written as bare `tuple[int, int]`, `int`, or `str`.

---

## 1. Why this is architecture, not style

The apeGmsh surface is dense: every composite is a small DSL
(`g.loads.surface.pressure(...)`, `g.constraints.tie(...)`,
`g.parts.part(...)`). Each factory accepts five to ten parameters —
many of which collide in meaning (a `tolerance` in model units vs a
`tolerance` in parametric coords, a `direction` that accepts a tuple
or an axis name, a `target` that accepts five different types). If
the signature is loose, the user has to guess. If the user has to
guess, they read the docstring. If the docstring drifts from the
code, they read the source. Each step is a failure of the library.

Static typing is the cheapest way to keep the signature honest. It
catches drift at edit time, not at runtime. It gives the IDE enough
information to complete parameter names and flag impossible
combinations. And it keeps the broker promise — that downstream of
`from_gmsh(...)` every array has a known shape and dtype — auditable
from one `mypy --strict` run instead of a grep.

The non-negotiables below are the *minimum* signal a well-typed
apeGmsh method must emit. Everything else in this doc is guidance.

---

## 2. Non-negotiables

These are the rules that cannot be relaxed without architectural
review. CI enforces them.

### 2.1 No `**kwargs` in public API

The library has zero legitimate uses of `**kwargs` in the public
surface. Every parameter a caller may pass is enumerated, named, and
typed. If you feel the urge to write `def foo(self, **kwargs)`, the
right tool is one of:

- a `dataclass` (for grouped related fields),
- a `TypedDict` (when the call site must pass a dict that came from
  somewhere else),
- or `typing.overload` (when the valid combinations are finite and
  orthogonal).

> [!warning] `**kwargs` is an admission that the signature is
> under-specified. The IDE loses autocomplete. The type checker loses
> argument validation. The reader loses the shape of the call. It is
> the opposite of the broker promise.

The one narrow exception is **internal helpers that forward to a
typed function**: a `_dispatch_to(fn, *args, **kwargs)` shim that
exists solely to re-call `fn(*args, **kwargs)` is fine, *as long as
the public entry point is fully typed* and the shim's own types use
`ParamSpec` (PEP 612):

```python
from typing import ParamSpec, TypeVar, Callable

P = ParamSpec("P")
R = TypeVar("R")

def _retry(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    ...
```

With `ParamSpec`, `*args` and `**kwargs` propagate the caller's
signature through the shim; without it, they are a hole.

### 2.2 Keyword-only after the first positional

Every public method has **at most one** positional parameter (the
"noun" — usually `target` or the label being operated on). Everything
else is keyword-only via `*`:

```python
# GOOD — the code talks
def surface(
    self,
    target: TargetLike | None = None,
    *,
    pg: str | None = None,
    label: LabelName | None = None,
    tag: list[DimTag] | None = None,
    magnitude: float = 0.0,
    normal: bool = True,
    direction: Vec3 = (0.0, 0.0, -1.0),
    reduction: Reduction = "tributary",
    target_form: TargetForm = "nodal",
    name: str | None = None,
) -> SurfaceLoadDef: ...

# BAD — what does the third positional mean?
def surface(self, target, magnitude, normal=True, ...): ...
```

The rule of thumb: if a reader has to count commas to know which
argument is which, the method is wrong. `g.loads.surface.pressure("slab",
-3e3)` is ambiguous — is `-3e3` magnitude? pressure? normal? The
call site should read `g.loads.surface.pressure("slab", magnitude=-3e3)`.

Two exceptions:

1. **Positional-only math primitives.** `resolver.edge_length(n1, n2)`
   is fine — the two node IDs are symmetric, there's no ergonomic
   reason to name them.
2. **`__init__` of Def/Record dataclasses.** `@dataclass` generates
   positional-then-keyword init; we let that stand because Defs/Records
   are mostly built by factories (keyword-only by the first rule) and
   direct construction is a rare internal path.

### 2.3 Every public parameter has an explicit type

No bare `def foo(self, x, y)`. Public means "listed in a module's
`__all__`", "reachable from `g.*`", or "part of a composite's
factory API". Internal helpers prefixed with `_` still should be
typed, but with latitude for `Any` in tight hot paths.

### 2.4 Every public function has an explicit return type

Including `-> None`. `-> None` is a signal that the function is
called for effect; the reader should not have to grep for `return` to
figure out.

### 2.5 `from __future__ import annotations` at the top of every module

This makes all annotations strings (PEP 563 / 649) by default:

- Forward references work without quoting.
- Heavy imports can move under `if TYPE_CHECKING:`.
- Runtime cost of annotations is zero.

It is already present in `solvers/Loads.py`, `solvers/Constraints.py`,
`core/LoadsComposite.py`, `core/ConstraintsComposite.py`, etc. When
you add a new module, add this line first.

---

## 3. The type vocabulary

The broker promise depends on a small shared vocabulary. These
aliases live in `apeGmsh/_types.py` and **every public signature that
takes one of these concepts uses the alias, not the underlying
primitive.**

```python
# apeGmsh/_types.py
from __future__ import annotations

from typing import Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray

# ── Identity primitives ───────────────────────────────────────────
DimTag:      TypeAlias = tuple[int, int]      # (dim, tag) — Gmsh ground truth
NodeId:      TypeAlias = int                  # mesh node tag
ElementId:   TypeAlias = int                  # mesh element tag
PhysTag:     TypeAlias = int                  # physical-group tag
EntityTag:   TypeAlias = int                  # bare geometry tag (dim implied by context)

# ── Name-space strings ────────────────────────────────────────────
LabelName:    TypeAlias = str     # Tier 1 label (pre-prefix)
PGName:       TypeAlias = str     # Tier 2 user-authored physical group
PartLabel:    TypeAlias = str     # key of g.parts._instances
SelectionName: TypeAlias = str    # key of g.mesh_selection

# ── Geometric primitives ──────────────────────────────────────────
Vec3:   TypeAlias = tuple[float, float, float]
Mat3:   TypeAlias = tuple[Vec3, Vec3, Vec3]
Axis:   TypeAlias = Literal["x", "y", "z"]
DirLike: TypeAlias = Vec3 | Axis

# ── Numpy shapes ──────────────────────────────────────────────────
NodeCoords:   TypeAlias = NDArray[np.float64]  # shape (n_nodes, 3)
Connectivity: TypeAlias = NDArray[np.int64]    # shape (n_elems, n_nodes_per_elem)
NodeTags:     TypeAlias = NDArray[np.int64]    # shape (n_nodes,)
ElemTags:     TypeAlias = NDArray[np.int64]    # shape (n_elems,)

# ── Literal enums (replace magic strings) ─────────────────────────
Reduction:   TypeAlias = Literal["tributary", "consistent"]
TargetForm:  TypeAlias = Literal["nodal", "element"]
TargetSource: TypeAlias = Literal["auto", "pg", "label", "tag"]
LinkType:    TypeAlias = Literal["beam", "rod"]
Coupling:    TypeAlias = Literal["rigid", "spring"]
DOF:         TypeAlias = Literal[1, 2, 3, 4, 5, 6]

# ── Target polymorphism ───────────────────────────────────────────
TargetLike: TypeAlias = (
    PartLabel | LabelName | PGName | SelectionName | list[DimTag]
)
```

### 3.1 Why aliases and not bare primitives

`def resolve(self, node_tags: NDArray[np.int64], ...)` tells the
reader "an array of int64". `def resolve(self, node_tags: NodeTags,
...)` tells the reader "the array of mesh node tags the broker
emitted". Both are accepted by mypy; only one tells the truth.

Aliases also give us **cheap invariants** without runtime overhead.
When `NodeTags` is always `NDArray[np.int64]`, a function that takes
`NodeTags` and returns `NodeTags` is telling the reader that *the
dtype is preserved*. If someone later widens the return to
`NDArray[np.int32]`, the type checker catches it.

### 3.2 Literal types replace every magic string

Every string parameter whose valid values are a fixed set becomes a
`Literal`. This turns runtime errors into edit-time errors:

```python
# BAD — typo at runtime
g.loads.surface.pressure("slab", reduction="tributery")  # misspelled
#                                  ↑ raises only at resolve() time

# GOOD — typo at edit time
Reduction: TypeAlias = Literal["tributary", "consistent"]
def surface(..., reduction: Reduction = "tributary"): ...
g.loads.surface.pressure("slab", reduction="tributery")  # mypy: error
```

Every Def/composite parameter currently spelled as a free-form string
with a finite valid set must migrate to `Literal`. Current offenders:

| Parameter     | Current type | Migrate to                                 |
| ------------- | ------------ | ------------------------------------------ |
| `reduction`   | `str`        | `Reduction`                                |
| `target_form` | `str`        | `TargetForm`                               |
| `target_source` | `str`      | `TargetSource`                             |
| `link_type`   | `str`        | `LinkType`                                 |
| `direction`   | `str \| tuple` | `DirLike`                                |
| `weighting`   | `str`        | `Literal["uniform", "length", "area"]`    |
| `coupling`    | (new field)  | `Coupling`                                 |

---

## 4. Dataclasses for Defs and Records

Tenet (ix) partitions the class space into three flavours. Here's
the typing contract each one honours:

### 4.1 Def dataclasses (pre-mesh intent)

- `@dataclass` — no manual `__init__`.
- `kind: str = field(init=False, default="<kind_name>")` — fixed per
  subclass; never settable by the caller.
- All mutable collection fields use `field(default_factory=...)`,
  never `= []` or `= {}`.
- Every field has an explicit type annotation; none is inferred.
- Fields that control dispatch (e.g. `reduction`, `target_form`) use
  `Literal` types.

```python
@dataclass
class LineLoadDef(LoadDef):
    kind: str = field(init=False, default="line")
    magnitude: float = 0.0
    direction: DirLike = (0.0, 0.0, -1.0)
    q_xyz: Vec3 | None = None
    reduction: Reduction = "tributary"
    target_form: TargetForm = "nodal"
```

### 4.2 Record dataclasses (post-mesh, resolved)

- `@dataclass` — same as Defs.
- **Numpy arrays use `NDArray[dtype]`** — never bare `np.ndarray`.
- **`kind: str = field(init=False, default="<kind>")`** — same rule.
- Methods like `constraint_matrix` and `expand_to_pairs` declare
  return types explicitly (`-> NDArray[np.float64]`,
  `-> list[NodePairRecord]`).

### 4.3 Composite classes

- `__init__(self, parent: _ApeGmshSession) -> None` with the parent
  imported under `if TYPE_CHECKING:` to avoid the circular import.
- Every factory method returns its Def subtype (not the `LoadDef`
  base) so `reveal_type()` at the call site shows the concrete shape.
- Resolvers return record-set types (`NodalLoadSet`,
  `NodeConstraintSet`), never bare `list`.

---

## 5. Generics and Protocols

### 5.1 Generic record sets

`NodalLoadSet`, `NodeConstraintSet`, `MassSet`, and any future record
sets share a shape. If the broker grows another record-bearing
subsystem, extract the base:

```python
from typing import Generic, Iterator, TypeVar

RecordT = TypeVar("RecordT", bound=LoadRecord | ConstraintRecord | MassRecord)

class RecordSet(Generic[RecordT]):
    def __iter__(self) -> Iterator[RecordT]: ...
    def by_pattern(self, name: str) -> list[RecordT]: ...
    def filter(self, kind: str) -> "RecordSet[RecordT]": ...
```

Concrete subclasses (`NodalLoadSet(RecordSet[NodalLoadRecord])`) give
the type checker the exact record type at every iteration site.

### 5.2 Protocols for solver adapters

Solver adapters (OpenSees, Abaqus, Code_Aster) should consume records
through a `Protocol`, not a shared base class. A protocol lets
implementations stay loose-coupled:

```python
from typing import Protocol

class ConstraintEmitter(Protocol):
    def emit_node_pair(self, rec: NodePairRecord) -> None: ...
    def emit_node_group(self, rec: NodeGroupRecord) -> None: ...
    def emit_interpolation(self, rec: InterpolationRecord) -> None: ...
```

`_opensees_constraints.py` then gets a concrete class that satisfies
the protocol without inheriting. New solvers do the same, no
registry.

### 5.3 Overloads for targeting

The loads factory methods accept four mutually-exclusive targeting
modes (`target`, `pg=`, `label=`, `tag=`). The current implementation
merges them into one signature with `None` defaults. The user-facing
surface is right; the *type signature* understates it. For public
factories where the exclusivity matters, use `@overload`:

```python
from typing import overload

@overload
def point(self, target: TargetLike, *,
          force_xyz: Vec3 | None = None,
          moment_xyz: Vec3 | None = None,
          name: str | None = None) -> PointLoadDef: ...
@overload
def point(self, *, pg: PGName,
          force_xyz: Vec3 | None = None,
          moment_xyz: Vec3 | None = None,
          name: str | None = None) -> PointLoadDef: ...
@overload
def point(self, *, label: LabelName,
          force_xyz: Vec3 | None = None,
          moment_xyz: Vec3 | None = None,
          name: str | None = None) -> PointLoadDef: ...
@overload
def point(self, *, tag: list[DimTag],
          force_xyz: Vec3 | None = None,
          moment_xyz: Vec3 | None = None,
          name: str | None = None) -> PointLoadDef: ...
def point(self, target: TargetLike | None = None, *,
          pg: PGName | None = None,
          label: LabelName | None = None,
          tag: list[DimTag] | None = None,
          ...): ...  # real impl
```

Overloads document intent to both the type checker and the reader:
*you must use exactly one of `target`, `pg=`, `label=`, `tag=`*.

---

## 6. Imports and layering

### 6.1 `TYPE_CHECKING` for heavy / circular imports

The session parent type is only needed by the type checker, not at
runtime. Import it under the gate:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession
```

Apply the same pattern to:

- `gmsh` in modules that must not have a runtime dependency on it
  (all of `solvers/_constraint_*.py`, `solvers/Loads.py`). Runtime
  uses are confined to the composites; the resolvers import `gmsh`
  only for type annotations, which `TYPE_CHECKING` hides from the
  import graph.
- Cross-composite references (e.g. `LoadsComposite` referencing
  `PartsRegistry` for typing only) to keep circular imports from
  forming.

### 6.2 No star imports

`from apeGmsh.solvers.Loads import *` is banned. The public re-export
surface is defined by `__all__` in each module; callers import what
they name. `ruff` enforces this with `F403`.

---

## 7. Numpy typing

### 7.1 Always typed, always shape-commented

```python
# GOOD
def resolve(
    self,
    node_tags: NodeTags,            # shape (n_nodes,)
    node_coords: NodeCoords,        # shape (n_nodes, 3)
    elem_tags: ElemTags | None = None,
    connectivity: Connectivity | None = None,
) -> NodalLoadSet: ...

# BAD
def resolve(self, node_tags, node_coords, elem_tags=None, connectivity=None):
    ...
```

The shape is not part of the Python type system today (PEP 646
variadic generics are young and mypy support is partial). A
`# shape (n_nodes, 3)` comment on the parameter line is the honest
compromise. When we adopt PEP 646 fully, `NDArray[np.float64,
Shape["N, 3"]]` replaces the comment.

### 7.2 Never `np.ndarray` bare

`np.ndarray` on its own carries no dtype information. Use
`NDArray[np.float64]` (or `np.int64`, `np.bool_`, etc.) at a
minimum. The alias table in §3 already packages the common cases.

### 7.3 Array → tuple at record-build time

Records carry `Vec3` (a tuple), not a `(3,)` ndarray. This is
deliberate: records are solver-agnostic and dataclass-hashable, and
tuples round-trip through JSON / msgpack cleanly. The cost is one
`tuple(float(v) for v in arr)` cast per record; the benefit is that
every downstream consumer sees the same type.

---

## 8. Docstring contract

Docstrings *complement* type hints; they never duplicate them.

### 8.1 Numpy-style, not type-restated

```python
# BAD — repeats the type
def surface(
    self,
    target: TargetLike | None = None,
    *,
    magnitude: float = 0.0,
    ...,
) -> SurfaceLoadDef:
    """
    Parameters
    ----------
    target : TargetLike | None
        ...
    magnitude : float
        ...
    """

# GOOD — says what the type can't
def surface(
    self,
    target: TargetLike | None = None,
    *,
    magnitude: float = 0.0,
    ...,
) -> SurfaceLoadDef:
    """Pressure or traction on surface(s) of *target*.

    A positive magnitude with ``normal=True`` is pressure *into* the
    face (outward normal points away from the loaded side). With
    ``normal=False`` the sign of magnitude is in the ``direction``
    frame.

    Raises
    ------
    ValueError
        If neither ``target`` nor any of ``pg=``, ``label=``,
        ``tag=`` is provided.
    """
```

The type checker will enforce the types; the docstring explains the
*semantics* the types cannot express (sign conventions, coupling
between parameters, raised exceptions, side effects).

### 8.2 Mandatory sections for public methods

- One-line summary (imperative mood).
- Optional extended description (when semantics are non-trivial).
- `Raises` section for every exception the method may raise.
- `Examples` section for every factory method that could plausibly be
  called from a user script.

### 8.3 Shape comments for numpy parameters

As in §7.1 — add a `# shape (...)` comment on the parameter line,
not in the docstring.

---

## 9. Tooling — what CI runs

Three tools, layered. Each is configured in `pyproject.toml`.

### 9.1 `ruff` — lint and format

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",       # pycodestyle errors
    "F",       # pyflakes
    "I",       # isort
    "N",       # pep8-naming
    "UP",      # pyupgrade
    "B",       # flake8-bugbear
    "ANN",     # flake8-annotations  <-- mandates type hints
    "RUF",     # ruff-specific
    "SIM",     # flake8-simplify
    "TID",     # flake8-tidy-imports
    "ARG",     # flake8-unused-arguments
    "PL",      # pylint
]
ignore = [
    "ANN101",  # missing-type-self       (too noisy; self is always Self)
    "ANN102",  # missing-type-cls
    "ANN401",  # any-type                 (allowed in tight internals)
]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["ANN"]    # tests get latitude
"**/__init__.py" = ["F401"]  # re-exports
```

Key rules the lint layer enforces:

- **ANN001, ANN201** — every public argument and return typed.
- **ANN003** — no untyped `**kwargs`.
- **B008** — no mutable default argument (use `field(default_factory=...)`).
- **UP007** — use `X | Y` not `Union[X, Y]` (py3.10+).
- **N803, N806** — `snake_case` for variables and arguments.

### 9.2 `mypy` — strict type checking on the public surface

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_optional = true

[[tool.mypy.overrides]]
module = ["gmsh.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["apeGmsh.viewers.*"]
# Qt/PyVista-heavy modules: allow limited Any until stubs improve.
disallow_any_expr = false
```

`strict` is the goal. Known exemptions go in `overrides` with a
comment explaining why. No per-line `# type: ignore` without a
reason: `# type: ignore[import-not-found]  # gmsh is vendored`.

### 9.3 `pyright` — secondary IDE-driven check

```jsonc
// pyrightconfig.json
{
    "include": ["src/apeGmsh"],
    "exclude": ["**/worktrees", "**/__pycache__"],
    "typeCheckingMode": "strict",
    "reportMissingImports": "error",
    "reportMissingTypeStubs": "warning",
    "reportPrivateImportUsage": "warning",
    "useLibraryCodeForTypes": true
}
```

`pyright` is faster than `mypy` and drives the VS Code / Pylance
experience. CI runs both; divergences are a bug in one tool's
inference and should be filed upstream — not hacked around.

### 9.4 `ANN401` — the `Any` policy

`Any` is allowed **only** in two places:

1. **Boundaries with untyped third-party code** (Gmsh's Python API,
   OpenSeesPy's runtime). Wrap the call, narrow the type at the
   boundary, document with `# Any at boundary: gmsh returns
   list[int] but typed as Any`.
2. **Tight numeric kernels** where `NDArray` operators erase types
   through broadcasting. Confined to one file, reviewed explicitly.

Everywhere else, `Any` is a code smell. CI flags it via `ANN401`.

---

## 10. Examples — bad vs good

### 10.1 Factory method

```python
# BAD
def tie(self, master, slave, **kwargs):
    """Tie two parts."""
    defn = TieDef(master_label=master, slave_label=slave, **kwargs)
    self.constraint_defs.append(defn)
    return defn

# GOOD
def tie(
    self,
    master_label: PartLabel,
    slave_label: PartLabel,
    *,
    master_entities: list[DimTag] | None = None,
    slave_entities: list[DimTag] | None = None,
    dofs: list[DOF] | None = None,
    tolerance: float = 1.0,
    name: str | None = None,
) -> TieDef:
    """Tie a slave surface to a master surface via shape-function interp.

    Each slave node is projected onto the closest master face, and
    the constraint equation ``u_slave = Σ N_i · u_master_i`` is
    emitted with shape-function weights.

    Raises
    ------
    KeyError
        If ``master_label`` or ``slave_label`` is not in ``g.parts``.
    """
    return self._add_def(TieDef(
        master_label=master_label, slave_label=slave_label,
        master_entities=master_entities, slave_entities=slave_entities,
        dofs=dofs, tolerance=tolerance, name=name,
    ))
```

Reading just the signature, the caller knows: two part labels
(positional or keyword), three optional entity-level scopes, a
tolerance in model units, a display name. No guessing.

### 10.2 Resolver method

```python
# BAD
def resolve_tie(self, defn, master_faces, slave_nodes):
    ...

# GOOD
def resolve_tie(
    self,
    defn: TieDef,
    master_faces: Connectivity,     # shape (n_faces, n_nodes_per_face)
    slave_nodes: set[NodeId],
) -> list[InterpolationRecord]:
    """Project each slave node onto the closest master face.

    Returns one :class:`InterpolationRecord` per slave node.  If the
    closest projection exceeds ``defn.tolerance`` the slave is
    silently skipped.
    """
    ...
```

### 10.3 Record constructor

```python
# BAD
@dataclass
class NodePairRecord(ConstraintRecord):
    master_node = 0
    slave_node = 0
    dofs = []
    offset = None

# GOOD
@dataclass
class NodePairRecord(ConstraintRecord):
    master_node: NodeId = 0
    slave_node: NodeId = 0
    dofs: list[DOF] = field(default_factory=list)
    offset: NDArray[np.float64] | None = None      # shape (3,)
    penalty_stiffness: float | None = None
```

### 10.4 Composite stored state

```python
# BAD
class LoadsComposite:
    def __init__(self, parent):
        self._parent = parent
        self.load_defs = []
        self.load_records = []
        self._active_pattern = "default"

# GOOD
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

class LoadsComposite:
    _parent: _ApeGmshSession
    load_defs: list[LoadDef]
    load_records: list[LoadRecord]
    _active_pattern: str

    def __init__(self, parent: _ApeGmshSession) -> None:
        self._parent = parent
        self.load_defs = []
        self.load_records = []
        self._active_pattern = "default"
```

Class-level annotations make the composite's state visible at a
glance — important because composites are the stateful flavour of
the three (tenet (ix)) and their state is the thing that's hardest to
reason about.

---

## 11. Migration plan

The library currently has partial coverage. Roll this out in four
ordered passes — each pass is a PR that leaves the tree green.

1. **Pass 1 — vocabulary.** Create `apeGmsh/_types.py` with the
   alias table from §3. No other code changes. All new code uses
   the aliases; old code adopts them opportunistically. CI gates:
   `ruff ANN` enabled on new files only (per-file-override).
2. **Pass 2 — public factories.** Convert every composite factory
   method to keyword-only + typed + returning the concrete Def
   subtype. `Literal` types replace magic strings. Add `@overload`
   stubs for the targeting variants on loads and constraints.
3. **Pass 3 — resolvers and records.** Every resolver method gets
   full annotations; every record dataclass tightens to `NDArray`
   and alias types. Enable `mypy --strict` on `solvers/` and
   `mesh/`.
4. **Pass 4 — the rest.** Viewers, Qt/PyVista modules, results
   glue. These get a per-module `mypy` override allowing limited
   `Any` until upstream stubs improve.

Each pass adds one module path to the strict list in
`pyproject.toml`. We do not flip global strict on day one because
the mess is real — partial strictness is a lie the team has to live
with.

---

## 12. Contributor rules — one-line form

1. No `**kwargs` in public methods. If you reach for it, stop and re-read §2.1.
2. At most one positional parameter per public method; everything else after `*`.
3. Every public parameter typed. Every public return typed.
4. String parameters with a fixed valid set become `Literal`.
5. Aliases from `apeGmsh/_types.py` beat bare `int`, `str`, `tuple[int, int]` every time.
6. `NDArray[dtype]` never bare `np.ndarray`. Shape comments on the parameter line.
7. `from __future__ import annotations` at the top of every new module.
8. `if TYPE_CHECKING:` for heavy or circular imports.
9. Docstrings explain *semantics*, never repeat types.
10. New modules enter CI's strict list. No new code ships outside strict.

---

## 13. Where this plugs in

- **Enforced by** `ruff check`, `mypy --strict`, `pyright` — all
  three run in CI. A PR that adds a `**kwargs` to a public method,
  or drops a return type, fails before review.
- **Referenced from** [[apeGmsh_principles]] (the tenets this doc
  operationalises), [[apeGmsh_architecture]] §3 (where the
  composites live), and every subsystem doc (which should adopt the
  vocabulary as it's updated).
- **Evolves with** Python version and type-checker versions. When
  PEP 646 support lands in mypy, shape annotations move from
  comments into the type system. When PEP 695 type-statement syntax
  (`type NodeId = int`) is universally supported, migrate aliases
  from `TypeAlias` to `type` statements.

Static typing is not a finished project; it's a contract maintained
over time. The only thing that keeps it honest is CI. Everything
else in this doc is how we stay readable while CI does its work.
