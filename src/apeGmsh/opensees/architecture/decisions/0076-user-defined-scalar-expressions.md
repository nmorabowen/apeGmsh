# ADR 0076 — User-defined scalar expressions on Results

**Status:** Accepted (2026-07-15). Slices 1–3 (engine → composite wiring
→ viewer reach) shipped in #796; Slice 4 (`mag()` vector helper +
sidecar persistence + stage-derivation propagation) shipped immediately
after. All four slices are landed; this document reflects the shipped
design.

## Context

Users want to visualize arithmetic combinations of recorded fields the
library does not ship as a named component — e.g. a demand/capacity
ratio `von_mises_stress / f_y`, a kinetic-energy-like node scalar
`velocity_x**2 + velocity_y**2`, a normalized settlement. Today the only
way is to pull slabs out with `results.nodes.get(...)` / `.gauss.get(...)`
and do the numpy by hand in a notebook — which never reaches the viewer
picker, so the field cannot be colored on the mesh or scrubbed in time.

The machinery to compute a scalar *on read from named base arrays*
already exists for the shipped invariants. `results/_derived.py`
(ADR-less, PR #784) computes von Mises / principal / Tresca / J2 / Lode
/ triaxiality from the stored Voigt tensor columns at read time; the
dispatch lives in `results/_composites.py`:

- `GaussResultsComposite.get(component=...)` checks
  `_derived.is_derived(name)` → `_compute_derived(...)` before falling
  through to `reader.read_gauss(...)` (lines ~1071–1081).
- `available_components()` returns `stored + available_derived(stored)`
  — and **this list is what the viewer scalar picker shows**.

Two structural facts constrain any custom-expression design:

1. **Fields are domain-partitioned.** Node fields (`displacement_x`,
   `velocity_y` — scalar per DOF) live on the node set; Gauss fields
   (`stress_xx` …) live on the Gauss-point set; fiber fields live on a
   third. These are different point counts and different array shapes —
   they cannot be combined in one field. `_derived` only ever composes
   within the Gauss tensor columns for exactly this reason. The user's
   motivating example (`v² + displacement`) is a **node** field.

2. **The viewer is an out-of-process, disk-driven consumer** (ADR 0014).
   `results.viewer()` / `show_web()` spawn `python -m apeGmsh.viewers`
   against the results file on disk and read a `<results>.viewer-session.json`
   sidecar. Anything that must appear in the picker has to survive
   serialization to that subprocess — a Python callable cannot; a string
   expression can. This is the decisive reason the recipe is declarative
   text, not a registered lambda.

## Decision

Add **user-registered named scalar expressions**, evaluated on read by a
restricted numpy evaluator — a direct sibling of `_derived`, differing
only in that the recipe is user-supplied instead of hard-coded.

### Registration is per-composite; the domain is the composite

`define()` / `undefine()` / `definitions` hang on the composite whose
domain they belong to. The domain is *implied by where you register* —
there is no `domain=` argument and therefore no way to express an
illegal cross-domain field:

```python
results.nodes.define(
    "kinetic_ish", "velocity_x**2 + velocity_y**2 + displacement_x",
)
results.elements.gauss.define("dcr", "von_mises_stress / 250.0", units="-")

results.nodes.get(component="kinetic_ish", pg="deck", time=-1)   # NodeSlab
results.nodes.available_components()        # -> [..., "kinetic_ish"]
results.nodes.definitions                    # {"kinetic_ish": ExprDef(...)}
results.nodes.undefine("kinetic_ish")
```

`define(name, expr, *, label=None, units=None)` — `label` / `units` are
display-only metadata for the picker / legend (default: `label=name`,
`units=""`). They never affect evaluation.

### Operand namespace = the composite's `available_components()` at define time

Identifiers in the expression resolve against exactly the components that
composite can already serve — **stored columns and existing derived
scalars alike**. So `von_mises_stress / 250.0` works because
`von_mises_stress` is an `available_component` of the Gauss composite;
custom names may themselves reference earlier custom names, resolved the
same way. There is no separate operand registry.

### The expression grammar is a restricted AST over numpy — never `eval`

A new module `results/_expr.py` owns:

- `ExprDef` — a frozen dataclass `(name, expr, operands, label, units)`;
  `operands` is the resolved identifier set, captured at define time.
- a compiler that parses `expr` with `ast.parse(mode="eval")` and walks
  the tree, **rejecting every node type not on an allow-list**:
  - `BinOp` over `+ - * / // % **` plus elementwise `& |` (numpy
    bitwise-on-bool, for combining comparison results), `UnaryOp` over
    `+ - `, `Compare`;
  - `Call` only to a fixed function table, each bound to a numpy
    implementation that is **safe on `(T, N)` arrays**:
    - elementwise unary/binary: `sqrt abs exp log sin cos tan sign`,
      `hypot` (2-arg), `clip` (3-arg);
    - `minimum` / `maximum` — the **elementwise** pair, arity exactly 2.
      `min` / `max` are **not** exposed: numpy's `np.min`/`np.max` are
      reductions that would silently collapse a `(T, N)` field to a
      scalar, and Python's builtins are variadic — either is a
      field-destroying footgun (review finding 2);
    - `where(cond, a, b)` — the elementwise conditional.
  - `Name` only for identifiers that resolve in the operand namespace;
  - `Constant` only for `int` / `float`.
  - Everything else — attribute access, subscripting, comprehensions,
    lambdas, walrus, starred, f-strings, dunder names, **and `BoolOp`
    (`and`/`or`) / `IfExp` (`a if c else b`)** — raises `ExprError` at
    compile time with the offending token.
- an evaluator that binds the operand names to their `(T, N)` arrays and
  returns a `(T, N)` float64 array.

**Why `and`/`or`/`if-else` are banned, not lowered** (review finding 1):
Python's `and`/`or`/`IfExp` dispatch on the *truthiness* of an operand,
which raises `ValueError: truth value of an array is ambiguous` on a
numpy array — they cannot be evaluated by walking the node. Supporting
them would mean *rewriting* `BoolOp`→`np.logical_and/or` and
`IfExp`→`np.where` in the compiler. That is a real translation, not a
free traversal; v1 ships the explicit `where(...)` function and boolean
`&`/`|` on comparison results instead, and the AST forms stay rejected.
Comparisons already cover the threshold use case
(`(von_mises_stress > f_y) * von_mises_stress`).

`numexpr` / `asteval` were considered and rejected (see below).

### Validation is fail-loud at `define()` — for the errors that *are*
### resolvable there

Mirrors the dimensional-resolution contract (never defer a *resolvable*
error to read time). `define()` raises immediately when:

- the expression does not parse under the restricted grammar
  (`ExprError`);
- any identifier does not resolve in `available_components()`
  (`ExprError`, listing the unknown name and the available set — same
  shape as `_compute_derived`'s "missing tensor components" message);
- the name shadows a stored or derived component, **or** an
  already-registered custom name (`ValueError`) — custom names share the
  flat component namespace, so both collisions are ambiguous and refused
  (re-`define` is *not* a silent overwrite; `undefine` first).

**What `define()` cannot catch, by construction** (review findings 3, 4).
Two failure modes are inherently read-time and the ADR does not pretend
otherwise:

- **Stage-scoped availability.** `available_components()` is
  stage-scoped (`reader.available_components(stage_id, level)`). A
  component recorded in one stage but not another means an operand valid
  at define time can be absent at read for a different `stage=`.
  `define()` validates against the **union of all stages'** available
  components (an operand recorded in *any* stage passes); a `get()` at a
  stage where that operand was not recorded then raises the ordinary
  "component not available in stage X" reader error. Validating against
  the union keeps the common single-stage / all-stages case fail-fast
  without falsely rejecting a genuinely per-stage operand.
- **Operand coverage mismatch.** Unlike the `_derived` scalars — which
  only ever combine co-recorded components of *one* tensor and so always
  share point coverage — a custom expression may combine two
  independently recorded fields (e.g. displacement on all nodes but
  velocity on a subset). For a given selection their slabs can then have
  different point counts. The evaluator therefore performs an explicit
  **shape-agreement check** across the bound operand arrays before
  evaluating and raises a legible `ExprError` ("operands 'velocity_x'
  (N=812) and 'displacement_x' (N=1024) cover different points for this
  selection") instead of a raw numpy broadcast error.

### `undefine()` refuses to strand a dependent

`ExprDef.operands` is used to guard removal: `undefine(name)` raises if
another registered definition lists `name` as an operand (naming the
dependent), so a live definition can never silently start failing at
read. Remove dependents first, or none.

### Compute-on-read wiring mirrors `_compute_derived`

Each composite gains a `_registry: dict[str, ExprDef]`. In `get()`, a
`name in self._registry` branch **precedes** the reader fallthrough:
read each operand as a slab for the same selection / stage / time, bind
`{operand: slab.values}`, evaluate, and return
`dataclasses.replace(base_slab, component=name, values=values)` — the
exact synthesis `_compute_derived` already does (lines ~1167–1172).
Operands that are themselves derived/custom recurse through the same
`get()`. `available_components()` appends the registry keys.

### Viewer reach via the session sidecar

On `viewer()` / `show_web()` launch, the active registries serialize
into a **dedicated** `<results>.defs.json` sidecar (a JSON list of
`{name, expr, domain, label, units}`), passed to the child via a
`--defs <path>` argument to `python -m apeGmsh.viewers`.
`viewers/__main__.py` reconstructs the registries on the results object
it builds, so the scalar picker lists the custom names and coloring
evaluates them through the identical read path. Because the recipe is a
string, this is plain JSON — no pickling, no code transport across the
process boundary.

**Not** the existing `<results>.viewer-session.json` (review finding 5):
that file is owned by `viewers/diagrams/_session.py`, carries
`DiagramSpec`s + active stage/step, is gated on `fem.snapshot_id`, and
is **overwritten on window close** when `save_session=True`.
Piggybacking definitions on it would let a close clobber them and
entangle them in diagram staleness refusal. Definitions get their own
file. In-process `blocking=True` launches need no sidecar at all — they
share the live Results object and its registries directly.

Slice 4 makes this same `<results>.defs.json` double as the **persistence**
store (the subprocess-transport write goes through `save_definitions`),
so a launch that carries definitions also persists them, and a later
`from_native` / `from_mpco` / `from_ladruno` auto-loads them.

### Scope boundaries

- **Scalar operands, plus the `mag()` vector helper** (Slice 4). Bare
  arithmetic operands are scalar component names (`velocity_x**2 +
  velocity_y**2`). The one vector-aware form is `mag(<family>)` — a
  special compiler form that expands a nodal vector family
  (`velocity`, `displacement`, `force`, `rotation`, `moment`, …, via
  `_vocabulary.vector_family`) to the Euclidean norm of its *recorded*
  components, clipping to the present axes (2-D → in-plane norm). It
  refuses a component name (`mag(velocity_x)`), the combined `reaction`
  shorthand (forces + moments is not one vector), and a non-bare
  argument (`mag(velocity + 1)`). No general vector algebra beyond the
  norm.
- **Persisted via the definitions sidecar** (Slice 4). `save_definitions()`
  writes `<results>.defs.json`; `from_native` / `from_mpco` /
  `from_ladruno` auto-load it at open (best-effort — a stale operand
  warns and is skipped, never blocks the open); loading is idempotent.
  This is the **same** sidecar as the viewer transport (unified — the
  subprocess launch persists too), and deliberately **not** an embedded
  H5 zone: a JSON sidecar works uniformly for read-only `.mpco` as well
  as native files. Definitions also propagate across stage / mode
  derivation (`results.stage("grav")` sees them — `_derive` shallow-
  copies the frozen `ExprDef` registry).
- **No unit checking.** `velocity**2 + displacement` is dimensional
  nonsense; the library does not police units anywhere and will not
  start here. The user owns dimensional sense.
- **No policing of NaN / inf.** `log` of a non-positive value, `sqrt` of
  a negative, or a divide-by-zero produce NaN / inf silently — exactly
  as the shipped `lode_angle` / `stress_triaxiality` derived scalars
  already do. The viewer color pipeline already tolerates NaN fields, so
  this is consistent, not a regression; not guarded.
- **Node + Gauss composites in v1.** Fiber follows the same pattern if a
  use case appears (second-use-case trigger).

### PR slices

1. **`results/_expr.py`** — `ExprDef`, restricted compiler, evaluator.
   Pure, no `Results` dependency. *Verify:* valid expressions evaluate
   correctly against hand-built column dicts; every disallowed AST node
   (`__import__`, attribute access, subscript, unknown identifier, call
   to a non-whitelisted name) raises `ExprError` at compile time.
2. **Composite wiring** in `_composites.py` — `define` / `undefine` /
   `definitions` + the `get()` branch + `available_components()`
   inclusion on the node and Gauss composites. *Verify:* define → `get`
   returns the arithmetic combination of the base slabs; the name
   appears in `available_components()`; shadowing and bad-operand define
   calls raise.
3. **Viewer reach** — serialize registries into the dedicated
   `<results>.defs.json` sidecar + `--defs` arg; `viewers/__main__.py`
   rebuilds them. *Verify:* subprocess viewer lists a defined scalar in
   its picker and renders the field (driven like the existing viewer
   gallery tests); the `viewer-session.json` save-on-close is
   untouched.
4. **`mag()` vector helper + persistence** — `_vocabulary.vector_family`
   accessor + `mag()` special form in the compiler; `save_definitions` /
   `load_definitions` + auto-load on the file-based constructors +
   `_derive` registry propagation; the viewer spawn routes through
   `save_definitions` (transport == persistence). *Verify:* `mag()`
   evaluates/clips/rejects; save → reopen auto-loads and reads; stale
   sidecar operand warns-not-fatal; a definition survives `.stage()`.

## Rejected alternatives

- **Registered Python callable** (`@results.derived("foo") def foo(cols):
  …`). Maximum power, but a callable cannot cross the viewer subprocess
  boundary or serialize into the session/H5 sidecar — so a custom field
  defined this way could never appear in the picker, which is the
  feature's whole point (ADR 0014, out-of-process viewer). Rejected.
- **`numexpr`** as the evaluator. Fast and sandboxed, but a third-party
  dependency with a fixed, small function table and no clean hook for a
  future `mag()` operand helper. The restricted-AST-over-numpy path
  needs no new dependency and stays under our control. Rejected for v1.
- **`asteval`** as the evaluator. More permissive than we want (it aims
  to run a large Python subset); tightening it back down to a safe
  allow-list is more work than walking a small AST ourselves, and it is
  another dependency. Rejected.
- **Top-level `results.define(name, expr, domain=...)`.** More
  discoverable as a single entry point, but reintroduces the
  domain-mismatch failure mode (`domain="node"` with a Gauss operand)
  that per-composite registration makes unrepresentable. Rejected.
- **A structured op-tree recipe** (`{op: "add", args: [...]}`) instead of
  a string. Serializes just as well, but is far worse to author by hand
  and buys nothing the restricted-AST string form does not already give
  (the string *is* parsed to an AST). Rejected.
- **Embedding definitions in the results H5** (a `/opensees/`-style zone)
  rather than a JSON sidecar. Rejected: `.mpco` files are read-only STKO
  archives with no place to write, and native results files are
  sometimes recreated in place; a sidecar works uniformly for both and
  unifies with the viewer transport. (Was floated as the Slice 4
  persistence target; the sidecar won.)

## Invariants

- **INV-1** — a custom scalar is evaluated through the *same* composite
  `get()` read path as any stored/derived component; no separate query
  API. Its operands are read for the identical selection / stage / time.
- **INV-2** — the evaluator executes **no** AST node outside the
  allow-list; the only callables reachable are the fixed numpy function
  table, and that table contains **only** array-safe elementwise
  functions (no reductions, no variadic `min`/`max`, no `BoolOp`/`IfExp`
  lowering). Pinned by a compile-time rejection test per disallowed node
  class and an evaluation test that every admitted function preserves
  the `(T, N)` shape.
- **INV-3** — *define-time-resolvable* errors — parse failure, an
  operand absent from **every** stage, a shadowed name — are raised at
  `define()`. The two inherently read-time failures (an operand absent
  from the *specific* `stage=` requested, and operand coverage mismatch
  for a selection) are raised at read with a legible `ExprError` /
  reader error, never as a raw numpy broadcast error. `define()` does
  not claim to certify readability at every stage/selection.
- **INV-4** — `available_components()` includes custom names, so the
  viewer picker lists them with no viewer-side special-casing.
- **INV-5** — definitions crossing to the viewer subprocess travel as
  JSON strings (name + expr + domain + display metadata) in a dedicated
  `<results>.defs.json`, never the diagram `viewer-session.json`; no
  callable or pickled object crosses the boundary.
- **INV-6** — a custom name never shadows a stored, derived, or
  already-registered custom component; `undefine()` never strands a
  dependent definition. Both refused at define / undefine time.

## Cross-references

- `results/_derived.py` (PR #784) — the on-read derived-scalar computer
  this ADR generalizes; `_compute_derived` is the structural template
  for the custom read path.
- ADR 0014 — viewer is a pure, out-of-process H5 consumer; the reason
  the recipe must be serializable text.
- Dimensional-resolution contract (`project_resolution_contract`) — the
  fail-loud-at-declaration precedent for `define()` validation.
