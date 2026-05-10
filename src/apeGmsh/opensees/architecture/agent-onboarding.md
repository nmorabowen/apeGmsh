# Agent onboarding

Each agent picks up one slice from
[parallel-execution.md](parallel-execution.md) and owns it
end-to-end: typed classes + namespace methods + unit tests +
contract list update.

This document is the **prompt template** the coordinator uses to
spin up a slice agent. Copy it, fill in the bracketed values, and
hand it to the agent.

## Universal context (every agent starts here)

The agent needs to read these in order before doing anything else:

1. [README.md](README.md) — TL;DR + reading order
2. [charter.md](charter.md) — the 14 principles + non-goals
3. [layout.md](layout.md) — folder structure + naming
4. [api-design.md](api-design.md) — the namespace API + capability shapes
5. [emitter.md](emitter.md) — the Protocol the agent's `_emit`
   targets
6. [testing.md](testing.md) — test layers + conventions
7. [parallel-execution.md](parallel-execution.md) — where this
   slice sits in the dependency graph

After that: the slice-specific section in `parallel-execution.md`
plus the relevant ADRs.

## The standard slice

Most slices share a shape. An agent owning a primitive family
delivers:

```
src/apeGmsh/opensees/<family>/<file>.py
    typed dataclass per OpenSees type
    matching namespace class with one method per type

tests/opensees/unit/primitives/test_<family>_<file>.py
    one TestClass per OpenSees type
    construction, validation, _emit, dependencies, repr

tests/opensees/contract/test_<family>_contract.py
    ALL_<KIND> list updated to include the new types
    (this file already exists; the agent appends)
```

Plus a PR with title `opensees: phase-<N><X>-<slice-name>`.

## Slice prompt template

Copy this, fill the brackets, paste it as the agent's initial
prompt:

```
You are implementing slice [PHASE] of the apeGmsh.opensees package.

CONTEXT (read these in order before anything else):
  1. src/apeGmsh/opensees/architecture/README.md
  2. src/apeGmsh/opensees/architecture/charter.md
  3. src/apeGmsh/opensees/architecture/layout.md
  4. src/apeGmsh/opensees/architecture/api-design.md
  5. src/apeGmsh/opensees/architecture/emitter.md
  6. src/apeGmsh/opensees/architecture/testing.md
  7. src/apeGmsh/opensees/architecture/parallel-execution.md

  Then read these specifically for your slice:
  - parallel-execution.md § "[PHASE_SECTION]"
  - [SPECIFIC_ADRS]

YOUR SLICE: [PHASE] — [SLICE_NAME]

Files you own (no other agent will edit these):
  - src/apeGmsh/opensees/[FILE_PATH]
  - tests/opensees/unit/primitives/[TEST_FILE]

Files you APPEND to (other agents may also append; do not reorder):
  - tests/opensees/contract/[CONTRACT_TEST]
    Add your new types to the [ALL_LIST] list.

Files you READ but do not modify:
  - src/apeGmsh/opensees/_internal/types.py — base classes
  - src/apeGmsh/opensees/emitter/base.py — Emitter Protocol
  - src/apeGmsh/opensees/emitter/recording.py — RecordingEmitter

CLASSES TO IMPLEMENT:
  [list of types, with parameter shapes from OpenSees docs / SRC]

For each class:
  1. @dataclass(frozen=True, kw_only=True, slots=True)
  2. inherits from [BASE_CLASS] in _internal/types.py
  3. fully typed parameters (no **kwargs, no positional *args
     except where the OpenSees command itself takes a variadic
     list)
  4. parameter validation in __post_init__ if needed (e.g.
     Concrete02 sign-flips)
  5. _emit(self, emitter: Emitter, tag: int) -> None implementation
     that calls the matching emitter method with positional args
     in the OpenSees order
  6. dependencies(self) -> tuple[Primitive, ...] returning
     children (for sections / elements that compose primitives;
     leaf primitives return ())
  7. matching method on the namespace class:
     def TypeName(self, *, <typed kwargs>) -> TypeName:
         return self._bridge._register(TypeName(**kwargs))

For each typed class, write a unit test (per the template in
testing.md § "Layer 1 — Unit") with at minimum:
  - test_construction
  - test_emit_records_correct_call (RecordingEmitter)
  - test_dependencies (empty for leaves; expected refs for composers)
  - test_validation (the per-class invariants)
  - test_repr_includes_type_token

When done:
  - mypy --strict src/apeGmsh/opensees/[FILE_PATH] passes
  - ruff check src/apeGmsh/opensees/[FILE_PATH] passes
  - pytest tests/opensees/unit/primitives/[TEST_FILE] passes
  - pytest tests/opensees/contract/ passes (your additions)

Report back when complete:
  - Files created
  - Number of classes implemented
  - Test count
  - Any deviations from the slice spec, with rationale
```

## Specialized slice templates

### Emitter slice (Phase 4)

The standard template, with these substitutions:

- File: `src/apeGmsh/opensees/emitter/[name].py`
- Test file: `tests/opensees/unit/test_emitter_[name].py`
- Plus parity tests: `tests/opensees/parity/test_[name]_parity.py`
- Read `parallel-execution.md` § "Phase 4" instead of Phase 1

Specific deliverables:

1. Implement every method on the `Emitter` Protocol.
2. Handle the `*_open` / `*_close` pairs correctly for the dialect
   (Tcl uses braces; Py uses stateful current-X).
3. Write parity tests that drive the standard fixtures
   (frame_3d, arch_csys, tank_cylindrical) through your emitter
   AND through `RecordingEmitter`, then assert structural
   equivalence.

### Recipe slice (Phase 7)

The standard template, with these substitutions:

- File: `src/apeGmsh/opensees/recipes/[name].py`
- Test file: `tests/opensees/unit/recipes/test_[name].py`

Specific deliverables:

1. The recipe is a function or class that takes high-level
   engineering parameters and returns a primitive (typically a
   Section).
2. The returned primitive is constructed via the same typed
   primitive classes the bridge exposes — no shortcuts.
3. Tests verify the recipe produces the expected fiber/patch
   layout via `RecordingEmitter`.

### Aggregate slice (Phase 5)

Different from primitive slices because Node and ElementGroup
aggregate. The template adds:

- Read `api-design.md` § "Aggregate types" carefully.
- The aggregate composes existing primitives; it doesn't define
  new emit shapes.
- Tests live in `tests/opensees/integration/` (not `unit/`)
  because aggregates exercise multiple primitive types together.

## Coordinator responsibilities

The coordinator agent (or human) owns:

1. **Spinning up slice agents** with the filled-in template.
2. **Reviewing PRs** at each sync point per the criteria in
   `parallel-execution.md`.
3. **Maintaining the contract test lists** (`ALL_*`) in
   coordination order, not commit order.
4. **Resolving merge conflicts** when two PRs touch the same
   contract list — append-only protocol means conflicts are
   line-level and trivial.
5. **Updating `parallel-execution.md`** with ✅ markers and PR
   refs as phases complete.

## Common pitfalls (and how to avoid them)

- **Forgetting to add to the contract list.** The contract test
  is parametrized over an explicit list. If a class isn't in the
  list, no contract test runs against it. CI cannot catch this —
  the coordinator must check at PR review.
- **`**kwargs` slipping into a user-facing signature.** P12
  forbids it. The agent's `mypy --strict` should catch this; the
  coordinator double-checks at review.
- **Hardcoding tags.** Primitives never know their tag at
  construction. The bridge's `TagAllocator` provides them at
  build/emit. If a test expects `tag=1`, it's wrong — let the
  allocator decide and assert against `bm.allocator.tag_for(prim)`.
- **Importing openseespy from a primitive.** P2 forbids it. The
  primitive calls `emitter.X(...)`; nothing else.
- **Editing `emitter/base.py`** to add a method "just for this
  primitive." The Protocol is frozen after Phase 0. If a new
  method is genuinely needed, escalate to the coordinator —
  it's an architecture event.
- **Writing tests against `LiveOpsEmitter` in Phase 1.** Use
  `RecordingEmitter` only. Live tests are a separate layer
  (Phase 5 / Phase 6 territory) and need the live marker.
- **Skipping `mypy --strict`.** Type checking is non-optional.
  PRs that don't pass strict don't merge.

## When in doubt

The agent's default move when uncertain:

1. Re-read the relevant ADR.
2. If the ADR doesn't answer it, the answer is in the OpenSees
   source at `C:\Users\nmora\Github\OpenSees_Compile\OpenSees\SRC`.
   Read the actual class headers / commands.
3. If neither resolves it, ask the coordinator. Do not invent.

The "do not invent" rule is the most important coordination
mechanism. Inventions diverge across agents; questions surface
sooner.
