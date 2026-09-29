# ADR 0002 — Typed primitives carry capabilities

**Status:** Accepted

## Context

Today's `apeGmsh.solvers.OpenSees` stores material/section/transform
declarations as `dict[name, dict[str, Any]]` blobs:

```python
ops._uni_materials["S"] = {"ops_type": "Steel02", "params": {"fy": ...}}
```

There is no Python type for "Steel02 material." Pyright sees `dict`;
the IDE cannot autocomplete parameters; validation is duck-typed at
emit time; capabilities (plotting, testing) cannot live on the
material because the material is not an object.

The legacy `apeSees` package (separate experimental codebase) already
demonstrated the pattern we want: each material is a typed Python
class (`Steel02`, `Concrete02`) with `.build()` and per-class
capabilities (`.tester`, plotting, parameter validation).

## Decision

Every OpenSees concept the user touches is a typed Python class.
Materials, sections, transforms, time series, patterns, elements,
recorders, analysis components — all dataclasses with full type
annotations and named parameters.

Capabilities ride on the instance:

- `Material.plot.backbone(...)`, `.test.cyclic(...)`, `.check.parameters()`
- `Section.plot()`, `.moment_curvature(...)`, `.dependencies()`
- `TimeSeries.plot()`, `.peak_value`, `.fft()`
- `ElementGroup.plot()`, `.summary()`, dependencies
- `Pattern` aggregates its loads for inspection

Each class implements `_emit(emitter: Emitter, tag: int) -> None` for
the emit boundary (see [emitter.md](../emitter.md)).

## Alternatives considered

1. **Keep the dict-based registry.** Rejected — no static typing,
   no autocomplete, validation deferred to emit, no place for
   capabilities to live.
2. **Port the apeSees package wholesale.** Rejected by the user —
   apeSees has scope creep (research code, neural moment-curvature)
   we don't want, and it's directly coupled to live `ops` rather
   than emit-target-agnostic.
3. **Typed primitives but no capabilities (data-only classes).**
   Rejected — capabilities are a major user value (material studies,
   section M-φ, frame transform visualization) and they belong on
   the object whose data they describe.

## Consequences

**Positive:**

- Pyright/mypy validates parameters at the call site.
- IDE autocomplete on parameter names and types.
- Per-class validation in `__init__` (Concrete02 enforces signs,
  Steel02_ape derives `b` from `fu, epsilon_u`).
- Per-class smart defaults (Concrete02 derives
  `max_tensile_strain` from softening properties).
- Capabilities co-located with data — discoverable in the IDE.
- Subclassing for engineering ergonomics (`Steel02_ape` is an
  alternate parameterization of `Steel02` that emits the same
  OpenSees command).

**Negative:**

- Each new OpenSees type requires a typed class. Boilerplate.
  Mitigated by `@dataclass(frozen=True, kw_only=True, slots=True)`.
- Per-class signature must match the namespace method's signature
  (see ADR 0003). Hand-written for v1; potential code-gen later.
- "Standalone instance vs registered with bridge" creates a small
  two-state issue (P11). Worth the cost — material studies need
  standalone use.

## Reference

- [charter.md P1, P11](../charter.md)
- [api-design.md](../api-design.md)
- Legacy reference: `C:\Users\nmora\Github\apeSees\src\apeSees\materials\base.py`
