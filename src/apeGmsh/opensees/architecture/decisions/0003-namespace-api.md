# ADR 0003 — Namespace API + static typing

**Status:** Accepted

## Context

OpenSees commands have the form `command <type> <tag> <params>`.
For a Python wrapper, we have to choose how the user names the type:

- (A) Type as a string: `ops.uniaxialMaterial("Steel02", fy=...)`
- (B) Type as a method: `ops.uniaxialMaterial.Steel02(fy=...)`

(A) reads literally as OpenSees Tcl. (B) is more Pythonic and
autocompletes. Both are defensible.

A separate constraint: **no `**kwargs` in user-facing signatures**
(P12). Static typing is required.

## Decision

Use **(B) — namespace API**. Each OpenSees command with type variants
is a namespace on the bridge; each variant is a typed method on that
namespace.

```python
ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
ops.element.forceBeamColumn(pg="Cols", section=sec, transf=t, n_ip=5)
ops.integrator.LoadControl(increment=0.05)
```

Commands without type variants are flat methods on the bridge:

```python
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
ops.analyze(steps=20)
```

Class names match OpenSees type tokens exactly, including
`lowerCamelCase` where OpenSees uses it (`forceBeamColumn`,
`elasticBeamColumn`).

## Alternatives considered

1. **(A) Type as a string with `Literal[]` overloads.** This
   preserves the "reads as OpenSees" surface and gives static typing.
   Rejected because the overload list per command is huge (~50
   uniaxial materials, ~30 nD materials, ~40 elements), and pyright
   handles them but the boilerplate is worse than the namespace
   approach. Also the "type as string" form makes refactoring
   silent — renaming a type breaks at runtime, not at the IDE.
2. **(C) Hybrid — both forms work.** Considered. Rejected because
   "two ways to do it" introduces decision fatigue and divergent
   coding styles in a single codebase. Pick one.
3. **String form with no static typing.** Rejected — P12 is a
   load-bearing principle.

## Consequences

**Positive:**

- IDE autocomplete on type names: `ops.uniaxialMaterial.<TAB>`.
- IDE autocomplete on parameter names per type.
- Pyright catches typos in type names ("Steel20" → unknown).
- Refactor safety: rename a type, IDE finds all usages.
- Reading the API tree, the user discovers the OpenSees command
  surface by tabbing.

**Negative:**

- Signature duplication: each typed class's parameters appear on
  the dataclass AND on the namespace method that constructs it.
  Mitigated by future code-gen if the duplication becomes painful;
  hand-written for v1 because explicitness wins on PRs.
- Marginal departure from "code reads as Tcl": `ops.uniaxialMaterial.Steel02(fy=...)`
  is two attribute accesses, not `uniaxialMaterial Steel02 fy=...`
  inline. Acceptable — the type tokens are identical.

## Reference

- [charter.md P12](../charter.md)
- [api-design.md](../api-design.md)
