# ADR 0006 — Bridge class is `apeSees`

**Status:** Accepted

## Context

The bridge class needs a name. Options surfaced during design:

- `Bridge` — concise but ambiguous; "bridge" is a domain noun in
  structural engineering.
- `OpenSeesBridge` — unambiguous at the import site.
- `apeSees` — matches the user's existing brand (the original
  research package at `C:\Users\nmora\Github\apeSees`).

## Decision

The class is named **`apeSees`**.

```python
from apeGmsh.opensees import apeSees

ops = apeSees(fem)
```

The package is `apeGmsh.opensees`. The import path (`opensees`)
identifies the engine; the class name (`apeSees`) identifies the
wrapper.

## Alternatives considered

1. **`Bridge`.** Rejected — ambiguous in a structural codebase;
   the word "bridge" appears in PG names, model descriptions, and
   recipes.
2. **`OpenSeesBridge`.** Considered. Rejected — verbose for the
   most common type in the package; the user prefers `apeSees`
   for brand alignment.
3. **`OpenSees`** as the class name. Rejected — collides with
   the OpenSees engine itself in any context where a user has both
   `apeGmsh.opensees.OpenSees` and `openseespy.opensees` open.

## Consequences

**Positive:**

- `ops = apeSees(fem)` reads cleanly.
- Brand alignment with the legacy `apeSees` package signals
  that this codebase is the canonical successor.
- Distinct enough that `from apeGmsh.opensees import apeSees`
  cannot be confused with anything in `openseespy`.

**Negative:**

- `apeSees` (CamelCase with internal capital S) violates
  `PEP 8` convention for class names. Acceptable as a brand
  spelling; documented in the import path.
- Users seeing `apeSees` for the first time may not immediately
  know what it is. Mitigated by the package import path
  (`apeGmsh.opensees`) which establishes context.

## Reference

- Legacy package: `C:\Users\nmora\Github\apeSees`
- [charter.md](../charter.md)
