# ADR 0009 — No back-compat with `apeGmsh.solvers`

**Status:** Accepted

## Context

`apeGmsh.solvers` is the legacy package: `g.opensees.materials.add_*`,
`g.opensees.elements.assign`, `g.opensees.ingest.X(fem)`, and so on.
The new `apeGmsh.opensees` reshapes nearly every concept (typed
primitives, namespace API, no ingest, explicit patterns). A
back-compat shim that lets old user code keep running would have to
translate between the two models at runtime.

## Decision

**No back-compatibility shim is provided.** Apps using
`apeGmsh.solvers` continue to work against the legacy package; apps
adopting the new bridge migrate explicitly. The two packages coexist
during the migration period, but no auto-translation layer is
written.

## Alternatives considered

1. **Shim layer.** Rejected — the model differences (dict registry
   vs typed primitives, ingest vs FEM-direct, implicit pattern vs
   explicit) are too deep. A shim would be either incomplete or
   slow, and would slow down the new package's development by
   pinning its surface to the legacy one.
2. **Soft deprecation in `apeGmsh.solvers`** (warnings, scheduled
   removal). Acceptable as a follow-up; not part of this ADR.
3. **Replace `apeGmsh.solvers` in place.** Rejected — old apps
   should not break the moment the new package lands. Coexistence
   during migration is required.

## Consequences

**Positive:**

- New package develops without legacy weight.
- Zero risk of subtle behavior drift caused by translation layers.
- Clear migration target: re-write declarations in the new style.

**Negative:**

- Existing apps need migration. Mitigated by:
  - Both packages coexisting until apps catch up.
  - Migration recipes documented per concept.
  - Two surfaces are visually distinct — no risk of accidental
    "I thought I was on the new one" confusion.

## Reference

- [charter.md non-goals](../charter.md)
- Legacy package: `apeGmsh.solvers`
