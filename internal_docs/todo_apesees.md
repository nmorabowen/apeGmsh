# apeSees — running to-do / tally

Tracking known gaps and desired implementations in the apeSees OpenSees bridge.
Status legend: `OPEN` (not started) · `WIP` · `DONE` · `PLANNED` (design accepted).

**Loads restructure** — LOAD-1 / LOAD-2 / DOC-1 / DOC-2 are addressed by
[ADR 0050](../src/apeGmsh/opensees/architecture/decisions/0050-dimension-indexed-loads-and-displacements.md)
+ [plan](plan_loads_displacements_restructure.md): dimension-indexed
`g.loads`, a new `g.displacements` composite, cross-dim gravity, and the
bridge-emit half that closes LOAD-1.

| ID | Status | Summary |
|----|--------|---------|
| LOAD-1 | SCOPED | element-form loads (`fem.elements.loads`) never emit + gravity volume-only — ADR 0050 P4 ([scope](scope_p4_crossdim_gravity_and_load1.md)). NOTE: nodal `g.loads` *do* emit (`_emit_broker_loads`); LOAD-1's "all ignored" was too broad |
| LOAD-2 | PLANNED | `body_force=` docstrings are 3D/volume-only (it's the per-solid-element arg) — ADR 0050 P5 |
| PATTERN-1 | OPEN | No path for multiple time series of same type with different factors |
| DOC-1 | PLANNED | `guide_opensees.md` ⇄ skill contradict on `g.loads` auto-emit — ADR 0050 P5 |
| NODE-1 | OPEN | No verb to create a user-defined node on the bridge (see ADR 0049) |
| DOC-2 | PLANNED | Namespace method docstrings (`ops.uniaxialMaterial.*`, `ops.nDMaterial.*`) missing/one-liners — ADR 0050 P5 |
| BRIDGE-1 | OPEN | broker `fem.nodes.sp` (g.constraints.bc, g.displacements) + `fem.nodes.masses` (g.masses) never auto-emit — asymmetric with broker loads/constraints. Found scoping P4; symmetric-broker-emission item |

---

## Implementations to work on

### LOAD-1 — `g.loads` silently ignored
apeSees silently ignores all `g.loads` declarations; the `body_force=` workaround is **not** equivalent because it bypasses the tributary / consistent reduction choice.

### LOAD-2 — `body_force=` docs are volume-only
`body_force` parameter docstrings are 3D/volume-only; no 2D example in the docs.

### PATTERN-1 — multiple time series, same type, different factors
No documented path for multiple time series of the same type with different factors.

### DOC-1 — contradictory docs on `g.loads` auto-emit
`guide_opensees.md` and the skill directly contradict each other on whether `g.loads` auto-emits.

### NODE-1 — user-defined bridge node verb
No verb to create a user-defined node on the bridge; the only non-mesh nodes today are `node_to_surface` phantom nodes, created implicitly by the constraint system. Reference node ndf must be **explicit** — silent fallback to global model ndf is wrong for mixed-ndf models.

### DOC-2 — namespace method docstrings
Namespace method docstrings (`ops.uniaxialMaterial.*`, `ops.nDMaterial.*`) are missing or one-liners; the underlying dataclasses are documented but the IDE never shows those.
