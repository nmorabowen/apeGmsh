# apeSees — running to-do / tally

Tracking known gaps and desired implementations in the apeSees OpenSees bridge.
Status legend: `OPEN` (not started) · `WIP` · `DONE`.

| ID | Status | Summary |
|----|--------|---------|
| LOAD-1 | OPEN | `g.loads` declarations silently ignored by apeSees |
| LOAD-2 | OPEN | `body_force=` docstrings are 3D/volume-only |
| PATTERN-1 | OPEN | No path for multiple time series of same type with different factors |
| DOC-1 | OPEN | `guide_opensees.md` ⇄ skill contradict on `g.loads` auto-emit |
| NODE-1 | OPEN | No verb to create a user-defined node on the bridge |
| DOC-2 | OPEN | Namespace method docstrings (`ops.uniaxialMaterial.*`, `ops.nDMaterial.*`) missing/one-liners |

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
