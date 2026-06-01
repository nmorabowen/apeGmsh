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
| LOAD-1 | DONE | **Bridge load consumption — [ADR 0051](../src/apeGmsh/opensees/architecture/decisions/0051-bridge-load-consumption.md) + [plan](plan_bridge_load_consumption.md), SHIPPED.** Loads are **opt-in** via `p.from_model(case)` (deleted `_emit_broker_loads`); geometry groups by **case**, bridge owns **patterns**; **all-nodal** (element-`eleLoad` dropped until further notice); two execution modes (non-staged `ops.analyze` / staged `s.pattern`) with a no-mixing guard (`BridgeError`); `s.remove_bc` alias. PRs #497 (BL-1 case rename) / #498 (BL-2 from_model + delete auto-emit) / #502 (BL-3 stage patterns) / #503 (BL-4 no-mixing guard; the `WarnUnconsumedModelLoads` + `ops.ignore_model_loads` reconciliation it also shipped was **subsequently removed** — deck is authoritative, ADR 0051 §7) / #504 (BL-5 remove_bc) / #506 (BL-6 docs). Element-form + cross-dim gravity (ADR 0050 §5) DEFERRED |
| LOAD-2 | PLANNED | `body_force=` docstrings are 3D/volume-only (it's the per-solid-element arg) — ADR 0050 P5 |
| PATTERN-1 | OPEN | No path for multiple time series of same type with different factors |
| DOC-1 | DONE | `guide_opensees.md` ⇄ skill no longer contradict on `g.loads`: both now state loads are **opt-in** (`p.from_model(case)`, no auto-emit). Closed by ADR 0051 BL-6 — rewrote `guide_opensees.md` §intro/§4/§9, the apegmsh-helper skill (`opensees-bridge.md` / `SKILL.md` / `api-cheatsheet.md` / `workflows.md`) |
| NODE-1 | OPEN | No verb to create a user-defined node on the bridge (see ADR 0049) |
| DOC-2 | OPEN | Namespace method docstrings (`ops.uniaxialMaterial.*`, `ops.nDMaterial.*`, + `analysis`/`element`/`section`/…) missing/one-liners. Independent of the loads reconciliation — a docstring-quality sweep, NOT closed by ADR 0051 BL-6 (which closed DOC-1). Own follow-up |
| BRIDGE-1 | REFRAMED | **Masses/constraints import round (after ADR 0051 loads).** Model-level `fix`/`mass` are already explicit (`ops.fix`/`ops.mass`); imposed-disp `sp` is opt-in via `from_model` (ADR 0051). What remains: an **import symmetry** `ops.mass.from_model(...)` / `ops.fix.from_model(...)` for `g.masses` / `g.constraints.bc`, and the "does mass ever auto" question (user: not yet). (No reconciliation-warning extension — the loads `WarnUnconsumedModelLoads` was removed; the deck is authoritative.) Its own short ADR/plan |

---

## Implementations to work on

### LOAD-1 — bridge load consumption (DONE, ADR 0051)
RESOLVED. `g.loads` no longer silently ignored *nor* silently auto-emitted:
a resolved load **case** reaches the deck via an explicit
`p.from_model(case)` import on a bridge pattern (`ops.pattern.Plain` or
the stage-scoped `s.pattern`). No auto-emit. The deck is authoritative —
the bridge applies exactly the cases a pattern imports and does NOT audit
the geometry's declared cases (the `WarnUnconsumedModelLoads` +
`ops.ignore_model_loads` reconciliation shipped in BL-4 was subsequently
removed; ADR 0051 §7). Element-form (`eleLoad`) consumption remains
DEFERRED — the all-nodal scope means a beam UDL lumps to end nodes (no
fixed-end moments) and beam/shell self-weight needs the section A/t seam.

### LOAD-2 — `body_force=` docs are volume-only
`body_force` parameter docstrings are 3D/volume-only; no 2D example in the docs.

### PATTERN-1 — multiple time series, same type, different factors
No documented path for multiple time series of the same type with different factors.

### DOC-1 — contradictory docs on `g.loads` auto-emit (DONE, ADR 0051 BL-6)
RESOLVED. `guide_opensees.md` and the apegmsh-helper skill now agree:
loads are **opt-in** via `p.from_model(case)`; there is no `g.loads`
auto-emit. Stale "synthesized Plain" / "auto-emit" / "don't double-declare
(2×)" language removed from both.

### NODE-1 — user-defined bridge node verb
No verb to create a user-defined node on the bridge; the only non-mesh nodes today are `node_to_surface` phantom nodes, created implicitly by the constraint system. Reference node ndf must be **explicit** — silent fallback to global model ndf is wrong for mixed-ndf models.

### DOC-2 — namespace method docstrings
Namespace method docstrings (`ops.uniaxialMaterial.*`, `ops.nDMaterial.*`) are missing or one-liners; the underlying dataclasses are documented but the IDE never shows those.
