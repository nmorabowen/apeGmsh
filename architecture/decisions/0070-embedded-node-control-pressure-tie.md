# ADR 0070 — `EmbeddedNodeControl` pressure tie (`-pressure` / `-kp`)

**Status:** Accepted (2026-06-24). Second of the four "Ladruno constraints
coverage" clusters (see `plan_ladruno_constraints_coverage.md`); the A1
slice. Extends ADR 0068 P4 (the `enforce="penalty_al"` →
`LadrunoEmbeddedNode` tie) by exposing the element's **pressure (u-p)**
tie. No `Emitter` Protocol change — it rides the existing
`emitter.element(...)` path via `CouplingControl.emit_flags`.

## Context

`g.constraints.tie(..., enforce="penalty_al", control=...)` emits the fork
`LadrunoEmbeddedNode`. ADR 0068 wired the `CouplingControl` knobs
(`-k`/`-kr`/`-enforce`/`-bipenalty`/`-absolute`) and explicitly gated the
route to **translations only** — `_validate_tie_enforce` rejects the
ASD-style `rotational`/`pressure`/`stiffness_p` *Def* fields on
`penalty_al`, with the message "configure the penalty_al element via
`control=`". So the design intent is clear: penalty_al element features
live on the **control object**, not the ASD-style fields.

`OPS_LadrunoEmbeddedNode.cpp` accepts a family of beyond-translation
knobs: `-pressure [-kp]` (u-p pressure tie), `-rot [-kr -krAlpha]`
(rotation tie), and the `-normal`/`-orient`/`-corot`/`-matN`/`-matT*`
material interface. The audit (`OPS_LadrunoEmbeddedNode.cpp`) shows these
are **interdependent**: `-rot`/`-corot` error without host gradients
(`-dNdx`/`-gradXi`), and `-corot` errors without a `-mat*`. Only
`-pressure`/`-kp` is runtime-valid with the shape-only weights the
node-to-surface resolver already emits. So that is the slice that ships
now; the rest land together in a follow-up (resolver gradient emission +
uniaxial-material-tag translation).

## Decision

* New `EmbeddedNodeControl(CouplingControl)` in
  `_kernel/_coupling_control.py`: adds `pressure: bool = False`,
  `kp: float | None = None`; `emit_flags` appends `-pressure [-kp Kp]`
  after the base flags (the fork parser is order-independent). Validates
  `kp > 0` and `kp` requires `pressure`. Being a `CouplingControl`
  subclass, it is accepted as `control=` on every surface tie def
  (`tie`/`tied_contact`/`embedded`) with **no** signature change, and
  `_coupling_control_flags` dispatches `emit_flags` polymorphically — so
  `build.py` needs no change. A plain `CouplingControl` still works (it is
  the base); the base RBE2/RBE3 path is untouched.
* **Persistence** — the `cpl_*` / `sr_cpl_*` H5 lanes gain `cpl_pressure`
  (uint8) + `cpl_kp` (float64) and their per-slave mirrors (**neutral
  schema 2.18.0**, additive). The decoder reconstructs an
  `EmbeddedNodeControl` iff `cpl_pressure` is set, else the base
  `CouplingControl` — so a base control never gets spuriously promoted.
  Pre-2.18.0 files lack the columns (presence-probed) and decode the base
  control.

## Why a dedicated control (not the existing TieDef.pressure field)

`TieDef` already has `pressure`/`stiffness_p`, but those route to the
**ASD** `ASDEmbeddedNodeElement` penalty element (`-p`/`-KP`) and are
deliberately forbidden on `penalty_al`. Reusing them would muddy the
penalty-vs-penalty_al separation the validation enforces. A dedicated
control keeps the two element families' knobs apart and gives the
genuinely-new rotation/material follow-up a home (it will extend
`EmbeddedNodeControl`, not the shared `CouplingControl`).

## Deferred (one coupled follow-up)

`-rot`/`-kr`/`-krAlpha`, `-dNdx`/`-gradXi`, `-normal`/`-orient`/`-corot`,
`-matN`/`-matT1`/`-matT2`. These need (a) host-gradient emission from the
node-to-surface resolver and (b) uniaxial-material-handle → ops-tag
translation, and the fork parser forbids partial combinations — so they
ship as one unit, not piecemeal. Tracked in
`plan_ladruno_constraints_coverage.md`.

## Consequences

* New `EmbeddedNodeControl`; `pressure` on `penalty_al` ties is reachable
  for the first time. No change to existing emission or to the base
  `CouplingControl` round-trip (defaults off ⇒ byte-stable).
* Neutral schema 2.17.0 → 2.18.0 (additive minor; two-version reader
  window per ADR 0023). The deck-archival H5 deferral from ADR 0069 is
  unaffected (controls ride the FEMData snapshot, which is fully wired).
