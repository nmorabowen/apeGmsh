# Progressive modeling — oracles, metrics, escalated complexity

Template rule for APE Studio habitats: **grow the finite-element model in
stages**, with a Socratic check at each stage. Applies to idealization,
geometry (apeGmsh), the **apeSees / OpenSees bridge**, constitutive wiring,
contact, soil–structure interaction, seismic procedures, and design-linked
workflows.

This is not an absolute ladder. The human and agent use **discernment**:
skip or merge stages when prior art, a trusted twin model, or clear
confidence justifies it. Prefer getting a **correct** model ahead of
token-minimal shortcuts — a wrong full-complexity deck usually costs more
tokens to repair than a short progressive path.

---

## Why escalate

| Prefer progressive growth when… | May jump complexity when… |
|----------------------------------|---------------------------|
| New topology, new interface, new constitutive class | Same habitat pattern already validated |
| Numerics are stiff (contact, soil, softening) | Trusted oracle + twin case exists |
| Acceptance metrics are not yet defined | Human explicitly accepts the risk |
| Agent or human is unsure | “Proceed complex” is an explicit gate |

**Token heuristic:** optimizing tokens by skipping validation often *increases*
total tokens (debug loops). Default to escalate; accelerate only with a named
reason recorded in `APE/memory/decisions.md`.

---

## Oracle first (always, even if light)

Before (or with) the first runnable deck, define an **oracle**: what would
make the next stage “believable”?

Oracles may be:

- **First-principles / generalized idealizations** — order-of-magnitude
  stiffness, frequency, capacity, contact onset displacement, elastic drift,
  etc.
- **External** — papers, codes, hand calcs, design guides, physical tests,
  prior FEM, documented twin models (store copies or links under optional
  `references/` when useful for this habitat).
- **Qualitative** — expected monotonic shape of a curve, which region yields
  first, whether uplift or gap closure should occur.

Write the oracle and the **metric** (number, curve feature, inequality) into
`APE/memory/brief.md` or `assumptions.md`. No metric ⇒ no claim of validation.

---

## Escalation ladder (menu, not dogma)

Use as a **conversation checklist** with the human. Order is typical, not
mandatory. Omit inapplicable rungs.

```text
  0  Oracle + success metrics (what “good” means)
  1  Geometry / topology sanity (mesh, labels, BCs) — often elastic smoke
  2  Linear (or mildly nonlinear) structural response vs oracle
  3  Add material nonlinearity (structure) — margins still coherent?
  4  Add geometric nonlinearity / large-disp if relevant
  5  Add interfaces (contact, ties, gaps) — onset and penetration/force metrics
  6  Add foundation / soil (simple constitutive first)
  7  Enrich soil / interface constitutives (e.g. simple → advanced sand/clay)
  8  Add dynamic / seismic procedure, records, PBD demands
  9  Production controls, reports, design checks
```

**Illustrative SSI path** (example only): elastic structure → check drifts /
frequencies → structural nonlinearity → (optional) geometric NL → simple
soil (e.g. Mohr–Coulomb / Parker-class idealizations) → advanced sand models
→ full seismic. Replace names with whatever the habitat needs; the pattern
is **simple → coherent → richer**.

**Contact path** (example only): elastic bodies + soft interface → onset
metric → tune penalty/constraint → then plasticity / cyclic / mortar as needed.

---

## Metrics of functionality (per stage)

Each stage should declare, before the run:

1. **What we will look at** (EDP, reaction, penetration, period, energy, …).  
2. **Band of sanity** (oracle inequality or qualitative shape).  
3. **Fail action** (ask human; do not silently add the next nonlinearity).

Promote lasting curves/stills into `reports/` when a stage is accepted.

---

## apeGmsh → apeSees bridge

Treat the bridge as part of the same escalation:

- Confirm **labels, constraints, DOFs, and units** on a minimal deck before
  stacking constitutives and contact.
- Prefer a **thin elastic (or linear) bridge case** that finishes and matches
  the oracle before enabling plasticity, contact elements, or soil materials
  in the OpenSees domain.
- When the bridge fails, **localize**: geometry/export vs element vs material
  vs integrator — do not add complexity to “try to make it converge.”

Socratic prompt pattern:

> “Stage *n* metric is … Oracle says … I propose next complexity *X*.
> Confirm, change metric, or skip with reason?”

---

## Relation to the Socratic dialogue

This file is the **growth policy**; `socratic-geometry.md` is the **confirm
loop** (show → ask → wait). Together:

1. Agree oracle + metric (Socratic).  
2. Build the smallest model that can test them.  
3. Accept or repair.  
4. Propose the next complexity rung (Socratic).  
5. Repeat until the habitat’s done-when in `APE/memory/brief.md` is met.

---

## Anti-patterns

- Full nonlinearity + contact + soil + seismic on day one with no oracle.  
- Changing three physics knobs at once after a failed Newton.  
- Declaring “validated” without a metric.  
- Skipping stages *silently* (always name the skip and why).  
- Burning tokens on a maximal deck when a two-hour elastic oracle would have
  caught the axis / BC / unit error.
