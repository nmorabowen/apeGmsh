# Plan — Docs learnability overhaul

**Status:** proposed · 2026-05-30 · from a 7-lens read-only pedagogy diagnosis (workflow `wq2paw9qn`)
**Audience:** a structural/earthquake engineer who knows FEM + OpenSees but is new to apeGmsh (and maybe new to "FEM as code").

## Diagnosis (all 7 lenses converge)

The docs are **Explanation + Reference heavy, with no real Tutorial and no How-to layer.** In Diátaxis terms:

| Mode | State |
|---|---|
| **Reference** (`docs/api/`) | strong, mode-pure — keep |
| **Explanation** | abundant but **over-prominent and impure** — blended into `first_steps.md` + 18 guides |
| **Tutorial** | **effectively absent as a labeled track** — the notebook gallery is the closest, but it's gapped, non-executing, and mislabeled |
| **How-to** | **entirely missing** — no task-recipe layer anywhere in the nav |

**Time-to-first-success today: 60–120 min.** There is no named "10-minute first model."

### The single biggest liability: `first_steps.md` (2,986 lines)
The front-door card says *"I'm new — orient me. The right place to start."* and points at a **15-lesson textbook** that doesn't reach a runnable model until §15.9 (line ~2,872), and whose own Epilogue admits the worked end-to-end example is still a TODO. It is Tutorial+Explanation+Reference fused under a Tutorial label.

### The other blockers
1. **Install is at the *bottom* of the home page** and absent from both "start here" docs → cold readers stall at `ModuleNotFoundError`.
2. **No How-to layer.** ~20 canonical tasks have no task-titled home. Worst case: *"fix a support"* lives in `guide_loads.md §11` titled *"Face-prescribed displacements."* `g.save`/`from_h5` and `g.compose` (shipped headline features) have no findable page.
3. **Broken learning ladder.** A full 19-slot/5-tier curriculum exists in `examples/CURRICULUM.md`, but only 8 rungs are built — and the built notebooks carry **dangling `Prerequisite: 10, 11` pointers to notebooks that don't exist.** The index punts to "60+ examples on GitHub."
4. **Notebooks don't execute** (`execute:false`) → rendered docs show code with **no plots, no verification numbers** → no "seeing it work."
5. **The gallery teaches the wrong half of the library.** All 8 notebooks build the OpenSees model with **raw `ops.*`** and ignore the typed `apeSees` bridge + loads/sections/masses composites — the headline product value has **zero examples**.
6. **No human "mental model" page.** The cleanest concept scaffold in the repo (4 numbered concepts, ~40 lines) exists **only in the AI-facing `skills/apegmsh/SKILL.md`**.
7. **Stale/contradictory facts** a learner trips on: "15 constraint kinds" (`first_steps`) vs "12 types" (`architecture.md`); and `first_steps.md`/`architecture.md` **still say the bridge "does NOT ingest / MP emission DEFERRED"** — the exact stale claim fixed in the skill + API docs in PR #443, never propagated to the learning docs.

## What's working — PRESERVE
- The home **hero + card grid** is a good front door (needs ordering/primacy, not replacement).
- The **conceptual prose** in `first_steps` is genuinely excellent (Part=template / Instance=placement, the three naming layers, Abaqus analogies). The problem is packaging, not content.
- The **8 notebooks are well-narrated individually** (≈50/50 prose/code, problem statement + sketch + closed-form check + recap) — far above a code-dump gallery. The failure is at the *gallery* level.
- **API reference** (autodoc) is strong.

## Target information architecture (Diátaxis spine)

```
Home  →  Tutorials  →  How-to (Recipes)  →  Concepts (Explanation)  →  API Reference  →  Internals/Design
```

- **Tutorials** (NEW section, first after Home): 3–4 hand-held, guaranteed-success journeys, built from the verified notebooks. Hello → Cantilever → (Parts/Assembly) → (Read & plot results).
- **How-to / Recipes** (NEW): ~20 task-titled pages, *harvested from existing guide snippets* (not written from scratch).
- **Concepts**: the de-blended guides + a short **Mental-model page** (adapted from `SKILL.md`). `first_steps` demoted here.
- **API Reference**: keep as-is.
- **Internals/Design**: demote `architecture/` + `Planning` to last (or a maintainer-only set).

## Phased action plan (impact / effort)

### Tier 0 — quick wins (days; cheap, high-leverage)
1. **Mint the "Your first model in 10 minutes" tutorial** — install → box → global size → `generate` → `fix` → point load → run → **one printed verified number** → `show_web`. Single path, **zero forks**, typed `apeSees` bridge. Make it the dominant #1 card. *(high / M)*
2. **Extract a human "Core concepts / Mental model" page** (~1–2 screens) from `SKILL.md:86-129`; make it the "First steps — orient me" target. *(high / S)*
3. **Hoist install** into/above the hero, and repeat the one-liner (`pip install apeGmsh[all]` + "needs openseespy") as cell 0 of the tutorial. *(high / S)*
4. **Re-order + relabel the front-door cards** to mirror Tutorial → How-to → Concepts → Reference; rename "First steps" card to "Mental model & concepts." *(high / S)*
5. **Propagate the PR #443 fact fixes** into `first_steps.md` + `architecture.md` (MP emission is automatic, not deferred; reconcile "15 vs 12 constraint kinds"). *(medium / S — correctness)*

### Tier 1 — mint the missing quadrants (1–2 weeks)
6. **Add the How-to / Recipes nav section**: an index + ~20 task pages harvested from guides. Fix the homeless **"Supports & boundary conditions"** page first; add **"Save & reload"** and **"Compose modules."** *(high / M)*
7. **Split `first_steps.md`**: keep Lessons 1–3 as the short Concepts/mental-model page; disperse Lessons 4–15 into the existing per-subsystem `guide_*.md`. *(high / L)*
8. **De-blend the guides**: each `guide_*.md` leads with a "Tasks on this page" jump-list, verb-titled recipe headings, and strips the "Grounded in the current source: <module paths>" manifests. *(medium / M)*

### Tier 2 — fix the example ladder (1–2 weeks)
9. **Surface `CURRICULUM.md` as the spine** on `examples/index.md` (ordered table: tier, #, title, prereq, status), and **kill every dangling prerequisite pointer** — build/stub the cited rungs (03, 06, 10, 11, 18) or rewrite the headers. *(high / M–L)*
10. **Add a true rung-0 hello notebook** (box → mesh → fix → point-load → one displacement, **no manual tributary traction**). *(high / M)*
11. **Add the missing high-level examples**: typed `apeSees` bridge end-to-end; CAD STEP-import → loads composite → solve. *(high / M)*
12. **Execute the first 2–3 notebooks** (commit outputs or `execute:true`) so rendered docs show success. *(high / M)*
13. **Defer the `capture` vs `emit_recorders` fork** — one strategy on the first-success path; introduce the second as its own later rung. *(medium / S)*

## Decisions needed (see chat)
- Sequencing/scope: Tier 0 quick wins first, or commit to the full IA restructure up front?
- Example ladder: build the missing notebooks, or stub + redirect + fix dangling pointers?
