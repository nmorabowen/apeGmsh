# Socratic model dialogue — human ↔ agent

Template rule for **every** APE Studio habitat: finite-element mechanics in
the widest sense (linear or nonlinear solids/structures, seismic and
performance-based analysis, soil–structure interaction, contact and
interfaces, reinforced concrete and steel, constitutive and ground
representation, multiphase or coupled problems as they appear).

This repo is the **template seed**. Copy the APE tree; put *this project’s*
facts in `APE/memory/`, not in these instructions. `steel-connection` is the
canonical worked example.

**Goal:** a shared understanding of geometry *and* physics *before*
construction in `models/…/src`. Prose alone is not the model language.

---

## Media that carry the conversation

| Tool | Who leads | Role |
|------|-----------|------|
| **apeSketch** | Human | Hand ink via `tools/apeSketch/open_interface.py`; archive under `human_sketches/` |
| **apeCAD** | Agent or human | Spatial scratchpad via `tools/apeCAD/open_interface.py`; archive under role folders |
| **Studio / figures** | Agent | Stills, deformed shapes, contours, history curves — after a draft exists |
| **Chat** | Both | Questions, alternatives (A/B), and stop / go gates |
| **Skills / libraries** | Agent opens on demand | Domain depth (contact, RC, quake, AISC, …) — listing ≠ loaded |

apeCAD and apeSketch are **intent bridges**, not solvers. Habitat archives:

- `tools/apeCAD/{human,agentic}_sketches/`
- `tools/apeSketch/{human,agentic}_sketches/`

apeGmsh (and design or ground-motion libraries) act only after the relevant
checkpoint passes.

---

## Hard rule (gate)

**Do not author geometry scripts, meshes, constitutive wiring, contact,
boundaries, or analysis decks until the human confirms the current
checkpoint** (or explicitly says “proceed without sketch / with my prose”).

If anything is missing, ambiguous, or conflicting: **ask first**. Prefer
one tight cluster of questions over a monologue. Never invent:

- orientation, topology, or which body/region is which  
- interface roles (contact master/slave, tied, soil–structure face, …)  
- constitutive class or material parameters the human did not give  
- analysis procedure (static, cyclic, NLRHA, explicit, …) or “done when”

---

## Dialogue loop (repeat per checkpoint)

```text
  Human intent (words / ink / CAD / prior memory)
           │
           ▼
  Agent restates + shows a scheme or decision table
  (apeCAD / sketch / labeled diagram / option A vs B)
           │
           ▼
  Human: confirm / correct / redraw / answer
           │
     ┌─────┴─────┐
     ▼           ▼
  Confirmed    Still unclear → ask again (Socratic)
     │
     ▼
  Next checkpoint → eventually build under models/
```

**Socratic means:** the agent surfaces *what it thinks it knows*, *what it
does not*, and *what would change the model or the results* — then waits.
The human owns **intent and acceptance**; the agent owns **FEM consequences
and numerical hygiene** after confirmation.

---

## Checkpoints (full model manufacture)

Use the ladder below as a **menu**, not a steel-only script. Skip stages
that do not apply; do not skip ambiguity. Persist confirms in `APE/memory/`.

| # | Checkpoint | Agent must show / ask | Typical before… |
|---|------------|------------------------|-----------------|
| 1 | **Scope & success** | What mechanics branch? What question must the model answer? Done-when (metrics, plots, limits) | Any modeling |
| 2 | **Units & frame** | Units; axes; gravity / positive load / ground-motion directions | Coordinates, spectra |
| 3 | **Idealization** | Continuum / frame / shell / soil domain / mixed; 2D vs 3D; symmetries | Geometry scripts |
| 4 | **Bodies & topology** | Parts, regions, excavations, members; roles (structure, fill, foundation, …) | Solids / frames |
| 5 | **Work points & couplings** | Joints, RBEs, constraints, embedded regions | Multi-point links |
| 6 | **Interfaces** | Contact, ties, gaps, fretting, soil–structure, fluid–structure — faces and roles | Interface elements |
| 7 | **Materials & constitutive** | Elastic / plastic / damage / concrete / steel / soil; parameters; regularization needs | Material blocks |
| 8 | **Boundary & excitation** | Supports; pressure/disp/velocity; gravity; static pattern; ground motion / apeQuake records | Loads / IM / TH |
| 9 | **Procedure** | Linear / nonlinear static; cyclic; eigenvalue; NLRHA; explicit; staged construction | Integrator / steps |
| 10 | **Discretization** | Element family, size, order; conformal vs non-matching; hourglass / locking risks | Mesh |
| 11 | **Numerics & controls** | Tolerances, line search, damping, contact penalty/Lagrange scale, time step | Production runs |
| 12 | **Acceptance & reporting** | What curves/fields prove the question; figures to promote into `reports/` | Case archive |
| 13 | **Build** | Scripts under `models/<id>/src/`; cases with `run.json` | Studio / batch |

Growth policy (oracles, escalated nonlinearity, apeSees bridge): see
`progressive-modeling.md`. Checkpoints above agree *what* to build; that
file agrees *in what order* and *how to know a stage worked*.

Examples of habitats that use the same ladder: connection contact; RC wall or
frame; nonlinear seismic building; performance-based evaluation; pile–soil
or DRM/SSI; steel design check fed by FEM; mixed continuum–frame systems.

---

## What the agent should produce to confirm

Prefer **visual or tabular + short labels** over paragraphs:

1. **apeCAD `Document`** — labeled points, axes, envelopes, interface notes.  
2. **apeSketch** — ask the human to ink the scheme when words fail.  
3. **Decision table** — materials, procedures, interface types (tentative).  
4. **Fallback scheme** — labeled orthographic sketch, mermaid, or minimal plot
   if CAD/ink clients are down.  
5. **Question block** — ~3–5 items, each with a *tentative* default marked as such.

Studio deformed/contour stills confirm **what was coded**, not intent, until
the human has accepted the checkpoint that led to that code.

---

## Phrases that must trigger a pause

- Vague nouns: “the beam”, “the soil”, “fixed”, “nonlinear”, “seismic”, “contact”
- Unnamed interfaces or missing master/slave (or equivalent roles)
- Constitutive model named without parameters or regularization intent
- Load or ground-motion direction not tied to axes
- “Same as last time” without a pointer into `APE/memory/` or a prior case
- Performance objective stated without engineering demand parameter / limit

Agent pattern: restate → show scheme or A/B → one confirming question → **wait**.

---

## Completeness

The loop covers the **whole** path: idealization → geometry → physics →
procedure → mesh → run → acceptance figures. When the human corrects
mid-build, **return to the failed checkpoint**, update memory, then rebuild.
Do not silently patch a wrong frame or wrong constitutive story.

---

## Memory handoff

| Persist in… | Content |
|-------------|---------|
| `APE/memory/brief.md` | Scope, idealization, done-when for *this* habitat |
| `APE/memory/assumptions.md` | Accepted defaults |
| `APE/memory/decisions.md` | Dated why (orientation, constitutive, kn, record set, …) |
| `reports/` | Authored narrative and promoted figures |

---

## Anti-patterns

- Building from a long chat with no scheme or checkpoint list.  
- Treating solver output as proof of intent.  
- Zero questions (assuming) or twenty questions (flooding).  
- Using apeCAD/apeSketch as a second FEM preprocessor.  
- Copying a worked example’s case details (e.g. steel-connection) into a
  new habitat’s instructions instead of into that habitat’s `memory/`.
