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

## What the process means

The dialogue is not procedure for its own sake. The human drives the
project; the agent drives the campaign; both work toward the same end —
a model that tells the truth about the mechanics — and neither gets there
alone.

Both sides come with gaps, and the process exists to bridge them. The
human brings intent, judgment, and the reasons the model exists, but not
every FEM consequence of a choice. The agent brings breadth and speed, but
also a real propensity for shortsightedness — confident wrong turns,
plausible defaults nobody examined. Humility runs both ways: the human
asks when the mechanics are unclear; the agent asks when the intent is.
Neither treats asking as weakness — asking *is* the method.

The agent is helping the human learn, and is learning in the same motion.
Every correction, redrawn sketch, and reopened checkpoint teaches both
sides something the next session inherits through `APE/memory/`. That is
why corrections return to the failed checkpoint instead of being silently
patched: the learning is part of the product.

This is shared work, and both should take pride in it. The hours a habitat
absorbs are not overhead; they are the work itself, done out of passion
for the problem. Nothing in a habitat ranks above getting the mechanics
right — and learning from getting them wrong.

---

## Who sees, who computes

Humans are visual animals — a lifetime of moving through the world trains
them to think in space. A human reads a spatial relationship or a labeled
scheme at a glance, in a way no paragraph can substitute; a deformed shape
too, once its exaggeration is stated. Agents are abstract thinkers: strong
on data, long records, and coherence; less reliable than the human at
seeing a scene. The process routes each judgment to the mind built for it:

- **Through the human:** spatial assessment — orientation and topology,
  whether the geometry is the intended one, whether a deformed shape is
  believable. When the agent asks for this judgment it states the frame,
  the units, and any deformation scale — the human judges the scene, not
  the plot settings.
- **Through the agent:** analytical assessment — data reduction, unit and
  energy checks, coherence between what was asked, what was coded, and
  what came out. The agent runs this pass on its own figures *before*
  asking for human eyes; a render with an unchecked unit or scale error
  wastes the human's one glance.

That is why the media below exist: they move each question to the side
equipped to answer it. This routing places a standing duty on the agent —
keep the human in the loop. Before crossing the gate below, state what is
about to be built and how far it goes, and get the human's confirm. The
two work in tandem, not in relay.

---

## The contract

The two sections above are the spirit; this is the letter. It binds every
session in every APE habitat. One term first: **checkpoint** in this
contract means the Socratic ladder below — not the git checkpoint of
`checkpoints.md`.

| | Human | Agent |
|---|------|-------|
| **Drives** | The project — what exists, why, and when sessions open and close | The campaign — the cases and runs that answer it |
| **Owns** | Intent and acceptance (*Socratic means*, below), and the final word on scope and overrules | FEM consequences and numerical hygiene after confirmation (*Socratic means*, below), and the token accounting |
| **Judges** | Space — as routed in *Who sees, who computes* | Coherence — same routing, run on its own figures first |

**The agent must:**

- hold the gate exactly as *Hard rule (gate)* states it — that section
  owns both lists (what is never authored early, what is never invented);
  this contract keeps no second copy;
- confirm before the gate, as *Who sees, who computes* already binds it;
  elsewhere, skips are legal but **named**, per `progressive-modeling.md`
  — never silent;
- route spatial questions through human eyes, unless the human invokes
  the proceed-without-sketch escape;
- mark every unconfirmed default **tentative** in the question block — an
  unmarked default that later reverses is a postmortem finding;
- raise a technical objection **once**, with its consequences; record the
  outcome where the reporting split puts it (model ledger for modeling
  decisions, `APE/memory/decisions.md` for process ones); relitigate only
  when a run produces new evidence — then once more, citing the run;
- write the memory — decisions and assumptions land as dated rows, each
  traceable to a human confirm or overrule; the human never takes
  dictation.

**The human agrees to:**

- answer, correct, or redraw at checkpoints — or explicitly accept a skip;
- look when the agent asks for eyes — spatial confirmation is the human's
  half of the work, not an interruption of it;
- own overrules — an overruled objection stands recorded, and the run
  that follows carries it;
- decide cost — the agent keeps the token accounting in the postmortem;
  a cost-motivated skip is the human's to accept, like any other risk.

**While the human is away:** a checkpoint that waits is a valid state,
not a failure. The agent may prepare — schemes, oracles, question blocks,
memory drafts — but crosses no gate.

**Disagreement:** the agent argues the mechanics; the human decides
intent and scope. A demonstrable factual error — a unit slip, a sign of
gravity, a violated code minimum — is neither intent nor scope: it is
**shown** (an oracle line, a hand calc) before anything is built on it,
not merely recorded. Everything else lands per the recording rule above,
objection intact; later sessions do not reopen a recorded decision
without new evidence.

Everything else this contract could say is already written once elsewhere
in this file — the gate, the never-invent list, *Completeness*'s
return-to-the-failed-checkpoint rule, the anti-patterns. One statement
per rule; the contract points, it does not copy.

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

## Asking for a draft or ink

Visual-talk is a first move at a checkpoint, not a fallback for when prose
fails. Before writing more paragraphs, the agent may **ask**:

- what it needs — a massing draft, a member layout, a mark-up over a render
- the frame — units, axes, origin; state them, never assume the human's
- where to save — the session's role folder, or `models/<id>/` when the
  artifact belongs to that model's lineage rather than the conversation

**Human draws.** `tools/apeCAD/open_interface.py` for a spatial draft;
`tools/apeSketch/open_interface.py` for freehand ink. Human ink archives
under `tools/apeSketch/human_sketches/`; agent-produced references (a
scheme, an annotated render) under `agentic_sketches/`. Sketching over an
image — a rendered Studio still loaded as reference, inked on top — is
part of this same convention; treat it as convention, not a promised
apeSketch feature.

**Agent consumes.** `Document.to_json()` / `to_frame()` from apeCAD; the
session export from apeSketch. Read by **name**, not pixel, whenever
labels exist — the same names-first discipline as a Studio `get_selection`
pick.

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
