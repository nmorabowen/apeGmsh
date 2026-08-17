# Session postmortem — closing ritual

Run this when the human ends the session. Keep it **generic** across FEM
habitats. Write outputs under `postmortem/sessions/YYYY-MM-DD_<slug>/` and
update `postmortem/backlog/open.md`.

Templates: `postmortem/templates/01_metrics.md`, `02_friction.md`.

---

## Intent

End-of-session **agentic accounting** (sometimes called the agentic
apology): an honest evaluation of how the session went, what it cost, where
the workflow rubbed, and what must be built next so the next session is
cheaper and clearer — in libraries and in habitat operations, not only in
chat habits.

---

## Procedure (agent)

1. **Confirm close** — restate that this is a postmortem (no more model
   builds unless the human reopens). Prefer running `scripts/finish.ps1`
   first (profilers + session scaffold), then complete friction triage.
2. **Slug** — short kebab name (`contact-campaign`, `ssi-elastic-oracle`, …).
3. **Create** `postmortem/sessions/YYYY-MM-DD_<slug>/`.
4. **Fill `01_metrics.md`** from:
   - any Studio / ADR profiler JSON under `postmortem/sessions/…`
     (`profile_studio.json`), habitat `.apegmsh/mcp_calls.jsonl` when the
     MCP server is `apegmsh-studio-habitat`;
   - transcript evidence (skills Read, habitat MCP, src search/read,
     shell, harvests);
   - honest **unknowns** when counts are unavailable (do not invent exact
     token bills). Prefer `lookup` metrics rising vs `src_search`.
5. **Fill `02_friction.md`** — Socratic with the human if needed: top
   frictions, false paths, missing Studio/bridge affordances, Socratic or
   progressive-modeling gaps.
6. **Triage into `backlog/open.md`** — each item needs:
   - **Target** (see `continuous-improvement.md` tags: `skills`,
     `opensees`, `apegmsh`, `apesees`, `studio`, `observability`,
     `tools-bridges`, `ape-instructions`, `memory`, `cases-oracles`,
     `reports`, `repro`, `evals-ci`, `continuity`, `human-loop`,
     `template`, `upstream`, …);
   - **Symptom** → **proposed change** → **priority** (P0–P3) →
     **next-postmortem proof**;
   - **Evidence** (session folder link).
7. **Optional promote** — if an item belongs upstream, note “open issue /
   ADR in `<github>/…`” rather than only leaving it in this habitat.
8. **Hand off** — point the human to the session folder + backlog delta;
   stop building unless asked.

---

## Metrics doc (what “good enough” means)

Capture at least:

| Signal | Why |
|--------|-----|
| Approximate tokens / chars by class (skill, src search, src read, habitat MCP, other) | Cost shape |
| Habitat MCP call count (0 is itself a finding) | Whether Studio was used as designed |
| Skill reads vs source scavenger-hunt rate | Door hygiene |
| Cases / figures / memory updates produced | Outcome density |
| Socratic / progressive gates kept or skipped | Process fidelity |
| FEM `profile.h5` captured on worth-it runs? | Analysis-time hotspots (see `analysis-profiling.md`) |

If a machine profile exists, attach `profile_studio.json` (and
`profile_promote.json` if promote was run) beside `01_metrics.md`.

---

## Friction doc → implementation (more than a document)

`02_friction.md` is the **diagnosis**. `backlog/open.md` is the **work
queue**. Full improvement map (skills, libraries, Studio, observability,
oracles, continuity, …): `continuous-improvement.md`.

Prefer changes that remove repeated pain. Skills and library code are
necessary but not sufficient — also triage process, habitat, and human-loop
gaps.

| Target | Examples of improvements |
|--------|--------------------------|
| **skills** | Triggers, anti-examples, stage-linked guidance |
| **opensees / apegmsh / apesees / design-libs** | APIs, diagnostics, staged NL, Results |
| **studio / observability** | Connected MCP, profiler on close, index vs grep |
| **tools-bridges** | apeCAD/apeSketch archives, confirm schemes |
| **ape-instructions / memory** | Gates agents follow; oracles in brief |
| **cases-oracles / reports** | Golden cases; figure/report path hygiene |
| **repro / agent-runtime / evals-ci** | One-command env; rules/MCP; smoke + process evals |
| **continuity / human-loop / template** | Start from backlog; question craft; seed new habitats |
| **upstream** | ADR/issue in the owning repo |

Discernment: not every friction becomes code. Some become a one-line
instruction fix; some become an upstream issue; some are accepted cost.
Each accepted row should name a **next-postmortem proof** (metric or
behavior that shows the fix worked).

---

## Relation to other playbooks

| Playbook | Role |
|----------|------|
| `socratic-geometry.md` | Confirm intent *during* the session |
| `progressive-modeling.md` | Escalate complexity *during* the session |
| `continuous-improvement.md` | Full map of *what* postmortems can improve |
| **This file** | Evaluate and improve *after* the session |

---

## Anti-patterns

- Closing with only “thanks” and no artifacts.
- Fake precision on tokens when no profiler ran.
- Friction essays with no backlog row / no target surface.
- Filing FEM physics bugs only in chat memory.
- Expanding scope into a new build during postmortem without asking.
