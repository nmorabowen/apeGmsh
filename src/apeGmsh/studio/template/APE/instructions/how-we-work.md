# How we work — APE Studio habitats

Cross-project playbook for **APE** habitats: finite-element **mechanics** in
wide generality (structures and continua; linear and nonlinear; static,
cyclic, and seismic; performance-based workflows; contact and interfaces;
reinforced concrete and steel; soil and ground representation; SSI and
related coupled problems).

Project-specific facts live only under `APE/memory/`. This repository is the
**template seed** — copy the APE tree for new habitats; replace memory and
`models/`, keep instructions unless the *way of working* changes. The
`steel-connection` repository is the canonical worked example built from
this seed.

## Soft folder contract (habitat root)

```text
APE/           # playbook + memory + skills + libraries doors
tools/         # apeCAD / apeSketch launchers + human|agentic sketch archives
models/        # script-owned idealization → geometry → analysis; cases per run
reports/       # technical_briefs / model_ledgers / model_reports / figures
postmortem/    # end-of-session metrics + friction backlog
references/    # optional PDFs / papers / notes for this model (MAY)
scripts/       # start.ps1 / finish.ps1 — template check, MCP probe, postmortem
.apegmsh/      # Studio generated only (gitignore)
ape.project.yaml
```

- **MUST:** `APEGMSH_STUDIO_ROOT` = habitat root; cases write `run.json` +
  `deck/` + `results/` + `logs/`; model scripts live in `models/<id>/src/`.
- **SHOULD:** run `scripts/start.ps1` at session open and `scripts/finish.ps1`
  at close; read `APE/memory/brief.md` before building; run the Socratic
  dialogue (`socratic-geometry.md`) for unclear geometry or physics;
  open intent UIs via `tools/apeCAD` / `tools/apeSketch`; promote lasting
  figures into `reports/`; on session close run
  `session-postmortem.md` into `postmortem/`. Follow locked
  `reporting.md` for briefs / ledgers / stage model reports.
- **MAY:** add `calcs/`, `drawings/`, `docs/`, extra models under `models/`;
  populate `references/` when external literature helps this model.

## Session hygiene

1. Start from `scripts/start.ps1` (template + mcp.json root), then
   `APE/README.md` → `instructions/how-we-work.md` →
   `instructions/studio-mcp.md` (verify live `status.root`) →
   `memory/brief.md`.
2. **Modeling is Socratic and progressive** —
   `instructions/socratic-geometry.md` (confirm intent) and
   `instructions/progressive-modeling.md` (oracle → metrics → escalate
   complexity, including the apeSees bridge). Prefer apeSketch / apeCAD
   over paragraphs. Ask when unsure; do not stack nonlinearity, contact,
   and soil in one unvalidated jump unless the human accepts that skip.
3. **Studio MCP:** use `apegmsh-studio-habitat`; **`lookup` before grepping
   apeGmsh `src/`** (ADR 0096). See `studio-mcp.md`.
4. Skills and libraries are **doors**: open catalogs when the domain needs
   them (contact, RC, quake, design, FEM theory, …); listing ≠ loaded.
5. Engineering skills are **GitHub-canonical** under `<github>/`
   (`~/Documents/Github/`). `~/.claude/skills` is legacy if a GitHub copy exists.
6. Do not invent memory: append decisions and assumptions when they stick.
7. Authored reports stay in `reports/`; `.apegmsh/` is not the archive.
8. **Session close** — run `scripts/finish.ps1` (profilers + postmortem
   scaffold), then complete `session-postmortem.md` friction triage into
   `postmortem/backlog/open.md`.

## Template stack (doors, not prescriptions)

Recommended skills and libraries for new habitats are curated in:

- `APE/skills/catalog.md`
- `APE/libraries/catalog.md`

Trim or extend per habitat (e.g. apeQuake for ground motions, apeConcrete /
apeSteel for design checks). The **doors and harvest scripts** stay.

## Units default

Unless `APE/memory/brief.md` says otherwise: **N · mm · tonne · s**
(via baseUnits when used). Always confirm units at checkpoint 2 of the
Socratic ladder for a new habitat.
