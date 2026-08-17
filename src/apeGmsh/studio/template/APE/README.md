# APE Project

This folder is the **APE Project** context for an apeGmsh Studio habitat:
*how we work*, *what this habitat remembers*, and *which skills / libraries
to open*.

**Scope of the template:** finite-element mechanics in the widest practical
sense — not one connection type. New habitats (nonlinear seismic, PBD,
SSI / ground, contact, RC, steel, mixed systems, …) copy this tree, clear
or rewrite `memory/`, and add their own `models/<id>/`.

This repo is the **template seed** itself — instructions stay generic; see
`steel-connection` for the first worked example (shear-connection model)
built from this seed.

## Layout

```text
APE/
├── README.md              ← you are here — session navigation
├── instructions/          ← cross-project playbook (template rules)
├── memory/                ← this habitat's persistent memory (grows)
├── skills/                ← skill references + harvest
└── libraries/             ← ape* library catalog + harvest
```

| Folder | Role | Grows? |
|--------|------|--------|
| **instructions/** | Rules for *all* APE Studio FEM habitats (session start, Socratic dialogue, MUST/SHOULD). Change when the *way of working* changes. | Slowly (template) |
| **memory/** | Persistent memory of *this* habitat: brief, assumptions, decisions. | Yes — every important session |
| **skills/** | Portable skill catalog (GitHub-canonical). Read when relevant. | When skills are added/localized |
| **libraries/** | ape* toolchain inventory and roles. | When libraries join the stack |

## Session start (agents and humans)

Read in this order unless the chat already has the same intake:

1. **`APE/README.md`** (this file) — orientation.
2. **`APE/instructions/how-we-work.md`** — Studio contract and hygiene.
3. **`APE/instructions/studio-mcp.md`** — verify live MCP `status.root`;
   `lookup` before grepping apeGmsh `src/`.
4. **`APE/instructions/socratic-geometry.md`** — shared picture of geometry
   *and* physics: show schemes, ask, confirm before build.
5. **`APE/instructions/progressive-modeling.md`** — oracle + stage metrics;
   escalate nonlinearity / contact / soil / seismic; apeSees bridge hygiene.
6. **`APE/memory/brief.md`** — units, intent, done-when for *this* habitat.
7. Skim **`APE/memory/decisions.md`** and **`APE/memory/assumptions.md`**.
8. Open **`APE/skills/catalog.md`** / **`APE/libraries/catalog.md`** only
   as needed for the domain (or their READMEs for how to navigate).

**Session end:** follow `APE/instructions/session-postmortem.md` →
`postmortem/sessions/` + `postmortem/backlog/open.md`.

Do **not** treat skill/library catalogs as already-in-context just because
they are listed.

## What belongs where

| Put it in… | Examples |
|------------|----------|
| `instructions/` | Soft folder contract, Socratic dialogue, progressive modeling, skills policy — **generic** |
| `memory/` | This habitat’s brief, assumptions, decision log, open questions |
| `skills/` | Harvested catalogs + harvest script |
| `libraries/` | Library roles + harvest status |
| `models/…` (repo root) | Idealization / geometry / analysis scripts and cases |
| `tools/` (repo root) | apeCAD / apeSketch launchers + human|agentic sketch archives |
| `postmortem/` (repo root) | End-of-session metrics, friction, living backlog |
| `references/` (repo root) | Optional PDFs / papers / notes for this model |
| `reports/` (repo root) | Authored narrative and promoted figures |
| `.apegmsh/` (repo root) | Generated Studio only — not memory |

## Refresh catalogs

```text
python APE/skills/harvest.py
python APE/libraries/harvest.py
```

## Soft manifest

Paths are listed in `ape.project.yaml` at the habitat root. `role: habitat`;
`scripts/init_habitat.py` fills in `name` and the first `models:` entry —
see `TEMPLATE.md` for how this seed was extracted.
