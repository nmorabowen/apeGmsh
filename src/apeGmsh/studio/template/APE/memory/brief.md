# Project brief — __HABITAT_NAME__

**Habitat:** One-line description of what this habitat models.
**Domain note:** Instructions under `APE/instructions/` are generic FEM
mechanics; *this file* is only this habitat's memory.
**Units:** State the unit system (default N, mm unless overridden).
**Status:** Where this habitat stands right now.

## Intent

What question must the model answer? State the mechanics branch, the
bodies/system involved, and what "done" looks like in one paragraph.

## Members / layout

| Role | Section | Length | Notes |
|------|---------|--------|--------|
| | | | |

Gravity / load sense and work points: state axes and where loads attach.

## Done when

1. First acceptance metric.
2. Second acceptance metric.
3. Cases recorded under `models/<model_id>/cases/` with `run.json`.
4. Narrative + figures under `reports/` (not only `.apegmsh/visors/`).

## Open / next (engineering)

- Next engineering question or risk to resolve.

## Agent intake (copy into new chats)

One paragraph: units, geometry summary, boundary conditions, and the
default parameters an agent needs to pick up this habitat without
re-reading everything. Scripts in `models/<model_id>/src/`. Habitat root =
repo root.

**APE door:** start at `APE/README.md`, then `APE/instructions/how-we-work.md`,
then this brief.

**Skills:** `APE/skills/catalog.md` + `catalog.json` (GitHub-canonical).
Refresh: `python APE/skills/harvest.py`.

**Libraries:** `APE/libraries/catalog.md` — bridges, apeGmsh, design,
access points. Refresh: `python APE/libraries/harvest.py`.
