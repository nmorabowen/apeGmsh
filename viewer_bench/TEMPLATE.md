# Provenance

Extracted from `steel-connection@0962863` (full sha
`09628632b3d24e01884de40d006cf7f73cfbaff8`, "APE Studio habitat restructure:
template seed + shear-connection worked example", 2026-08-16) on 2026-08-17.

`APE/instructions/`, `scripts/`, `postmortem/`, and `reports/` structure
were copied from that commit with the `shear_connection` specimen model,
specimen reports, and specimen session archives left behind; five
instruction files (`how-we-work.md`, `socratic-geometry.md`, `reporting.md`,
`analysis-profiling.md`, `continuous-improvement.md`) had their
self-references inverted (this repo is now the seed; `steel-connection` is
the worked example). `APE/memory/` was replaced with fill-in-prompt stubs.
`APE/skills/` and `APE/libraries/` catalogs were emptied — any personal
skill/library roster is the template author's own and is documented as
optional, never required (see `APE/skills/README.md` /
`APE/libraries/README.md`).

## Sync policy

Pattern promotion flows **template-ward**: when a workflow gate proves
itself in a habitat built from this seed, promote it into
`APE/instructions/` here (not just that habitat's memory) — see
`APE/instructions/continuous-improvement.md`. Habitats built from this seed
do not push their model-specific facts upstream; only the *way of working*
travels back.
