# ape-studio-template — APE Studio habitat template

An **APE Studio habitat** is an apeGmsh-Studio-centric FEM project: a durable
playbook under `APE/` (how we work, this habitat's memory, skill/library
doors), session lifecycle scripts (`scripts/start.ps1` / `finish.ps1`), a
soft-contract checker (`scripts/check_template.py`), and one model lineage
under `models/<id>/` with `src/` (script-owned geometry/analysis) and
`cases/` (one folder per run). It scales from a single elastic oracle to
nonlinear, contact, seismic, or SSI work without changing the folder
contract.

## Usage

1. **Use this template** on GitHub (or clone this repo).
2. Run the one-shot personalizer from the new repo's root:

   ```text
   python scripts/init_habitat.py --name <habitat-name> --model <model_id>
   ```

3. Open `APE/README.md` and follow session start.

## Seed / replace contract

This repo is the **seed**: copy `APE/`, `tools/`, `scripts/`, `postmortem/`,
`reports/`, and `ape.project.yaml` as-is; `init_habitat.py` replaces the
placeholders and scaffolds `models/<model_id>/`. Fill in `APE/memory/`
(`brief.md`, `assumptions.md`, `decisions.md`) with the new habitat's own
facts — the instructions under `APE/instructions/` stay generic across FEM
mechanics domains and should not need habitat-specific edits.

**Canonical worked example:** `nmorabowen/steel-connection` — a continuum
steel shear-connection contact campaign built from this seed. Read it to see
the contract in use end to end (elastic oracle → contact → reporting →
postmortem).
