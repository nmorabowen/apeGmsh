# Reporting — document types (LOCKED)

**Status: locked** (human accepted panel 2026-08-16).  
Habitat-specific facts go in report bodies; this file is the cross-project rule.

| Folder | Role |
|--------|------|
| `reports/technical_briefs/` | Project + per-model architecture |
| `reports/model_ledgers/` | Model-scoped modeling decision accounting |
| `reports/model_reports/` | Deep calculation reports + results (per progressive stage) |
| `reports/figures/` | Canonical promoted figures |
| `APE/memory/` | Short habitat/process memory (not a model report) |
| `references/` | Optional external PDFs |

```text
technical_briefs  →  “how this habitat/model is organized”
model_ledgers     →  “what we chose and why (modeling audit)”
model_reports     →  “full calculation memory + results vs oracle”
```

---

## Locked decisions

| Topic | Rule |
|-------|------|
| Briefs | **One** `project_architecture.md` + **one** brief per `models/<id>/` |
| Ledger vs memory | **Split:** `APE/memory/decisions.md` = habitat/process; `model_ledgers/<id>.md` = modeling audit |
| Report granularity | **One report package per progressive stage** + thin `model_reports/<id>/README.md` index |
| Chapter menu | 14 core themes; optional: damping, geometric NL, recorders/outputs, parallel — **only when used** |
| Mesh convergence | Subsection **when a study was run**; otherwise state “not studied” |
| Solver MUST block | Always: **analysis type, constraints, numberer, system, test, algorithm, integrator**; plus **contact/penalty knobs** when contact exists |
| Figures | Canonical **`reports/figures/`**; local `model_reports/<id>/…/figures/` only as aliases/copies if needed |
| Studio `emit_report` | Markdown archive **SHOULD** land under `model_reports/<id>/` (or the stage folder) |
| Stage accepted | **Oracle line + EDP table + ≥1 still/curve + case links** (`run.json`) |
| Case provenance | Stage reports name each quoted case's `model_sha` (from `run.json`); `git_dirty: true` cases are disclosed as dirty. Disclosure, not enforcement (ADR 0095 Amendment 8) |
| Language | Template **English**; translations MAY later |
| Audience | Default **engineer + agent**; client-facing = optional export from briefs (no second mandatory tree) |

---

## MUST / SHOULD / MAY

### technical_briefs

| | |
|--|--|
| SHOULD | `project_architecture.md` + `models/<model_id>.md` for each active model |
| MAY | Stay short; point to ledger and stage reports for depth |

### model_ledgers

| | |
|--|--|
| SHOULD | Append-only modeling decisions (alternatives, why, evidence) |
| MAY | Mark rows superseded; defer solver minutiae to the stage report |

### model_reports

| | |
|--|--|
| SHOULD | One stage folder (or chapter set) per progressive stage with the core theme menu |
| MUST (when claiming results) | Oracle comparison + figures/tables + links to cases |
| SHOULD | Cross-ref alternative meshes/models when they exist |

---

## Model report layout

```text
reports/model_reports/<model_id>/
├── README.md                 ← index of stages
└── stages/
    ├── <stage_id>/           ← e.g. elastic, contact, nl_material
    │   ├── report.md         ← calculation memory for that stage
    │   └── (optional emit_report outputs)
    └── …
```

### Core themes (every stage report SHOULD address)

1. Intent & oracle  
2. Geometry  
3. Mass / inertia  
4. Materials  
5. Elements  
6. Mesh (+ convergence only if studied)  
7. Kinematics / interaction  
8. Boundary conditions  
9. Loads / excitation  
10. Analysis type  
11. Solution controls (MUST solver block above)  
12. Cases run  
13. Results (plots, tables, stills, post-process, vs oracle)  
14. Limitations & next stage  

### Optional themes (when used)

Damping · geometric nonlinearity · recorder/output requests · parallel/MPI · strategy ladder details beyond the MUST block.
