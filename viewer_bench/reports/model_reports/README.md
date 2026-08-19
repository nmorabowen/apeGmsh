# model_reports/ — calculation reports per model

**Locked:** one package **per progressive stage** + index README.  
Rules: `APE/instructions/reporting.md`.

```text
model_reports/
├── README.md
└── <model_id>/
    ├── README.md
    └── stages/
        └── <stage_id>/
            └── report.md   (and optional emit_report / narrative)
```

## Core themes (SHOULD)

Intent/oracle · geometry · mass · materials · elements · mesh · kinematics · BCs · loads · analysis type · **solution controls (MUST block)** · cases · results vs oracle · limitations/next.

## Solver MUST block

analysis type · constraints · numberer · system · test · algorithm · integrator · (+ contact knobs if contact).

## Stage accepted (MUST)

Oracle line · EDP table · ≥1 still/curve · links to `cases/<case>/run.json`.
