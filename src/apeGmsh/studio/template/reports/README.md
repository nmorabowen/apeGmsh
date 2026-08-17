# reports/ — authored engineering documentation

**Status:** taxonomy **locked** — see `APE/instructions/reporting.md`.

```text
reports/
├── README.md
├── technical_briefs/         ← project + per-model architecture
├── model_ledgers/            ← modeling decision accounting
├── model_reports/            ← per-model / per-stage calculation reports
│   └── <model_id>/stages/<stage>/
└── figures/                  ← canonical promoted stills / curves
```

| Folder | Job |
|--------|-----|
| **technical_briefs/** | Habitat + model architecture (short) |
| **model_ledgers/** | Why this idealization / mesh / contact / … |
| **model_reports/** | Full calculation memory + results vs oracle (per stage) |
| **figures/** | Shared image pool referenced by reports |
