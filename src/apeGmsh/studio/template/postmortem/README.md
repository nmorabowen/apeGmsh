# postmortem/ — end-of-session agentic accounting

When the human says the **session is over** (or asks for a postmortem),
prefer `scripts/finish.ps1` first, then complete friction triage per
`APE/instructions/session-postmortem.md`.

```text
postmortem/
├── README.md
├── templates/
│   ├── 01_metrics.md
│   └── 02_friction.md
├── sessions/
│   └── YYYY-MM-DD_<slug>/
│       ├── 01_metrics.md
│       ├── 02_friction.md
│       ├── 03_optimization_revisions.md   ← finish scaffold
│       ├── profile_studio.json            ← token metrics (--json)
│       └── profile_promote.json           ← --promote eligibility
└── backlog/
    ├── README.md
    └── open.md
```

## Deliverables

| # | Artifact | Purpose |
|---|----------|---------|
| 1 | `01_metrics.md` + `profile_studio.json` | Tokens / MCP / search rates |
| 2 | `02_friction.md` → `backlog/open.md` | Frictions + implementations |
| 3 | `03_optimization_revisions.md` | Finish checklist + backlog snapshot |

## What this is not

- Not a substitute for `reports/` engineering chapters.
- Not automatic without `.apegmsh/mcp_calls.jsonl` under **this** habitat
  (use MCP server `apegmsh-studio-habitat`).
