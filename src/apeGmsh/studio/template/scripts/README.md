# APE habitat session scripts

| Script | Role |
|--------|------|
| `start.ps1` / `start_session.py` | Template check · mcp.json root · Studio status probe · session marker |
| `finish.ps1` / `finish_session.py` | Bridge port check · Studio token profiler · postmortem scaffold · optimization revisions |
| `run_case.py` | Run one model case: layout + logs + deck disclosure + `run.json` with git provenance |
| `check_template.py` | Soft-contract folder/file audit |
| `template_drift.py` | Habitat `APE/`+`scripts/` vs installed template — Lane B candidates |

## Usage (from habitat root)

```powershell
.\scripts\start.ps1
.\scripts\start.ps1 -Slug contact-campaign

.\scripts\finish.ps1
.\scripts\finish.ps1 -Slug contact-campaign
```

Or:

```text
python scripts/start_session.py
python scripts/finish_session.py
```

## MCP note

Cursor owns the **stdio** MCP process. Start/finish verify project
`.cursor/mcp.json` and record `.apegmsh/session.json`.

| Server id | Use |
|-----------|-----|
| `apegmsh-studio-habitat` | This habitat (required) |
| `apegmsh-studio-lib` | apeGmsh library worktrees only |

After start, the **agent** must call `status` on `apegmsh-studio-habitat`
and confirm `status.root`. See `APE/instructions/studio-mcp.md`.

Env: `APEGMSH_PYTHON` (optional), `APEGMSH_SRC` (apeGmsh `src` path; default
`~/.cursor/worktrees/apeGmsh/mcp-ledger/src`).
