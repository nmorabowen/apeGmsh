# Drive apeGmsh Studio from an out-of-process consumer

**Goal:** bring up the habitat (MCP and/or Qt host) against a model
project and poll `.apegmsh/` safely — without inventing a second API
over `g.model.*`.

This page is the S5e spawn / ownership / poll contract (ADR 0095
Amendment 3). Identity is always labels / physical groups / phase, not
tags.

## Project root (INV-15)

Habitat files live under `<root>/.apegmsh/`. Resolve root in this order:

1. Explicit `root=` on the MCP tool, or CLI `--root`
2. Environment `APEGMSH_ROOT`
3. Nearest ancestor of the start path that already contains `.apegmsh/`
   (a stale `~/`.apegmsh` is ignored unless start *is* home)
4. Process cwd (last fallback only)

One MCP process may serve multiple projects by passing different
`root=` values. Do not `Set-Location` the apeGmsh library checkout.

## Spawn

### MCP (agent door)

```powershell
# Quiet Ladruno/apeGmsh banners before Python starts (JSON-RPC).
$env:LADRUNO_OPENSEES_QUIET = "1"
$env:APEGMSH_QUIET = "1"
$env:APEGMSH_ROOT = "C:\path\to\model-project"   # optional default root

python -m apeGmsh.studio.mcp
```

Cursor `mcp.json` can point at `scripts/studio-mcp.ps1` (quiet env only).
Pass `root=` on each tool when `APEGMSH_ROOT` is unset.

Needs `pip install apeGmsh[mcp]` (and usually `[viewer]` for the host).

### Qt host (human viewport)

```text
python -m apeGmsh.studio script.py --phase model --root <project>
python -m apeGmsh.studio script.py --phase mesh --root <project>
```

Picks write `<root>/.apegmsh/selection.json`. Replay is a subprocess /
held-open session — agents do not share that Gmsh kernel (INV-5).

### Headless replay (no window)

```text
python -m apeGmsh.studio script.py --phase mesh --no-viewer --root <project>
```

Same gate as MCP `run_until`.

## File ownership

| File | Writer | Reader | Write style |
|---|---|---|---|
| `selection.json` | Qt host (picks) | MCP `get_selection` / `status` | atomic replace |
| `names.json` | `run_until` / replay | MCP `status`, highlight resolve | atomic replace |
| `runs.jsonl` | `run_until`, `results_pin` | MCP `status` | append (best-effort) |
| `highlight.json` | MCP `highlight` | Qt host (file poll) | atomic replace |
| `mcp_calls.jsonl` | MCP adapter (side effect) | `studio.profile` | append (best-effort) |
| `visors/` | `render` / `animate` / assess figures | agents / reports | ordinary writes |

Snapshot JSON (`selection` / `names` / `highlight`) is single-writer with
atomic replace (INV-16). JSONL ledgers are append-best-effort: more than
one tool may append (`run_until` and `results_pin` both write
`runs.jsonl`). Authored chapters go in `docs/`, never under `.apegmsh/`
(INV-12).

## Poll contract

There is no event stream yet. Consumers should watch mtimes (or re-read
after each tool call):

- After a spatial pick: `selection.json`
- After `run_until`: `names.json` and the last line of `runs.jsonl`
- After `highlight`: `highlight.json` (host applies; does not rewrite
  the envelope)
- After `render` / `animate`: paths under `visors/` (prefer `--json`
  `written` lists)

`status` is the cheap aggregate: no replay, no Qt. Torn snapshot JSON
degrades to `names_error` / `selection_error`; torn or partial
`runs.jsonl` lines degrade to `ledger_error` (good lines are kept) —
it does not raise. Tool-level `ok: true` means the verb ran; check those
`*_error` fields for habitat health.

Empty / whitespace `root=` or `APEGMSH_ROOT` is treated as unset (falls
through the INV-15 chain) — do not pass `""` hoping it means cwd.

## Cold start

`.apegmsh/` is generated and typically gitignored. A fresh clone returns
`status.empty == true`. Call `run_until(script, phase="model", root=…)`
(or the CLI equivalent) before expecting names, picks, or reports.

`status.contract_version` is the habitat semver (`1.0.0` today). Published
JSON Schema + goldens live in the `apeGmsh.studio.schemas` package
(INV-17) so a Workbench-style consumer can validate without importing
the FEM stack.

## Trust boundary

`run_until` executes the caller’s Python file on purpose. Treat it as a
**local trust boundary** — not a service to expose on a network port.

## What not to do

- Do not wrap `g.model.*` / `apeSees` as MCP tools (INV-10).
- Do not drive `viewer()` / `show_web` for agent diagnosis (INV-6).
- Do not treat tags or node ids as identity (INV-3).
- Do not invent an incompatible event log before this poll contract
  proves insufficient.
