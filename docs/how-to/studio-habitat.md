# Drive apeGmsh Studio from an out-of-process consumer

**Goal:** bring up the habitat (MCP and/or Qt host) against a model
project and poll `.apegmsh/` safely — without inventing a second API
over `g.model.*`.

This page is the S5e spawn / ownership / poll contract (ADR 0095
Amendment 3). Identity is always labels / physical groups / phase, not
tags.

## Create a habitat

An APE Studio **habitat** is a full FEM project scaffold: the `APE/`
playbook (instructions, memory, skill/library doors), session lifecycle
scripts, a soft-contract checker, and a `models/<id>/{src,cases}`
lineage. Stamp one from the template apeGmsh ships (ADR 0095 Amendment
6):

```text
python -m apeGmsh.studio init --name <habitat-name> --model <model_id> [--root DIR]
```

Copies the packaged template into `DIR` (default cwd), substitutes the
habitat name / model id, aligns `.cursor/mcp.json`, scaffolds
`models/<model_id>/{src,cases}`, and finishes by running the new
habitat's own `scripts/check_template.py --strict`. Refuses (nonzero,
no side effects) if the target already looks initialized or is
non-empty in a conflicting way. Habitats are self-contained — they
carry their own lifecycle scripts and checker, so one survives apeGmsh
version drift.

Then open `APE/README.md` in the new habitat and follow session start.

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

The toolbar's Refresh action (or F5) re-runs the entry script at the
current phase in the host process (ADR 0095 S6a). A live `busy.json`
claim reports busy and does nothing (INV-18); an unchanged file is a
no-op ("Up to date"); a failed replay keeps the previous frame
(INV-4). Ledger records gain a `trigger` field (`open` / `refresh`).

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
| `host.json` | Qt host (claim / clear) | MCP `status` | atomic replace |
| `project.json` | successful `run_until` / replay | MCP `status`, `run_until` default | atomic replace |
| `busy.json` | `run_until` / replay (exclusive) | MCP `status` | O_EXCL create / unlink |
| `mcp_calls.jsonl` | MCP adapter (side effect) | `studio.profile` | append (best-effort) |
| `visors/` | `render` / `animate` / assess figures | agents / reports | ordinary writes |
| `assess.json` | `--assess` / MCP `assess` | `emit_report` (via `ReportBundle`) | atomic replace |
| `pins/<id>/assess.json` | `results_pin` (copies the live snapshot) | `emit_report` when that pin is referenced | atomic replace |

Snapshot JSON (`selection` / `names` / `highlight` / `host` / `project` /
`assess`) is single-writer with atomic replace (INV-16). JSONL ledgers are
append-best-effort: more than one tool may append (`run_until` and
`results_pin` both write `runs.jsonl`). Authored chapters go in
`docs/`, never under `.apegmsh/` (INV-12).

`--assess` / MCP `assess` writes `.apegmsh/assess.json` (target paths,
findings, skipped checks, figures, and the same markdown text the verb
prints) in addition to returning its payload (ADR 0095 Amendment 5).
`emit_report` reads it — quoting the verdict, findings, skipped list,
and target/timestamp in the Assess section — instead of always
printing "(none — run assess first)". `results_pin` copies the live
snapshot into `.apegmsh/pins/<id>/assess.json`; when a pin is
referenced, `emit_report` prefers that pinned copy over the live file,
so deleting or overwriting `assess.json` after pinning does not change
an already-pinned chapter's verdict.

## Poll contract

There is no event stream yet. Consumers should watch mtimes (or re-read
after each tool call):

- After a spatial pick: `selection.json`
- After `run_until`: `names.json`, `project.json`, and the last line of
  `runs.jsonl`
- After opening / closing the Qt host: `host.json` (also reflected in
  `status.host`; a dead PID is reported as `running=false`, `stale=true`)
- After `highlight`: `highlight.json` (host applies; does not rewrite
  the envelope)
- After `render` / `animate`: paths under `visors/` (prefer `--json`
  `written` lists)

`status` is the cheap aggregate: no replay, no Qt. MCP
`status(mode="brief")` is the **default** agent shape (root, last-run
intent, label/PG lists, counts summary, pick summary, plus CLI-shaped
`text`) — keep start rituals on brief so a real model does not burn
~55 KB of `names.entities` every turn. Use `status(mode="full")` only
when debugging or when a Qt-adjacent workflow needs the complete
`collect_status` dict. **Root check → brief; entity hunting →
`lookup` / `names.json`, not a full status dump.**

`status` includes `host: {running, pid, phase, stale}`, `project`
(entry script), and `busy: {busy, pid, op, phase, stale}`. Concurrent
`run_until` calls on the same root return `error.code == "BUSY"` while
another replay holds `.apegmsh/busy.json` (S5h). Torn snapshot JSON
degrades to `names_error` / `selection_error`; torn or partial
`runs.jsonl` lines degrade to `ledger_error` (good lines are kept) —
it does not raise. Tool-level `ok: true` means the verb ran; check
those `*_error` fields for habitat health.

Empty / whitespace `root=` or `APEGMSH_ROOT` is treated as unset (falls
through the INV-15 chain) — do not pass `""` hoping it means cwd.

`run_until` may omit `script=` when `project.json` already names the
entry script.

## Cold start

`.apegmsh/` is generated and typically gitignored. A fresh clone returns
`status.empty == true`. Call `run_until(script, phase="model", root=…)`
(or the CLI equivalent) before expecting names, picks, or reports.
That also writes `project.json` so later `run_until(phase=…)` calls can
omit the script path.

`status.contract_version` is the habitat semver (`1.6.0` today). Published
JSON Schema + goldens live in the `apeGmsh.studio.schemas` package
(INV-17) so a Workbench-style consumer can validate without importing
the FEM stack. Paths that fall under the habitat root (`script`, `cwd`,
and MCP `path` / `output` / `written` / `run_until.script`) are
root-relative posix; the `root` field itself stays absolute as the
resolve anchor. A `project.json` entry outside the habitat root is
refused (`OUTSIDE_ROOT`) when used as the `run_until` default.

## Trust boundary

`run_until` executes the caller’s Python file on purpose. Treat it as a
**local trust boundary** — not a service to expose on a network port.

## What not to do

- Do not wrap `g.model.*` / `apeSees` as MCP tools (INV-10).
- Do not drive `viewer()` / `show_web` for agent diagnosis (INV-6).
- Do not treat tags or node ids as identity (INV-3).
- Do not invent an incompatible event log before this poll contract
  proves insufficient.
