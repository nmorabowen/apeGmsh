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
python -m apeGmsh.studio init --name <habitat-name> --model <model_id> [--root DIR] [--no-git]
```

Copies the packaged template into `DIR` (default cwd), substitutes the
habitat name / model id, aligns `.cursor/mcp.json`, scaffolds
`models/<model_id>/{src,cases}`, and finishes by running the new
habitat's own `scripts/check_template.py --strict`. Refuses (nonzero,
no side effects) if the target already looks initialized or is
non-empty in a conflicting way. Habitats are self-contained — they
carry their own lifecycle scripts and checker, so one survives apeGmsh
version drift.

`init` also leaves the habitat under **local git**: a repo on branch
`main` with one initial commit (`studio init: <name>`) — the only git
write any studio code ever makes (INV-24). `--no-git` opts out; a
machine without git gets a WARN and a valid un-versioned habitat; a
target already inside a work tree is left alone (no nested repos).

### Git and checkpoints (ADR 0095 Amendment 8)

The habitat's git workflow is the **checkpoint contract** — the full
text ships in every habitat as `APE/instructions/checkpoints.md`. In
short: `main` holds checkpoints only (commits where model source,
accepted cases, and reports agree); work happens on
`work/<model>/<slug>` branches, one question or lineage step each; a
branch becomes a checkpoint by a `--no-ff` merge after the
stage-accepted bar plus a review pass, and gets an annotated
`checkpoint/<model>/<slug>` tag. Each case's `run.json` carries
`model_sha` + `git_dirty` (from the habitat's own
`scripts/_habitat.py::git_provenance()`), so results on disk always
point back at the source that produced them. Results themselves never
enter git — the shipped `.gitignore` keeps bulk artifacts out and
`run.json` visible; no LFS. Process files — postmortems, memory, the
backlog — commit directly to `main`; the checkpoints-only rule governs
the model surface (`models/`, `reports/`). Forges are optional and per habitat: a PR
is the review surface for the merge and nothing more — the whole
contract works in a local-only repo, and nothing in studio calls a
forge.

`init` writes three machine-specific values into `.cursor/mcp.json`,
none of which can live in package data: the MCP `command` (the
interpreter that ran `init`), `PYTHONPATH` (where that interpreter's
apeGmsh lives — pinning it stops another editable install from winning),
and `APEGMSH_STUDIO_ROOT` (this habitat's absolute path). The server is
launched as `<python> -m apeGmsh.studio.mcp`; the quiet-banner env vars
sit in the same block, which is early enough because MCP applies `env`
before the interpreter starts.

Cloning a habitat onto another machine: all three are absolute paths, so
re-run `init` into a fresh directory or edit the file once on the new
machine (`scripts/start_session.py` fails with the exact root mismatch
and the fix). The lifecycle scripts are independent of it —
`scripts/start.ps1` / `finish.ps1` resolve their own interpreter, in
order: `APEGMSH_PYTHON`, an activated `VIRTUAL_ENV`, the office venv
names, then PATH `python`, printing whichever it picked.

Then open `APE/README.md` in the new habitat and follow session start.

## Use the example library

Before building a shape from scratch, check whether the packaged
example library already covers it — a curated set of small,
**oracle-bearing** apeGmsh recipes that ship as apeGmsh package data
and are copyable anywhere apeGmsh is installed (ADR 0095 Amendment 7).
Each example is a self-contained directory: the runnable script, a
`manifest.json` (tags, what it teaches, `requires`, provenance, and
named oracle metrics with tolerances), a short `README.md`, and a
`verify.py` that checks the metrics after a run and prints one
PASS/FAIL line each.

```text
python -m apeGmsh.studio example list [--tag <domain>]
python -m apeGmsh.studio example show <name>
python -m apeGmsh.studio example copy <name> [--dest DIR]
```

`list` prints name / tags / title for every packaged example; `show`
prints the manifest summary plus the README; `copy` lands the whole
directory in `DIR` (default `cwd/<name>`), refusing to overwrite an
existing directory. An unknown name is a clear error, nonzero exit.
Running the copied script and then `verify.py` is the agent's job —
`requires` in the manifest says what the example needs (`mesh-only` /
`openseespy` / `ladruno`). CLI only, same stance as `init` — no MCP
tool.

An example without an oracle does not ship (INV-22): every metric in
`manifest.json` was produced by an actual run, not invented. Discovery
displaces src-grep — check `example list --tag <domain>` before
reading `apeGmsh` internals to figure out how to wire up a case the
library already covers.

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
| `progress.json` | the live solve's stream tee (`opensees._run`) | run monitors (poll while busy) | atomic replace |
| `mcp_calls.jsonl` | MCP adapter (side effect) | `studio.profile` | append (best-effort) |
| `visors/` | `render` / `animate` / assess figures | agents / reports | ordinary writes |
| `assess.json` | `--assess` / MCP `assess` | `emit_report` (via `ReportBundle`) | atomic replace |
| `pins/<id>/assess.json` | `results_pin` (copies the live snapshot) | `emit_report` when that pin is referenced | atomic replace |

Snapshot JSON (`selection` / `names` / `highlight` / `host` / `project` /
`assess` / `progress`) is single-writer with atomic replace (INV-16). JSONL
ledgers are append-best-effort: more than one tool may append (`run_until`
and `results_pin` both write `runs.jsonl`). Authored chapters go in
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
- **While a solve runs**: `progress.json` — `{i, n, t, done, warnings}`
  refreshed once per `APEGMSH_PROGRESS` marker (the emitters aim for
  ~20 per analyze, so this is a slow file, not a firehose). Read it
  instead of tailing the solver log. It is written **only inside a
  habitat** (`.apegmsh/` must already exist — a plain script run
  elsewhere creates nothing) and **only once the deck emits a
  marker**, so a deck with no `analyze` leaves it untouched. The run's
  last write carries `done: true` and `ok`, on failure as well as
  success. Nothing prunes the file, so treat a `progress.json` older
  than the current run as history: gate on mtime freshness, or on
  `busy.busy` being true.

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

`status.contract_version` is the habitat semver (`1.8.0` today). Published
JSON Schema + goldens live in the `apeGmsh.studio.schemas` package
(INV-17) so a Workbench-style consumer can validate without importing
the FEM stack. Since 1.8.0 the stamp is also written **into**
`names.json` and `progress.json`, so a consumer that only reads files —
one that cannot or will not make a `status` call — can still refuse an
unknown major. Older files simply have no `contract_version`; absence
means "pre-1.8.0", not "wrong major". Paths that fall under the habitat
root (`script`, `cwd`,
and MCP `path` / `output` / `written` / `run_until.script`) are
root-relative posix; the `root` field itself stays absolute as the
resolve anchor. A `project.json` entry outside the habitat root is
refused (`OUTSIDE_ROOT`) when used as the `run_until` default.

`runs.jsonl` interleaves two record kinds, so it has two published
schemas. A **run** line carries no `kind` and validates against
`ledger.schema.json`; a **pin** line carries `kind: "pin"` and validates
against `ledger_pin.schema.json`. Discriminate on `kind` before
validating — a pin line has no `script` / `phase` / `ok` and is
correctly refused by the run schema.

A timed run line also carries `started_at` (ISO-8601 UTC) and
`duration_s` (seconds, monotonic), covering the failed replay as well as
the successful one. `ts` stays the moment the line was written — the end
of the run — so the two are not redundant. Both keys are **omitted**
rather than written as `null` when there is no honest value, so a reader
that finds `duration_s` can trust it. Lines written before 1.8.0 have
neither.

`render(session=…)` draws one pane of a saved ADR 0098 session snapshot
(`<results>.viewer-session.json`) instead of a results file. Its content
is the slot catalog the human arranged, not a view token, so `view` /
`component` / `step` / `deform` / `pack` are refused with it
(`INVALID_ARGS`) rather than ignored; `camera` and `output` still apply.

A session written by the **retired** viewer is refused and **never
renamed** — the rename-aside belongs to the interactive flow. Note the
refusal is by the file's *payload shape*, not its name: the new session
adopted `<results>.viewer-session.json` at ADR 0098 S6a, so the name
alone no longer tells the two apart. A v13 payload carries an integer
`schema_version` and no `kind`; a live one carries
`kind: "apegmsh.results.session"`.

The same still is available from the command line:

```
python -m apeGmsh.results.session render <results>.viewer-session.json shot.png
```

`results_pin(session_snapshot=…)` pins the ADR 0098 presentation session
(`<results>.viewer-session.json`). Unlike `model_h5` / `results`, which
are recorded by path and hash only, that file is **copied** into
`.apegmsh/pins/<pin-id>/session_snapshot.json`: the live snapshot is
rewritten every time the human saves, so a hash alone would pin content
that is already gone. The ledger key is `session_snapshot` because
`session` already means the run's session name. A file that is not a
session snapshot is refused (`INVALID_ARGS`) — and the discriminating
case is a v13 payload sitting at exactly that adopted path, not a file
with an obviously different name.

## Trust boundary

`run_until` executes the caller’s Python file on purpose. Treat it as a
**local trust boundary** — not a service to expose on a network port.

## What not to do

- Do not wrap `g.model.*` / `apeSees` as MCP tools (INV-10).
- Do not drive `viewer()` / `show_web` for agent diagnosis (INV-6).
- Do not treat tags or node ids as identity (INV-3).
- Do not invent an incompatible event log before this poll contract
  proves insufficient.
