# Studio: replay, picks, MCP and habitats (ADR 0095 / 0096)

Read this when the task uses `python -m apeGmsh.studio`, its MCP server, a
Studio habitat, or the packaged example library. Writing a model script
itself is `model-scripts.md`.

## Replay and picks

`python -m apeGmsh.studio script.py` replays the script up to `--phase`
(default `model`: it **stops before** `generate()`; `--phase mesh` stops
before `apeSees` / `Results`). The session stays open. It opens MeshViewer if
the replay produced elements, otherwise ModelViewer.

- Picks write `.apegmsh/selection.json` under the resolved project root
  (`root=` / `--root` / `APEGMSH_ROOT` / `APEGMSH_STUDIO_ROOT` / nearest
  `.apegmsh/` / cwd; ADR 0095 INV-15).
- A successful stop writes `.apegmsh/names.json` (labels, physical groups,
  counts and bounding boxes of what exists, not what was clicked) and appends
  `.apegmsh/runs.jsonl` (timestamp, hash, phase, counts).
- `python -m apeGmsh.studio --status` reads those files without replaying.
- The host window refreshes itself (ADR 0095 Amendment 4): a Refresh toolbar
  action / F5 replays the entry script, and a file watch replays it when the
  saved file's mtime settles (`--no-watch` or the Watch toggle turn it off).
  Both honor the `busy.json` lock and append `runs.jsonl` with
  `trigger: "refresh"` / `"watch"`. After editing the script the human's
  window follows the save: the agent need not reopen the host, and must not
  treat a `BUSY` `run_until` as an error while the host is replaying.
- Before a spatial edit, read the envelope, `names.json` and `--status`.
  Identity is `labels` / `physical_groups` / `phase`, not dimtags. An unnamed
  pick's first script edit is `.to_label()` / `.to_physical()`.
- After a solve, still `assess()` + `render()`, not `viewer()`. When the agent
  must look, write visors under `.apegmsh/visors/` (`results.render` /
  labelled matplotlib) and read those PNGs; never capture the OS desktop or
  the human's Qt window.
- Studio generation order (CAD then mesh, quotations on, parallel processes)
  is a **recommendation** for a given script, not a required path (ADR 0096
  INV-14). Authored chapters go in `docs/`; visors stay under
  `.apegmsh/visors/`.

## Habitats

`python -m apeGmsh.studio init --name <habitat> --model <model_id>` stamps a
fresh APE Studio habitat from the packaged template (ADR 0095 Amendment 6):
playbook, lifecycle scripts, checker, model lineage. It leaves the habitat
under local git, a repo on `main` with one initial commit; `--no-git` opts
out, and a machine without git gets a WARN and a valid habitat (Amendment 8).

Habitat git follows the **checkpoint contract**
(`APE/instructions/checkpoints.md`): `main` holds checkpoints only, work
happens on `work/<model>/<slug>` branches, and a `--no-ff` merge plus a
`checkpoint/<model>/<slug>` tag is acceptance. Each case's `run.json` carries
`model_sha` / `git_dirty` via the habitat's own
`scripts/_habitat.py::git_provenance()`. Results never enter git (the
shipped `.gitignore`); lifecycle scripts print git state but never write it;
forges are optional, and a PR is only the review surface.

## The example library

Before building a covered shape from scratch, check
`python -m apeGmsh.studio example list [--tag <domain>]`, the oracle-bearing
example library (ADR 0095 Amendment 7). `show <name>` prints the manifest and
README; `copy <name> [--dest DIR]` lands a runnable directory (script,
`manifest.json`, `README.md`, `verify.py`) anywhere apeGmsh is installed, and
refuses to overwrite. Every example's metrics were produced by an actual run
(INV-22). Discovery displaces reading the source, as ADR 0096's lookup does,
but for recipes rather than symbols. It is a CLI only, not an MCP tool.

## The read-only MCP server

`python -m apeGmsh.studio.mcp` (needs `pip install mcp`; ADR 0095 S4a–S4e /
S5a, ADR 0096 S3). Tools: `status`, `get_selection`, `lookup`, `run_until`,
`assess`, `render`, `animate(kind=history|yield)`, `results_pin`,
`emit_report(format=markdown|html|canvas)`, `highlight`,
`promote_selection`. Names first, no Qt, no `setup=`. Every tool accepts an
optional `root=` (INV-15).

- `status(mode=brief|full)` defaults to **brief** (root, last run, labels,
  physical groups, counts, pick, plus `text`); use `mode=full` only for
  debugging. **Root check → brief; entity hunting → `lookup` /
  `names.json`, not a full status dump.**
- `lookup(symbol)` inspects the generated index, See-family style (~20
  lines). It is not CAD and reads no source.
- `animate(kind=yield)` is a von Mises contour of `von_mises_stress` on an
  auto-scaled deformed shape (0.12 of the diagonal; not an iso-clip; the
  π-plane comes later). `kind=formation` comes later.
- `highlight` writes `.apegmsh/highlight.json` only (host file-poll +
  `select_batch`); it leaves `selection.json` alone.
- `promote_selection` suggests
  `g.model.select(None, dim=).in_box(lo, hi).to_label()` / `.to_physical()`
  and leaves the `.py` unwritten.
- `emit_report`: markdown is the archive (`docs/`); html is a print skin
  there; canvas is a live Cursor projection (IDE `canvases/`, not
  git-portable) and still writes the markdown chapter. Run `assess` any time
  before `emit_report`: it writes `.apegmsh/assess.json`, and the chapter's
  Assess section picks up that snapshot's verdict, findings and skipped list
  automatically (`--pin` copies it, ADR 0095 Amendment 5). Do not paste
  `assess` output into the chapter by hand.
- Do not invent MCP tools for `g.model.*` / `apeSees` primitives. The
  token-budget profiler is `python -m apeGmsh.studio.profile` (a sidecar, not
  an MCP tool).

## MCP wiring (ADR 0095 Amendment 12)

Each project stamps its own `apegmsh-studio-habitat` server in
`.cursor/mcp.json` (and `.mcp.json` for Claude Code), binding **both**
`APEGMSH_ROOT` and `APEGMSH_STUDIO_ROOT` to that project's Studio root. The
user-level `~/.cursor/mcp.json` server is deliberately **unbound**: it
carries no root env and no `cwd`, because a global root is stale the moment
you switch projects. Set `LADRUNO_OPENSEES_QUIET=1` before Python starts (the
office venv `.pth` banner goes to stdout and would break JSON-RPC):

```json
{
  "mcpServers": {
    "apegmsh-studio-habitat": {
      "command": "<venv>/Scripts/python.exe",
      "args": ["-m", "apeGmsh.studio.mcp"],
      "env": {
        "LADRUNO_OPENSEES_QUIET": "1",
        "APEGMSH_ROOT": "<studio root>",
        "APEGMSH_STUDIO_ROOT": "<studio root>",
        "APEGMSH_PYTHON": "<venv>/Scripts/python.exe",
        "PYTHONPATH": "<apeGmsh>/src"
      }
    }
  }
}
```

For a Workbench project with a Studio overlay, `<studio root>` is the
**overlay folder** (e.g. `Staged Model/`), not the Workbench root. Do not pin
`cwd` to the library repo: it has its own `.apegmsh/`, and the ancestor walk
would bind there.

**Agents: check `status.root` first.** If it is not the habitat you mean,
pass `root=` on *every* Studio tool call for the rest of the session; do not
assume the server is bound correctly.

## Lookup discipline and promotion (ADR 0096)

Before claiming a method or signature exists, call MCP `lookup(symbol)` or
run `python -m apeGmsh.studio.lookup SYMBOL`. If that misses, read
`api-cheatsheet.md` (the matching heading). Do not `Read`
`src/apeGmsh/studio/_api_index.json`.

Promotion (ADR 0096 S5): an MCP `lookup` **miss** (`kind=miss`, not
ambiguous) that repeats three or more times in the **model-cwd**
`.apegmsh/mcp_calls.jsonl` is eligible for a reviewed index or router PR.
`python -m apeGmsh.studio.profile --promote` prints that list and writes
nothing. A skill error (skill against index) owes a PR after one
confirmation. Working steps stay comments in *that* `.py` and/or a
Recommendation in `workflows.md`, never a required pipeline. Propose a PR; do
not write `skills/apegmsh/` or run `sync_skill.py` in-session, and do not
edit `.claude/skills/apegmsh-helper/`.
