# Studio / habitat MCP — agent door (ADR 0095 / 0096)

Short session guide. **Read this when using Studio tools.** Full FEM/policy
docs stay in `how-we-work.md` / Socratic / progressive.

## Server id (this habitat)

| Server | Role |
|--------|------|
| **`apegmsh-studio-habitat`** | This workspace — `APEGMSH_STUDIO_ROOT` = habitat root |
| `apegmsh-studio-lib` | User-level apeGmsh worktree (library work) — do **not** use for this habitat |

After `scripts/start.ps1`, call **`status` on `apegmsh-studio-habitat`** and
confirm `status.root` equals the habitat path printed by start. If it does
not, reload MCP / disable the colliding server.

## Tool map (when to use)

| Tool | Use for | Avoid |
|------|---------|--------|
| **`lookup(symbol)`** | apeGmsh / apeSees public API (~20 lines) | Grepping apeGmsh `src/` for authoring |
| **`status`** | Last run / names / empty? Prefer brief reading — payload can be large | Treating it as intent confirmation |
| **`run_until`** | Replay script to `model` / `mesh` / `results` | Qt windows |
| **`assess`** | Verdict + markdown from results | Replacing a locked model report |
| **`render`** | Stills under `.apegmsh/visors/` | Leaving figures only in visors — promote to `reports/figures/` |
| **`animate`** | history / yield movies | |
| **`results_pin`** | Stamp model.h5 / results into ledger | |
| **`emit_report`** | markdown archive (SHOULD land under `model_reports/<id>/`); html/canvas as needed | Putting authored chapters only in `.apegmsh/` |
| **`get_selection`** | Qt pick envelope | |
| **`highlight`** | Point at named faces (writes highlight.json) | |
| **`promote_selection`** | Suggested `select…to_label()` — does **not** edit `.py` | Assuming the script was rewritten |

There is **no** profiler MCP tool. Token profile:
`python -m apeGmsh.studio.profile` (see `finish.ps1`). FEM stack profile:
`ops.profiler` / `STEEL_PROFILE` (`analysis-profiling.md`).

## ADR 0096 — token hygiene

1. **`lookup` before `src_search`** for apeGmsh API questions.  
2. Skills catalogs are doors — listing ≠ loaded.  
3. Watch `src_search_rate` and `habitat_mcp` in postmortem metrics.  
4. Do not dump full `status` entities into chat when a one-line root check
   suffices (upstream may add a brief mode — backlog `studio`).

## emit_report targets

| format | Typical target |
|--------|----------------|
| `markdown` | `reports/model_reports/<id>/…` (archive) |
| `html` | print/docs path if used |
| `canvas` | IDE canvases (not git-portable) |
