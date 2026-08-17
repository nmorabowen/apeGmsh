# Continuous improvement — what postmortems are for

Postmortems exist so the **next** session is cheaper and clearer. Skills and
library implementations matter, but they are only two levers. Use this map
when filling `02_friction.md` and `postmortem/backlog/open.md`.

Keep it generic: every APE FEM habitat can burn down the same classes of
improvement; only evidence and priorities are habitat-specific.

---

## Improvement surfaces (full map)

| Surface | Tag | What “better” looks like |
|---------|-----|---------------------------|
| Skills (content & triggers) | `skills` | Right skill opens early; anti-examples; links to progressive stages; GitHub-canonical |
| Library implementations | `opensees` `apegmsh` `apesees` `apequake` `design-libs` | APIs, diagnostics, defaults, staged NL, Results |
| Studio / habitat MCP | `studio` | Connected MCP; assess/render/report paths; index lookup beats src grep; profiler on close |
| Intent bridges | `tools-bridges` | apeCAD/apeSketch used; human vs agentic archives; confirm schemes before build |
| Playbook / instructions | `ape-instructions` | Socratic + progressive gates that agents actually follow |
| Habitat memory | `memory` | Brief/decisions/oracles that prevent re-discovery next chat |
| Case & oracle library | `cases-oracles` | Golden elastic/contact/SSI cases; named metrics; copyable recipes |
| Reports & artifacts | `reports` | Figure naming; emit_report → `reports/`; less visor scavenger hunt |
| Observability | `observability` | Automatic agent `profile_studio.json` / `profile_promote.json`; token class rates; MCP call ledger |
| FEM analysis profiling | `fem-profile` | Ladruno `profile.h5` on worth-it cases; rollups feed opensees-performance / expert |
| Agent runtime / Cursor | `agent-runtime` | Rules, hooks, MCP config, skill discovery pointed at `<github>/` |
| Env & reproducibility | `repro` | PYTHONPATH, Studio root, venv, one-command open tools |
| Evaluation & CI | `evals-ci` | Smoke cases; agent evals for “asked before build”; gate checks |
| Session continuity | `continuity` | Start next session from backlog P0/P1; don’t lose thread |
| Human interaction design | `human-loop` | Question budget; when to interrupt; confirm UX |
| Template seeding | `template` | New habitat from APE/tools/postmortem in minutes |
| Upstream governance | `upstream` | ADR/issue in the right repo; don’t bury product work only in one habitat |

---

## Beyond “write better skills / code”

High-leverage process moves that postmortems should look for:

1. **Close the observability loop** — every close attaches a real profile
   (Studio profiler / ledger). Without numbers, improvement is folklore.
2. **Burn down backlog at session start** — first agenda item: P0/P1 from
   `postmortem/backlog/open.md`, not only the new physics ask.
3. **Promote patterns into the template** — if this habitat learned a
   gate, put it in `APE/instructions/` first (Lane B below), not only in
   this habitat’s memory.
4. **Golden cases as oracles** — progressive modeling needs reusable
   “stage 2 elastic passed” artifacts other habitats can clone.
5. **Profile worth-it FEM runs** — `ops.profiler` / `analyze(profile=)` →
   `profile.h5` under the case; feed hotspots into opensees-performance /
   opensees-expert (`analysis-profiling.md`). Keep separate from the agent
   token sidecar.
6. **Make the happy path the default** — habitat MCP connected; reports
   path default; skill doors in session start; tools launchers one command.
7. **Agent-readable failures** — Newton/contact/BC errors that say *what
   to check next* beat silent divergence (library + Studio).
8. **Eval the process, not only the FEM** — did we ask before build? did
   we escalate? did habitat_mcp stay 0?
9. **Shrink scavenger-hunt** — index/lookup and skills should displace
   raw `src_search` for known APIs (watch `src_search_rate`).
10. **Human-loop craft** — fewer mega-questions; labeled A/B schemes;
   explicit “proceed without sketch” when skipping.
11. **Repro pack** — same Python, same Studio root, same tool ports every
    time; document in habitat README, fix in `repro` backlog rows.
12. **Split habitat vs upstream** — instruction fix today; library ADR
    this week; don’t pretend a chat workaround is a product.
13. **Cadence** — periodic review of `backlog/open.md` (e.g. weekly):
    merge duplicates, close Done, pick one P1 to implement.

---

## Promotion lanes

Not every improvement lives at the same altitude. Three lanes, one deferral:

- **Lane A — habitat memory** (immediate): brief / decisions / assumptions
  in this habitat. This already happens every session.
- **Lane B — template**: edit *this* habitat's `APE/` copy first, prove it
  across at least one session, then PR the change into apeGmsh's
  `src/apeGmsh/studio/template/**`. Habitat↔template drift is by design;
  reconciliation is manual and voice-preserving — never generated.
  `scripts/template_drift.py` lists the candidates.
- **Lane C — upstream** (library / skills / fork): a backlog item that
  belongs upstream becomes an ADR amendment or issue in the right repo and
  runs the propose → implement → probe → merge cadence — the loop, not
  just "note an issue".
- **Deferred — cross-habitat rollup**: one habitat exists today; name it,
  build nothing.

---

## How a friction becomes the right kind of work

```text
Friction observed
    │
    ├─► one-line playbook / memory fix     → ape-instructions | memory
    ├─► skill trigger / example            → skills
    ├─► Studio / MCP / profiler            → studio | observability
    ├─► apeCAD / apeSketch UX or archive   → tools-bridges
    ├─► bridge / mesh / Results API        → apesees | apegmsh
    ├─► solver / contact / constitutive    → opensees
    ├─► golden case / oracle recipe        → cases-oracles
    └─► needs product decision             → upstream (ADR/issue)
```

Smallest wedge first. Prefer a metric that will prove the fix in the
**next** postmortem (`habitat_mcp > 0`, lower `src_search_rate`, stage-N
oracle met on first try).

---

## Success criteria for the improvement system itself

The postmortem machinery is working when:

- The same friction does not reappear three sessions without a backlog row.
- P0/P1 items get closed with a link (commit, ADR, instruction path).
- Metrics trends move (MCP used, search rate down, fewer false paths).
- New habitats inherit playbook fixes without re-living the same habitat
  pain in chat alone.
