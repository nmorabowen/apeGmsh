# ADR 0096 — Agent token budget (lookup vs judgment)

**Status:** Proposed (2026-08-14)

**Does not amend** [ADR 0095](0095-apegmsh-studio.md) INV-10
(MCP wraps habitat verbs, not `g.model.*` / `apeSees`). 0095
Amendment 1 rejected “New ADR 0096 for the adapter” because S4 was
already the Cursor door. This is a **different** decision: how the
agent spends tokens while *using* that habitat (skill reads, `src/`
greps, MCP calls). Append-only: this file does not rewrite 0095.

**Evidence:** a column-base studio session (footing + plate + W12×65)
in which showing quotations and launching mesh burned tokens on
`Grep` / `Read` of `src/apeGmsh/viewers/**` and the ~400-line skill,
while `skills/apegmsh/SKILL.md` **then** said “verify exact signatures
in `src/apeGmsh/`” and `references/api-cheatsheet.md` said “when in
doubt, grep `src/apeGmsh/`” (deleted at S1). Constrained by ADR 0079 (skill is
derived documentation; distillation is rewrite, not splice), ADR 0094
(assess + stills; agents do not drive Qt), and ADR 0095 (studio
habitat, INV-10 / INV-11 / INV-12).

## Context

Coding agents already write apeGmsh Python through the skill and
already have a habitat MCP (0095 S4a–S4e). What they do *not* have
is a law for **where tokens may be spent**:

- Lookup (how the library is spelled) is currently a grep into
  `src/` plus slurping the whole skill tree. That is cheap for a
  human who knows the file; it is expensive for an agent that
  re-discovers `add_box` every session.
- Judgment (load idealization, mesh size, what to name, whether a
  still matches intent, script edits) is the work the human hired
  the agent for. That spend is the product.
- Using the program produces learning: skill lies, lookup misses,
  sequences that worked for *this* model. Two failure modes sit on
  either side of that learning: freeze the catalog and keep grepping,
  or let every session graft itself into `SKILL.md` and the MCP
  catalog (pollination).
- ImageMage / studio (0095 Amendment 1: stills, pin, animate,
  report) is a habitat around the script, not a second modeller.
  There are many valid ways to generate a model. A cube still has
  one spelling.

We cannot read Cursor’s billed usage from inside apeGmsh. A proxy
we control is enough: search-into-`src/`, whole-skill slurps, and
habitat MCP payload size.

## Decision

### Part 1 — Two budgets

**Budget A — Lookup (must stay cheap).** How the library is spelled.
Default path, in order:

1. Skill router in `skills/apegmsh/SKILL.md`: a short table
   (task → one `references/*.md` file). Read that file. Do not load
   the whole skill tree.
2. Generated API index (S2): signatures of public composites, built
   at `scripts/sync_skill.py` time from live `src/` so freshness is
   a build artifact, not an agent grep. Door:
   `python -m apeGmsh.studio.lookup SYMBOL` returning ~20 lines
   (signature, skill pointer, one-line doc). Never a module dump.
   `--check` diffs the committed index against a live harvest.
3. `src/` read/grep is an **index miss**, not the happy path.
   Allowed only when the index has no hit *and* the change is inside
   apeGmsh itself (library maintenance). Authoring a model script is
   not library maintenance.

The skill’s “verify in `src/`” and the cheatsheet’s “when in doubt,
grep `src/`” lines are invariants to delete at S1. ADR 0079 still
holds: the skill is derived documentation, not a second API.

**Budget B — Judgment (this is where spend belongs).** Engineering
criteria, spatial identity, and script edits:

- Habitat MCP already in 0095 INV-10: `status` / `get_selection` /
  `assess` / `render` / `animate` / `emit_report` / `highlight` /
  `promote_selection`.
- Choosing load idealization, mesh size, material, what to name,
  whether the still matches the intent.
- Writing/diffing the `.py` (0095 INV-2 / INV-9: the script owns
  geometry).

### Part 2 — Observe open, mutate closed (INV-13)

Using the agent is how we learn where lookup is expensive. That
learning must **consolidate** into the skill / index / MCP catalog —
never by letting a session graft itself into those surfaces.

```
Sessions → profiler ledger → reviewed promotion PR → skill / index / MCP
Sessions ──────────────────────────────────────────x (forbidden)
```

Two failure modes this ADR refuses equally:

| Closed forever | Open pollination |
|---|---|
| Freeze skill + MCP after S0; ignore measured `src_search` misses; the catalog never learns | Agent (or a daemon) writes `SKILL.md` / adds MCP verbs / dumps session prose into the cheatsheet every turn |

- **Open:** profiler + `.apegmsh/mcp_calls.jsonl` + lookup **misses**
  (symbol, which reference was tried, whether `src/` was opened) +
  **skill conflicts** (skill said X, index or a green test says Y) +
  **working-step notes** (the sequence that actually ran).
- **Closed:** skill body, generated index, MCP tool catalog. A miss
  does not become a new verb at runtime. Promotion is a
  human-reviewed PR: a cheatsheet line, a router row, an index
  entry, or — rarely — a new See-family tool named here. Same
  discipline as 0095 INV-11 (`kind=` is a closed catalog).
- 0095 INV-12 already said authored docs do not live in
  `.apegmsh/`. This ADR adds: session transcripts are not a skill
  source. Distillation (ADR 0079) is rewrite-from-evidence, never
  splice-from-chat.

A frozen catalog with a high miss rate is not closed architecture;
it is an unmeasured grep habit. An auto-updating skill is not
learning; it is pollination.

**Promotion bar** (S5): a MCP `lookup` **miss** (`kind=miss`, not
ambiguous) is eligible at **3** repeats of the same symbol in the
**model-cwd** `.apegmsh/mcp_calls.jsonl`. CLI lookup and transcript
`src_search` are out of scope. The fix is one index/router line, not
a new MCP CAD verb. `python -m apeGmsh.studio.profile --promote`
prints eligibility and writes nothing. A **skill error** (stale
signature, “grep `src/`” as the happy path) does not wait for
repetition — one confirmed conflict with the generated index or a
green `# verified:` test is enough to owe a skill PR.

### Part 3 — Rigid API vs recommended generation paths (INV-14)

Not everything that “worked” becomes a law. Split the catalog by
**one spelling** versus **many procedures**.

| Rigid (normative) | Recommended (non-normative) |
|---|---|
| How a cube is made: `g.model.geometry.add_box(...)` (or the indexed synonym). Names before tags. MCP habitat verbs that exist. `# verified:` snippets that *are* the call. 0095 INV-10. | How a model is *generated* from the ImageMage / studio bridge: CAD-then-mesh, parallel `--phase model` + `--phase mesh`, quotations on, assess-then-render. Many scripts are valid. |

ImageMage is a habitat around the script, not a second modeller. A
session that found a good order may keep that order as a
**consideration**:

- Prefer comments (or a short header) **in the `.py`** — the script
  owns geometry and may own *this model’s* chosen procedure without
  teaching every future model the same pipeline.
- Optionally a labeled **Recommendation** in
  `skills/apegmsh/references/workflows.md` (“one way that worked”),
  never “the” studio path.

The skill and MCP must **not** fail the agent for skipping parallel
mesh, quotations, or a particular still `kind=`. Those are Budget B.
Forcing them would freeze ImageMage the same way wrapping `add_box`
in MCP would freeze the library.

Two promotion *classes*, both still closed-write:

| Class | Lands in | Does not land in |
|---|---|---|
| **Skill error** (rigid) | Patch `skills/apegmsh/` then `scripts/sync_skill.py` | Chat; hand-edit of `.claude/`; MCP verb invented to paper over the lie |
| **Working steps** (recommended) | Comments in *that* `.py`, and/or a non-normative Recommendation in `workflows.md` | A mandatory pipeline in `SKILL.md`; a new MCP CAD tool; a transcript dump |

In-session the agent **may propose** a skill diff (error) or script
comments (recommendation). It must **not** silently rewrite the
derived `.claude/skills/apegmsh-helper/` copy. Canonical remains
`skills/apegmsh/`; derive is still `sync_skill.py`. `# verified:` is
for rigid calls only, not studio procedure.

MCP does not grow `fix_skill` or `remember_steps` in S0–S5. The
ledger records the event; the PR or the script comment is the
mutation. In-session agents propose a patch; they do not write
`skills/apegmsh/` or run `scripts/sync_skill.py`.

The column-base session is both classes: “grep `src/`” is a skill
error (fix the skill). “CAD quotations + mesh in a second process”
is a recommendation for that model — keep it in the script if
useful; do not make it the only ImageMage path.

### Part 4 — MCP growth (0095 INV-10 restated, not reversed)

MCP still must not wrap `g.model.*` / `g.mesh.*` / `g.constraints.*`
/ `apeSees` primitives, tags-as-identity, or Qt.

MCP **may** wrap the **lookup workflow** as a See-family verb,
because that is catalog inspect, not CAD:

| Allowed later | Why it is habitat, not the library |
|---|---|
| `lookup(symbol)` | Same payload as the CLI index; ~20 lines; skill pointer. Widens 0095’s already-named `status / inspect` row (`inspect` was never a shipped tool). |
| `agent_profile` **or** a ledger file the profiler reads | Measurement, not authoring. |

MCP must not grow `add_box`, `generate`, `FourNodeTetrahedron`, or
“return this source file.” If lookup cannot be answered from the
generated index, the tool returns a miss + which skill reference to
read — it does not grep `src/` inside the MCP process (0095 INV-5:
no live Gmsh; this ADR adds: MCP is not a source browser).

Wrapping CAD to save tokens is rejected. That freezes a moving
library (0095 already rejected it). The cheap path is the index +
skill, not more verbs.

### Part 5 — Profiling is a sidecar, not the product

Same administrative shape as `apeGmsh.studio` / `apeGmsh.assess`
(not a composite, not re-exported from `apeGmsh/__init__.py`).

Two inputs:

- **Owned:** `.apegmsh/mcp_calls.jsonl` (MCP adapter appends tool
  name + payload bytes; S4).
- **Imported:** a Cursor (or other) JSONL transcript path. Classify
  each tool call: `skill_read` / `index_lookup` / `habitat_mcp` /
  `src_search` / `src_read` / `other`. Heuristic tokens = chars/4.
  Emit a summary: counts, estimated tokens, **`src_search` rate**
  (defect).

CI may later fail a skill-file token budget (`SKILL.md` +
per-reference caps). That is a skill-hygiene gate, not a
live-session meter.

Do **not** put the profiler in the MCP process as a required
round-trip every turn. Profiling that costs tokens to run has
failed.

## Slices

S0 is this ADR. Later slices wait on ratification. No profiler, no
MCP `lookup`, no skill rewrite, no index generator in the S0 PR.

| Slice | Ships after ratification |
|---|---|
| **S0** | This ADR + README row + CHANGELOG. |
| **S1** | Skill router + delete “grep `src/` by default”; `scripts/sync_skill.py` remains the only derive path. |
| **S2** | Generated API index + `python -m apeGmsh.studio.lookup`. |
| **S3** | MCP `lookup` wrapping that CLI (See family), if S2 payloads are stable. |
| **S4** | Transcript / MCP-ledger profiler CLI + optional CI skill-size budget. |
| **S5** | Promotion: lookup/skill errors → rigid skill/index PR; working ImageMage steps → script comments and/or non-normative `workflows.md` Recommendation. Never a forced generation pipeline. |

**S1 shipped (2026-08-14):** skill router in `skills/apegmsh/SKILL.md`;
deleted “verify / grep `src/`” as the authoring lookup;
`scripts/sync_skill.py` remains the derive path.

**S2 shipped (2026-08-14):** generated `src/apeGmsh/studio/_api_index.json`
from live composites + `apeSees` namespaces; door
`python -m apeGmsh.studio.lookup SYMBOL` (~20 lines: signature, skill
pointer, one-line doc). Rebuild via `--build` or `scripts/sync_skill.py`
(write path). `--check` and `test_committed_index_matches_live_harvest`
diff live signatures against the committed JSON. Ambiguous hits print
bounded signatures.

**S3 shipped (2026-08-14):** MCP `lookup(symbol)` is See-family inspect
of the generated index (same ~20-line payload as the CLI). Miss returns
the skill pointer; the MCP process does not grep `src/`. No CAD verbs.

**S4 shipped (2026-08-14):** sidecar
`python -m apeGmsh.studio.profile` reads `.apegmsh/mcp_calls.jsonl`
(adapter appends tool name + payload bytes) and/or a transcript JSONL.
Heuristic tokens = chars/4. Defect metric is `src_search_rate`.
`--skill-budget` is a skill-file character gate. Not an MCP tool
(`agent_profile` stays forbidden).

**S5 shipped (2026-08-14):** observe-only MCP-lookup-miss eligibility.
`python -m apeGmsh.studio.profile --promote` lists `kind=miss`
classes that hit the bar (**3** repeats) in the **model-cwd**
`.apegmsh/mcp_calls.jsonl`. Ambiguous hits are not misses. CLI
lookup and transcript `src_search` are out of scope. Skill-error /
working-step doors are restated, not detected. It writes nothing.
Canonical remains `skills/apegmsh/`; humans run `sync_skill.py`
after merge. ImageMage generation is a non-normative Recommendation
in `workflows.md` (INV-14). No `fix_skill` / `remember_steps` MCP
verbs. This is not a closed Budget A / INV-13 learning loop.

**Index sidecars (2026-08-14):** the S2 harvester also indexes `Part`
/ `Results` / `Cluster` / `Job` / `Assembly` (constructor + public
methods; nested Part composites are not walked) and fluent
`g.model.select` (`in_box` / `to_label` / `to_physical`). Not a new
slice and not a new MCP verb — those symbols no longer force `src/`
grep.

**Measured column-base session (2026-08-14):** transcript profiler
on the footing + plate + W12×65 chat: 293 events,
`src_search_rate` 0.0648. Habitat MCP 3 (`status`×2, `assess`×1).
Index lookup went through CLI Shell — the live Cursor MCP process
still had no `lookup`. 65 `src_read` + 19 `src_search` were almost
all `viewers/**` (isolate the footing). The defect metric
understates: Shell `rg` of `src/` classifies as `other`.
`.apegmsh/mcp_calls.jsonl` was never written (adapter cwd `$HOME`),
so `--promote` had nothing to score. Last-step `results.assess()`:
0 error, `RES.U_VS_DIAG` ‖u‖=21.3 m / diag=3.70 m (info) — J2
mechanism under LoadControl, not a footing issue. Pin
`pin-20260815T001254Z`.

## Alternatives considered

| Rejected | Why |
|---|---|
| **MCP tools for `g.model.*` / `apeSees` to save tokens** | 0095 INV-10; freezes a moving library; skills already author Python. |
| **Amendment of 0095 instead of this ADR** | Different decision. 0095 already used “no 0096” for the *adapter*. |
| **Trust the 400-line skill + grep `src/` as the lookup path** | The measured waste. |
| **Live billed-token API from Cursor as a library feature** | Not ours; proxy metrics only. |
| **Profiler as a mandatory MCP tool every turn** | Measurement that burns the budget. |
| **Autodoc / dump-the-module MCP** | Not ~20 lines; recreates grep. MCP is not a source browser. |
| **Auto-pollinate the skill/MCP from transcripts** | Chat is residue; 0079 distillation is rewrite, not splice; 0095 INV-11 already forbids open-ended catalogs. A proposed skill diff in-session is allowed; applying it is a PR. |
| **No door to fix a wrong skill** | Agents keep grepping `src/` to work around stale instructions. |
| **Remember working steps only in chat** | The next session pays the lookup tax again. They belong as comments in that script (and optionally a non-normative Recommendation), not as a new MCP verb. |
| **One forced ImageMage / studio pipeline in the skill** | Generation has many valid paths; recommendations must not become fail-loud procedure. |
| **Freeze the catalog with no promotion path** | Then `src/` grep stays the real lookup and this ADR is theater. |

## Consequences

**Positive:**

- Token spend has a named split: lookup is a catalog; judgment is
  the product.
- 0095 INV-10 survives. MCP `lookup(symbol)` is See-family inspect,
  not CAD.
- Skill errors have a door (S1 / S5) without inviting pollination.
- ImageMage procedures stay recommendations; the script can carry
  *this* model’s chosen order without freezing every future model.

**Negative:**

- Until a miss is promoted, agents still pay `src/` grep on symbols
  the index does not harvest (`Part` / `Results` / `Cluster` at S5).
- The profiler is a proxy (chars/4, path classification), not
  Cursor’s bill. Optimization is against `src_search` rate, not
  against an invoice we cannot see.
- Promotion is slower than auto-write. That is the point.

## Open questions (do not block S0)

1. Closed at S4: `SKILL.md` ≤ 24_000 chars; each `references/*.md`
   ≤ 200_000 (`--skill-budget`).
2. Closed at S3: new MCP tool `lookup(symbol)`. Did not reuse the
   unimplemented 0095 `inspect` name.
3. Transcript schema versions (Cursor JSONL vs a documented generic
   event list) — S4 classifies heuristically; a stricter schema is
   later.
4. Closed at S5: bar = **3** repeated MCP `lookup` misses
   (`kind=miss`) of the same symbol in the model-cwd
   `.apegmsh/mcp_calls.jsonl`. Ambiguous is not a miss. A skill
   error still owes a PR after one confirmation.

## Acceptance (S0)

- This file exists, Status Proposed, names INV-13 and INV-14, the
  two budgets, the S0–S5 table, and the rejected alternatives.
- `decisions/README.md` has a 0096 row.
- CHANGELOG has a new Unreleased section; the frozen
  `## Unreleased — …` ledger line is untouched.
- No profiler implementation, no MCP `lookup`, no skill rewrite.

## Reference

- [0079-documentation-architecture.md](0079-documentation-architecture.md)
- [0094-agent-assess-and-viewer-render.md](0094-agent-assess-and-viewer-render.md)
- [0095-apegmsh-studio.md](0095-apegmsh-studio.md)
- Canonical skill: `skills/apegmsh/` (derive: `scripts/sync_skill.py`)
