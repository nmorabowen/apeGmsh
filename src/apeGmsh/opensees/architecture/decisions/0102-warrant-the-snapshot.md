# ADR 0102 — Warrant the snapshot, kill the second product

**Status:** Proposed (2026-08-30)

**Owner:** nmora

**Sibling:** OpenSees-fork ADR-87 (same day, same review). That ADR
is about a C++ kernel whose merge path cannot refuse. This one is
about a Python compiler whose merge path *can* refuse, and which
grew a second kernel anyway.

This is a **governance** ADR. It adds no schema bump, no primitive,
no MCP tool. Status is `Proposed` until the owner accepts it in a
PR that does the catalogue work, not merely adds this file.

## Context

Census on this tree (HEAD `88c7df9a`, 2026-08-29, `main`):

| Quantity | Count |
|---|---|
| Age | 2026-04-05 (`pyGmsh v0.1`) → 147 days |
| Commits / merged PRs | 2,650 / 1,074 |
| GitHub stars / forks | 6 / 6 |
| `src/apeGmsh/**/*.py` | 625 files, ~290k lines |
| `viewers/` | 203 files, **84k lines** |
| `opensees/` (bridge) | 92 files, 71k lines |
| `studio/` | 48 files, 11k lines |
| `tests/test_*.py` | 816 files, ~250k lines |
| `assert True` | **0** |
| ADRs in this folder | 99 (0101 SANISAND is latest) |
| Table status `Proposed` | 25 (many already shipped) |
| `docs/*.md` | 81 (ADR 0079 cap was ~60 pages) |
| `internal_docs/*.md` | 147 files, ~57k lines |
| `CHANGELOG.md` | **712k characters**, 8,410 lines |
| Package version | still `2.0.0` |
| Neutral / OpenSees schema | **2.31.0 / 2.21.0** |
| Architecture README | still says schema 2.2.0, 81 primitives, Phase 8.6 |

The charter of the *bridge* (`architecture/charter.md`) is one
sentence: translate `FEMData` plus typed declarations into Tcl / py
/ live, then stop. “Nothing more: the bridge does not own analysis
strategies, recording strategies, or post-processing.”

The library did not stop. `viewers/` is larger than `opensees/`.
`CHANGELOG` Unreleased is a session dump. Twenty-five ADRs still
say Proposed after the code landed. The README still offers “any
solver” through `FEMData`; the only bridge on disk is OpenSees.
ADR 0100 Amendment 3 already recorded that the 51 M hex emit
ceiling **did not move to a new order of magnitude**. ADR 0083 S3
still owes a cut-face field on solids (clipping a tet box opens
empty). `tests.yml` documents that three fork/stock divergences
reached `main` (#1021) because the live surface reported green
without executing.

That is not “the physics is fake.” It is a compiler that works,
wrapped in a second product that a named human cannot warrant, and
a catalogue no human reads.

## Decision

### D1 — Keep the snapshot. End the second product.

The destination “describe once → `FEMData` → OpenSees deck +
`Results`” exists. It is how Cerro Lindo and the SSI boxes are
built. The destination “84k lines of Qt plus an agent habitat plus
1,074 PRs *are* a warranted modelling system” does not.

New work is justified only if it raises warrant of the snapshot /
emit / read path, not surface area of chrome.

### D2 — Freeze new families.

Until D3–D6 are true, **no new**:

- MCP tool / habitat verb
- viewer ontology (0098 already replaced 0058; do not replace 0098)
- constraint *kind* (`contact` / `tie` / `interface` / RBE / embed
  already exist)
- typed wrapper for the next Ladruno class tag

Allowed without a new family ADR:

- bugs, fail-loud guards, interpreter/fork-vs-stock parity
- lock / live-stock / emit-cost / qt lanes that already gate
- collapsing Unreleased; flipping Proposed→Accepted or killing the row
- finishing a *named* draft already in this table, only if a human
  can derive the residual (0083 S3 is an example)

SANISAND (0101) is inventory until D6. It is not a precedent for
0103.

### D3 — The catalogue must be readable in a diff.

- `CHANGELOG.md` Unreleased is the moving truth (0079). It is not a
  diary. One Unreleased block, current work only; history belongs
  under version headings. 712k characters is a defect.
- This README: one status cell per ADR. Shipped + still
  `Proposed` is a defect. Flip or kill, in the accepting PR.
- `architecture/README.md` must state the live schema and stop
  claiming Phase 8.6 / 81 primitives / schema 2.2.0.
- Package `version` in `pyproject.toml` either moves or we stop
  implying 2.0.0 is a product stamp.

### D4 — CI that already says no stays the merge bit. Panels are not.

Unlike the OpenSees fork, this repo *has* a merge path that can
refuse: lock tests, emit-cost gate, ruff, mypy ratchet (baseline 0),
qt-window lane, suite timeout, live-stock with an import guard so
an all-skip cannot go green. **Keep that.** Do not skip live-stock.
Do not treat “N-agent adversarial panel PASS” as a check run.

The nightly `benchmarks.yml` job still **asserts nothing** (the
comment in `tests.yml` emit-cost-gate: 124 commits reported PASS).
That job is inventory until it has a threshold.

### D5 — Split the surface: warranted / inventory / failed.

Until a feature is in the first column, Studio, the splash, and
client decks treat it as experimental. **Examples are candidates,
not a census** (a named human has not derived them under D6).

| Bucket | Meaning | Candidates (this review) |
|---|---|---|
| **Warranted** | A named human can author it from labels without this folder open; a gate turns red if emit is `return []`; both stock and fork paths are honest | `FEMData` + labels/PGs, `apeSees` Tcl/py/live emit, MP auto-emit (0022), fail-loud ndf/shell-on-solid (0046/0048), `Results.from_*`, compose + equation mortar on hex20/hex8 (0085/0086) |
| **Inventory** | Code exists; Proposed-after-shipped; chrome; missing stock path; human cannot derive cold | Studio/MCP (0095), results session window (0098), section builder GUI (0080), SANISAND wrap (0101), visual-design ADRs 0087–0090, partitioned contact (0092) |
| **Failed / absent** | Gate missed, ceiling did not move, or advertised and not built | “Any solver” via `FEMData`, ADR 0100 51 M ceiling as a new order of magnitude, 0083 S3 cut-face field on solids, charter “deck and stop” as a description of the *library* |

Moving inventory → warranted is **work**. It is not an Accepted flip
in the table.

### D6 — A named human derives it, or it is not shipped.

For every object we continue to call shipped:

- One human name on the ADR row (not “Guppi”).
- That human can write the emit for that object on a whiteboard
  (the Tcl line, the H5 group, the fail-loud).
- A gate that would turn red if the implementation were replaced
  by a no-op (not “the Qt window opened”).

If that bar is too high for Studio or the session viewer, those
packages are `draft`. Honesty is cheaper than a second product.

### D7 — This is APE’s compiler. Say so.

A pip package with a DOI and six stars is not a commons, and it is
not “solver-agnostic” until a second bridge exists. Either:

- land a *thin* second consumer of `FEMData` (even a dumb
  CalculiX/Abaqus keyword dump), or
- stop putting “any solver” on the README.

Both can wait. **D2 cannot.** Wrapping the next Ladruno tag makes
the snapshot harder to warrant every week, and it couples this
compiler to a fork that ADR-87 has just asked to freeze.

## Alternatives rejected

- **Do nothing.** The factory continues. This file becomes evidence
  we knew.
- **Delete `viewers/`.** Not in scope. The window is how humans
  look. D2 freezes *growth*, not the package.
- **Copy OpenSees ADR-87’s “Zone-A required”.** Wrong transplant.
  This CI already builds and tests on every PR. The hole here is
  catalogue + second kernel, not a babysitter that merges on a
  classTag check.
- **Freeze the snapshot too.** That is how Cerro Lindo dies. D2 is
  new *families*, not bugfixes on emit.

## Consequences

- Agents may not open 0103 because they still have context.
- A viewer pixel ADR (0087–0090 class) needs a warrant argument,
  not a screenshot campaign.
- Ladruno wrappers after 0101 are inventory by default.
- Accepting this ADR without collapsing Unreleased and the
  Proposed-shipped rows is not acceptance.

## Acceptance of *this* ADR

A PR that only adds this file is not acceptance. Acceptance is:

1. Frontmatter **Status: Accepted**.
2. `CHANGELOG.md` Unreleased cut to current work; prior dump under
   a dated heading or deleted.
3. This README: Proposed-but-shipped rows flipped or marked
   inventory in one cell.
4. `architecture/README.md` schema / primitive / Phase-8 claims
   match the tree.
5. One-page D5 table lives here or at the top of this README —
   not in a 712k changelog.

Until then this document is an opinion with a number.

## Risks

- **We freeze and a client model needs one more constitutive wrap.**
  That is a new ADR that argues *warrant*, not momentum. D2 is the
  default no.
- **We freeze chrome and the session window stays the 0098
  self-reversal.** Allowed under D2: finish a named draft (0083 S3,
  0098 owed gizmos) if a human can derive it. Not allowed: 0098b.
- **“Warranted” becomes another agent badge.** D6 is a human name
  and a whiteboard.
- **The owner does not accept.** Then the second product continues.

## Implementation log

- 2026-08-30 — drafted from an adversarial census of this tree
  (LOC by package, ADR table, `tests.yml`, `CHANGELOG.md`,
  `architecture/README.md`, GitHub PR/star counts, ADR 0100 A3,
  ADR 0079 page cap). No code. Status Proposed.
- 2026-08-30 — **red/blue re-review** (same model family; evidence
  against the tree). Disposition of D1–D4 and D6–D7 **holds**. Two
  footnotes: (1) the bridge charter binds `apeSees`, not `viewers/`
  — ADR 0014 already split that package; rewrite “charter violation”
  as “0014 did not budget a second kernel (84k > 71k).” (2) D5’s
  candidate “warranted” column is still an agent-filled list until
  D6. Studio is 11k lines; the LOC factory is `viewers/` + the 712k
  changelog, not MCP. CI-can-refuse was *not* overstated — do not
  transplant OpenSees ADR-87 D4 here.
