# Agent surface — AGENTS.md, task guides, a quirk lint

Revision 1. Not yet adversarially reviewed.

Status: **built, one PR** (branch `claude/agent-surface`, cut from
`origin/main` @ `299eb071`, 2026-09-25). No production code changes. One
test is fixed, one test line is waived, and CI gains one step. Ported from
the Ladruno fork's pilot (WP-115) and from apeWorkbench (S-agents,
apeWorkbench #234). The method ports; the rules do not, and each rule here
comes from this repo's own incidents.

## Problem

apeGmsh rarely loses a lesson. It fails to put the lesson in front of the
agent at the moment of action. `CLAUDE.md` held only generic behaviour
advice with no project facts. The repo's process lessons (merge hazards,
orphaned pushes, the editable-install trap, suite pollution) lived only in
the maintainer's agent memory, which other agents never see.

**Baseline (2026-09-25): 14 lessons recurred, across about 32 incidents.**
11 are firm (the offending act came after the lesson) and 3 are latent (the
code predated the lesson). A read-only sweep over the ADRs,
`internal_docs/`, CHANGELOG, 2,727 commits and the memory entries produced
the count. The rows this plan acts on were re-checked by hand.

| Lesson (written) | Recurred | Held by |
|---|---|---|
| ADR numbers collide (a6f664d0, 05-31; memory by 06-12) | a second 0065 (#676 → #677), renumbers 0072→0073 (#741) and 0074→0079 (#817) | rule `adr-number` |
| Tests import schema versions from `tests/fixtures/schema.py` (87d569f1, 05-22) | stale at 2.12.0 and 2.13.0 (fixed 60252205, #642), at 2.16.0 (#738); a live instance fixed here | rule `schema-literal` |
| Compose carries every FEMData stream (#707, 06-20) | embed_ties, contacts, contact_planes dropped (#912/#913, 08-11) | rule `compose-streams` |
| Merge only on verified CI; mypy stays at 0 | red-main episodes: #738, #757, #917 + #920/#921 (re-greened 1f075023), and six mypy restorations 06-25..08-19 | AGENTS.md; needs required status checks |
| `--base main`, never hand-stacked (05-23) | #858 (07-25) merged into a stacked branch. **Its LogStrain2D commit `ec6508e0` is not on main today.** | AGENTS.md; open question |
| Push-after-merge orphans (05-25) | #757's fix commits | AGENTS.md |
| Fail-loud resolution (05-16) | 45340ac3 (08-04): a silent no-op in from_h5 | bridge guide |
| Docs/skill lag the source (06-09) | 124b4e30: seven stale lines on the viewer default | `test_skill_docs_drift.py` (already built) |
| CHANGELOG sections contiguous (06-12) | mangled inside #773, #783 | ADR/docs guide |
| h5py `.get()` probes (latent), Tcl paths unquoted (latent), contact kn padding (latent) | 414ca711, be4aa1e7, #744 | guides |

## Shape

1. **`AGENTS.md` is the single source; `CLAUDE.md` is `@AGENTS.md`.** The
   old `CLAUDE.md` moved in verbatim, under "Behavioural guidelines". The
   map is new:
   - what the repo is, and the two skills and their audiences
   - where things live, with an anti-duplication clause
   - each CI lane's local command and its trap
   - how work lands: the memory-only lessons, moved into the repo
   - the task-guide table

   *Accept:* every line of the old file is present except its `# CLAUDE.md`
   title, checked line by line.
2. **Three task guides**, in `.claude/skills/apegmsh-*/SKILL.md`, matching
   what the last 300 merged PRs are made of:
   - `apegmsh-bridge-feature`: the bridge and fork adoption touch 166 of
     301 commits.
   - `apegmsh-viewer-results`: viewers and results touch about 90.
   - `apegmsh-adr-docs`: CHANGELOG is touched by 165 commits and an ADR is
     cited in 163 titles.

   Each is under 100 lines, and each item points at a file and heading.
   Every cited file, test, commit and PR was checked.
   **Coordination with #1170** (a parallel, viewers-only surface):
   `apegmsh-viewer-results` stays the single `AGENTS.md` entry for
   `results/` and `viewers/`, and routes viewer internals to #1170's
   `apegmsh-viewers-change` and `apegmsh-viewers-visual-check`. Its own
   Qt/VTK items were dropped so the two guides cannot disagree. One of
   them was wrong: #1170's probe shows `add_key_event` fires only while
   the viewport has focus, rather than VTK swallowing the key
   (`internal_docs/viewer_lessons.md` in #1170).
   *Accept:* `sync_skill.py --check` and `test_skill_docs_drift.py` still
   pass (the new directories sit beside the mirror, not in it).
3. **`scripts/check_quirks.py`**, stdlib only, about 2 s:
   - **`adr-number`**: no two `decisions/NNNN-*.md` share a number, and
     every ADR file has a link row in `decisions/README.md`. It has no
     waiver.
   - **`schema-literal`**: in `tests/` (not `tests/fixtures/schema.py`), an
     `==`/`!=` between a `N.N.N` literal and an expression naming
     `schema_version`. Stamping an old version into a fixture is an
     assignment and passes.
   - **`compose-streams`**: in the two rebuilds that must carry a whole
     model (`mesh/_compose.py` and `mesh/_femdata_h5_io.py`), every
     `ElementComposite`/`NodeComposite` call passes every `__init__`
     parameter. A `*args`/`**kwargs` call is skipped.

   The waiver is `# apegmsh-lint: <rule>-ok <reason>`; the reason is
   mandatory, and a stale waiver is itself a finding. CI runs the scan as
   the **last step of `static-gates`**, and `tests/test_check_quirks.py`
   (22 cases) runs in `suite`.
4. **Live instances:**
   - `tests/opensees/h5/test_h5_partitions.py`'s back-compat test stamped
     and asserted `"2.20.0"`, which is `OPENSEES_PRIOR_MINOR` today. It is
     a real bump hazard, hand-edited at 11 bumps since 05-22, so it is
     fixed to use the fixture constant.
   - `test_h5_emitter.py:37` echoes an explicit constructor override
     (`"1.2.3"`), which no bump can stale, so it is waived.
5. **Measurement.** The baseline is above. After about 10 work packages,
   count review and post-merge findings that match a lesson already
   written down. If the count does not drop, stop investing in guides and
   keep the lint.

## Rejected approaches

- **A fail-loud rule** (an `except` that only returns `None`/`False`/`pass`).
  It has 26 hits scoped to resolvers and composites, and 958 repo-wide. The
  incident is real (45340ac3), but classifying 26 sites is its own work
  package. It stays a guide item.
- **An h5py `.get()` rule over all readers.** 54 hits today, some of them
  dict noise; the existing guard covers `_mpco*.py`. It stays a guide item,
  and a scoped rule is an open question.
- **A CHANGELOG contiguity rule.** A blank line is missing before 56
  sections today, so the rule would be noise.
- **`compose-streams` over every composite constructor.** The MPCO and
  .ladruno readers and the `_fem_factory` builds legitimately lack streams
  (they miss 1 to 11 parameters each). The scope is the two carry-all
  rebuilds, where noise is zero and both historical incidents lived.
- **Rules for process lessons** (red-main merges, stacked bases,
  push-after-merge). They are not code patterns; required status checks,
  a repository setting, would cure the largest class.
- **A studio guide.** Studio is 59 of 301 commits, but the sweep found no
  recurring studio lesson to point at. Add one when a lesson bites twice.

## Results (2026-09-25)

**Mutation acceptance.** The lint ran on `git archive` trees.

| Tree | Rule | Expected | Got |
|---|---|---|---|
| `7b4f5716` (#676) | adr-number | flags the second 0065 | `2 ADRs numbered 0065`, and the new one is unindexed |
| `ab8638bb` (#677) | adr-number | duplicate gone | no duplicate. Older ADRs still lacked index rows then; clean today |
| `d58575ed^` | schema-literal | flags the 2.16.0 pins | `test_rebar_element_h5_roundtrip.py:83`, `test_reinforce_tie_h5_roundtrip.py:128` |
| `d58575ed` (#738) | schema-literal | those pass | pass |
| `60252205^` | schema-literal | flags the compose_inspect pin | `test_compose_facade.py:270` |
| `60252205` (#642) | schema-literal | passes | pass |
| `7b4f5716` (06-19) | compose-streams | before #707 | `_compose.py` and `_femdata_h5_io.py` both omit `reinforce_ties` |
| `7c7883b4^`, `746b513c^` | compose-streams | flags the 08-11 drop | `_compose.py:2634 omits embed_ties, contacts, contact_planes` |
| `c739f6b7` (#913) | compose-streams | passes | pass |
| this branch | all | clean | clean (one waiver, one fix) |

**Gate fails first** (one-line mutations, restored afterwards):
- dropping the waiver line reds the CI scan
- blinding `compose-streams` fails 2 of 22 self-tests
- blinding `schema-literal` fails 6
- ignoring ADR duplicates fails 1

## Open questions

- **Recover #858.** `ec6508e0` (LogStrain2D + `geom=` on the plane
  elements, 2026-07-24) merged into
  `claude/apegmsh-facet-extractor-bug-e37f5c` and never reached `main`.
- **Required status checks on `main`.** This would cure rows 3–4, the
  largest recurring class (about 9 red-main episodes).
- **Watch item:** 13082b74 (08-19) calls `restoreDockWidget` in
  `viewers/session/_host.py`. `test_dock_invariant.py` does not scan that
  file, and whether the call touches a navigation dock has not been read.
- Scoped rules for the fail-loud and h5py `.get()` lessons (see *Rejected*).
- **Hand-off from #1170:** `results/capture/spec.py` `_stable_section_tag`
  promises a "Deterministic" tag but uses the builtin `hash()` of a
  string, which is salted per process. #1170's G-HASH guard covers only
  `viewers/`. Merge order: #1170 first, since this PR's viewer-results
  guide points at its two guides.
