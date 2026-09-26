---
name: apegmsh-bridge-feature
description: >
  Read before changing apeGmsh's OpenSees bridge (`src/apeGmsh/opensees/`):
  adding or changing a typed primitive (material, section, element,
  time series, pattern, recorder, analysis option), consuming a Ladruno
  fork feature, touching the emit path (Tcl/py/live/H5), the response
  catalog, a FEMData stream, or the neutral/opensees H5 schema. A
  checklist of the traps bridge PRs have hit. For changing apeGmsh only;
  *using* the bridge in a model is the `apegmsh` skill, out of scope here.
---

# Read this before changing the OpenSees bridge

Each item points at where its lesson lives. Open that file and grep for
the quoted heading. Paths are from the repo root; `arch/` means
`src/apeGmsh/opensees/architecture/`.

## Before code

- [ ] Read `arch/charter.md`, "Principles". The 14 principles are what new
      code is judged against; P2 (primitives never touch `ops`), P7 (one
      way to declare each thing) and P12 (static typing first) are the
      ones PRs trip on.
- [ ] A new primitive follows `arch/agent-onboarding.md` "The standard
      slice", and adds exactly what `arch/testing.md` "What every PR adds"
      lists for its kind (the `ALL_*` contract lists included).
- [ ] Check the source, not the docs, for what exists today (AGENTS.md,
      "What this repo is"). If you are numbering an ADR, list the decisions
      directory on `origin/main` first; the `adr-number` quirk rule holds
      this.

## The fork (Ladruno) vs stock OpenSees

- [ ] Fork-only behaviour carries the `ladruno_fork` marker, which the root
      `tests/conftest.py` auto-skips off-fork, and a row in
      `docs/concepts/backend-capabilities.md` "The fork-only surface".
- [ ] Everything else must still run on **stock** openseespy. The
      `live-stock` lane exists because three fork/stock divergences reached
      main (#1021); its "Backend must really import" step says why a
      skipped `live` test is not a pass.
- [ ] A capability probe that can be inconclusive returns `None`, not
      `False`. See the ADR 0110 D5 row in the decisions README.

## Shared literals and streams

- [ ] The response catalog (`opensees/_response_catalog.py`) and the
      `expected` sets in `tests/test_results_element_response*.py` are
      shared literals. Two green PRs that both edit one can merge red
      (#605 + #606 → #608; AGENTS.md "How work lands").
- [ ] A new FEMData stream (a new `ElementComposite`/`NodeComposite`
      parameter) is carried by `mesh/_compose.py` **and** the model.h5
      round-trip in `mesh/_femdata_h5_io.py`. The `compose-streams` quirk
      rule fails the PR otherwise: the compose rebuild dropped streams
      twice (#707, #912/#913).
- [ ] Fail loud. A resolver or chain phase that swallows an error and
      returns `None`/`False` turns a missing tie or contact into a silent
      no-op (45340ac3, "chain-phase sessions fail loud instead of silently
      dropping defs"; `tests/test_resolution_contract.py`).

## Schema

- [ ] A schema bump follows `arch/h5-schema.md` and ADR 0023, "Bump
      cadence — locked policy" and "Per-zone read validation — two-version
      reader window". Cite the ADR's Decision section, not older prose.
- [ ] Tests read versions from `tests/fixtures/schema.py`, never a literal.
      The `schema-literal` quirk rule holds this: hard-coded versions
      turned main red at 2.12.0, 2.13.0 and 2.16.0 (#642, #738).

## Emitted decks

- [ ] Paths in an emitted Tcl deck are brace-quoted. An unquoted
      `recorder -file` with backslashes or spaces "exits 0 having written
      nothing" (be4aa1e7, #1086).
- [ ] Emit cost is gated against a committed baseline (`emit-cost-gate`,
      `tests/benchmarks/test_emit_regression_gate.py`). A 2-3x slowdown
      once reported PASS for 124 commits (#876).

## Gates before the PR

- [ ] `ruff check src/apeGmsh/opensees` (a hard gate) and
      `mypy src/apeGmsh/opensees` (baseline **0**). Six red-main episodes
      were merges past this gate; never raise the baseline.
- [ ] Warn-as-contract code is run with `-W error::<Category>`
      (AGENTS.md "Build and test").
- [ ] `python scripts/check_quirks.py`.
- [ ] The curated `suite` lane locally, since boundary tests live outside
      the package you touched. Add a CHANGELOG section per
      `internal_docs/changelog_workflow.md`.
