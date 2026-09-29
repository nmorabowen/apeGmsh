### ADDED — quirk lint `doc-path`: cited paths and symbols in the agent-facing docs must resolve (program slice N1.2, #1226)

`scripts/check_quirks.py` gains a repo-level `doc-path` rule beside `adr-number`. It scans
`AGENTS.md`, the task guides (`.claude/skills/apegmsh-*/SKILL.md`, not the derived
`apegmsh-helper` mirror) and the non-ADR `src/apeGmsh/opensees/architecture/*.md`, and flags
every backticked repo path (`x/y.py`, `x/y.md`, ..., optionally `:line`, `#Lnn`, `::symbol` or
`::Class.member`) that
does not resolve from the doc's folder, the repo root, `src`, `src/apeGmsh`,
`src/apeGmsh/opensees` or the architecture folder, every relative Markdown link that does
not resolve from the doc, every `::symbol` the cited `.py` file does not define at the top
level or in that class (AST), and every suffix it cannot read.
ADRs are never scanned, nor are the historical May-2026 plan docs (`phase-*.md`, `*-scope.md`,
`plan_*.md`; link N3 deletes them and the exclusion), and the rule has no waiver. Self-tests in
`tests/test_check_quirks.py`.
The first scan found 59 dead citations; the 20 in the living docs (three task guides,
`_DEFERRED.md`, `h5-schema.md`, `testing.md`, `parallel-execution.md`) are corrected here.
The P6 prototype `internal_docs/program/prototypes/docpaths.py` is deleted.
