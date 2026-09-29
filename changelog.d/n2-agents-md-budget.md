### CHANGED — `AGENTS.md` cut to 150 lines or fewer, with a lock guard (program slice N2.1, #1245)

`AGENTS.md` is loaded into every agent's context; chain N (#1197) cuts it from 251 lines to
146 without dropping a rule that names a lesson. Restated prose and the behavioural
guidelines are condensed, the CI-lane table and the `land_pr.py` checklist keep every lane
and check, and the program row now points at the slice-card template
(`internal_docs/program/slice_card.md`). Two lessons move into the task guide that owns
them: the Ladruno-fork local-run baseline (the fork `.pyd`, the `integration_ladruno`
exclusion) to `apegmsh-bridge-feature`, and the offscreen-Qt `ViewerWindow` trap to
`apegmsh-viewer-results`; AGENTS.md keeps the general rule and points at each guide.
`tests/test_skill_docs_drift.py::test_agents_md_line_budget` (in `lock-tests`) fails when
the file grows past 150 lines.
