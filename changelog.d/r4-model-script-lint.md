### ADDED — advisory model-script lint `skills/apegmsh/scripts/lint_model_script.py` (program slice R4, #1442)

The skill now ships a stdlib-only lint for agent-written model scripts. It checks the
machine-checkable Script English rules S1, S2, S3, V1, V3, V4, V5, T1, T2 and T4 from
`references/model-scripts.md` (`--rules S1,V5` selects rules; `path:line: RULE message` per
finding; exit 1 on findings). It is advisory, not a CI gate. `scripts/sync_skill.py` now also
mirrors `skills/apegmsh/scripts/*.py` byte for byte into `.claude/skills/apegmsh-helper/scripts/`.
Self-test: `tests/test_lint_model_script.py`.
