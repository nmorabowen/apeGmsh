### ADDED — the skill's Python fences are resolved, and its skeleton runs in CI (program slice SK1-S1, #2005)

`tests/test_skill_fences.py` (suite lane) resolves every Python fence of `skills/apegmsh/` against the
live code through the static resolver that guarded the results docs, now shared as
`tests/_doc_resolver.py`: no missing module, attribute or keyword, with `g`, `fem`, `ops` and `results`
seeded as the session, the snapshot, the bridge and the results reader. The same test fails a fence that
does not parse, a freshness stamp, a `<file>.py:<line>` citation, a dead link, a machine path in the
description, a description over 1,024 characters and a SKILL.md over 200 lines. Today's 77 findings sit in
the shrink-only `tests/skill_fences_baseline.txt` (pinned at 77; a stale line fails) until the slices that
own each skill file clear them. `opensees-bridge.md` and `ladruno.md` stay out of the gate until #1989,
which owns them, lands. `tests/test_skill_model_scripts.py` (live lane) now also runs the SKILL.md
Core workflow skeleton under `-W error::UserWarning` and requires every banked script to pass the skill's
`lint_model_script.py`. Each gate has a control replaying the defect it exists for (`edges=`, the
`gauss.select` claim, the skeleton without `Base`, the `2.19.0` row, the 1,815-character description, a
line citation).
