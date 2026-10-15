### CHANGED — the apeGmsh skill opens with a skeleton, rules and one routing table (SK1-S3, #2007)

The `apegmsh` skill's SKILL.md is now 161 lines, down from 434: a five-line mental model,
the executed skeleton, nine rules that each give their reason, a routing table that sends
each task to one reference file, and a closing step ("run `lookup` for signatures, run
`scripts/lint_model_script.py` before returning"). A human reader is sent to the docs
learning path first, then to `references/model-scripts.md`. The Studio, MCP and habitat
text moved to a new `references/studio.md`; the troubleshooting list and the version and
layout facts moved to `references/gotchas.md`, which now points at the removed-symbols
registry instead of listing removed names in prose. Both skill descriptions are under
1,024 characters (931 and 955, from 1,785 and 1,308), say what the skill is for first,
and carry a TRIGGER/SKIP list: changing apeGmsh itself goes to the AGENTS.md task guides,
raw gmsh to `gmsh-structural`, and OpenSees Tcl/C++ work to `opensees-expert` /
`opensees-ladruno`. `scripts/sync_skill.py` writes the mirror's description as a YAML
folded block (the old plain scalar held a `: ` that PyYAML refused) and refuses one over
1,024 characters; `tests/test_skill_fences.py` now enforces both size limits and the
200-line cap with no baseline exemptions, and parses both front matters with PyYAML.
