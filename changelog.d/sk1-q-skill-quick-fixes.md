### FIXED — the skill's Core workflow runs, and two false facts are gone (SK1-Q, #2004)

The `apegmsh` skill's "Core workflow" skeleton now runs end to end: it creates the `Base`
and `Tip` physical groups the bridge calls name (it used to stop with
`BridgeError: physical group 'Base' not found`), restates `g.masses` with
`ops.mass_from_model()` instead of carrying the same mass twice through a material `rho`,
names every number in a data section, and asserts both end faces resolved, so
`lint_model_script.py` reports no findings on it. The fence carries a
`<!-- reference-script: skeleton.py -->` marker for the reference-script runner. The
skill no longer names a machine path in SKILL.md, the false "`gauss.select` not shipped"
gotcha is deleted (all five results sub-composites have `.select()`), the cheatsheet's
size-field fence uses the real keywords (`distance(curves=)`, `threshold(distance_field=)`),
and the stale tracked `distributable_skills/apegmsh-helper_2026-05-30.zip` is removed.
