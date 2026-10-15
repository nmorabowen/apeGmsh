### ADDED — the skill's removed-symbols registry and lint rule R1; false facts fixed (SK1-S2, #2006)

`skills/apegmsh/scripts/removed_symbols.json` lists every apeGmsh symbol the skill says is
removed or not shipped (`g.opensees`, `g.node_ndf`, `fem.nodes.get_ids`, the Assembly v1
verbs, the `path` plot kind, and others), each with what to use instead. Lint rule R1 in
`lint_model_script.py` flags any of them in a model script, through session and import
aliases, and refuses to run without the registry. `tests/test_removed_symbols.py` proves each
entry absent from the live code (its parent resolves and only the last name is missing, so a
typo does not count) and fails on any "not shipped" / "was removed" / "no longer" / "does not
exist" sentence in the skill whose symbol is not in the registry. Lint V3 no longer flags
`in_box`: a box selection promoted to a group (`.select().in_box().to_physical()`) is the
allowed second choice after a label (maintainer ruling on #1197), while hand-matched
coordinates (`np.isclose`, loops over `nodes.coords`) stay flagged. `workflows.md` no longer
counts mass twice: its solid, multi-part and pushover recipes drop the material `rho` and
call `ops.mass_from_model()`, so each deck carries the declared mass once (240 t for the
10 x 5 x 2 m concrete block) and runs under `-W error::UserWarning`. `fem-broker.md` drops
its freshness stamp, line citations and restated schema constants, binds raw openseespy where
it calls `node`, and stops teaching the removed `fem.elements.get(dim=3)`. The fence
baseline shrinks from 97 to 78 lines.
