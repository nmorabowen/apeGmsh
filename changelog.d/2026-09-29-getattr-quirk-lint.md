### ADDED — quirk lint: `getattr-private` and `getattr-undefined` (program slice C1.2, #1227)

`scripts/check_quirks.py` gains two rules for `getattr`/`hasattr` probes with a
literal attribute name under `src/apeGmsh`. `getattr-private` flags a `_private`
name read off a non-`self`/`cls` object when no module of the reading top-level
subpackage defines it (the `bridge._primitives` read in `results/capture/spec.py`).
`getattr-undefined` flags a name nothing in `src/apeGmsh` defines, which can only
ever return its default (the `_sec_tags` probe that outlived 14445604). Sites that
predate the rules are held by the keyed ratchet
`scripts/quirks_getattr_baseline.txt` (`path::name`, only shrinks, a stale line is
a finding); the one-site `# apegmsh-lint: <rule>-ok <reason>` waiver works as for
every other rule. `tests/test_check_quirks.py` covers each shape to flag and to pass.
