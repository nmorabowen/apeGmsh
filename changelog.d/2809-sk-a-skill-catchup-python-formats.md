### CHANGED: the skill reads results as `.ladruno` and teaches `<stem>.user.h5` (#2809)

The `apegmsh` skill caught up with the results format work. `results.md`
now opens runs with `Results.from_ladruno`, shows `convert_mpco` as the way a
stock `.mpco` reaches Python, and says that `from_native`, `to_native` and
`from_mpco` are being retired under ADR 0131 (they still work). It states how
a `.ladruno` pairs with its model archive, including ADR 0133's `fem_hash`
pairing for a bare FEMData archive. It also says that `EigenResult.results()`
and `Results.demo()` write `.ladruno`. A new section covers `<stem>.user.h5`
(ADR 0130): `write_user_file`, `results.combinations()` and `evaluate`, the
rules and the refusals, `apeSees.combination` rows archived in `model.h5`, and
the `/modes`, `/spectra` and `/entities` zones. It also covers the
apeGmshViewer `--render`, `--describe`, `--present` and `--snapshot` commands.
`SKILL.md` and `gotchas.md` now send agents to `render_stills` for stills on
Windows and keep the VTK stills for Linux. They no longer say the desktop app
opens native files.
