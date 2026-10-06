### CHANGED — one BackendInfo decides fork vs stock; OpenSeesTarget gains mode="auto" (F2-b, #1498)

`apeGmsh.opensees._target.BackendInfo` (`kind`, `build`, `version`, `source`)
is now the resolver's one verdict on the imported OpenSees binary, from one
signal: `ops.ladrunoBuild()` returning a 40-character git sha.
`get_backend_name()`, `get_backend_build()` and
`apeSees.capabilities().has_fork` / `.build` / `.version` all derive from
`emitter.live.get_backend_info()`, so they cannot disagree. **A fork build
predating `ladrunoBuild` (fork PR #718, 2026-08-10) now reads as stock**, as
does a `ladrunoBuild()` that raises or answers `""` / `"unknown"`.

`OpenSeesTarget(mode=...)` takes `"auto"` (default: follow `BackendInfo`),
`"fork"` (same as `require_fork=True`, which still means fork) or `"stock"`.
`NativeWriter.open()` accepts `opensees_backend=` / `opensees_build=` and
writes them as optional `results.h5` root attrs (no schema bump; see
`architecture/h5-schema.md`).
