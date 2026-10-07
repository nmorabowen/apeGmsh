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

The live fork gates read the same `BackendInfo`: the `equationConstraint`
stock-rows marker, the fork-only command gate (`LadrunoProjection`, fork
integrators, `ops.augment`), the `TenNodeTetrahedron` refusal and the
LadrunoRC `-betaC` / `-crackedNu` refusal, plus the strut-and-tie overlay's
backend tag. A fork build without the `ladrunoBuild` stamp is therefore
gated as stock, and each refusal says to rebuild the fork.
`Tet10UnverifiedBuildWarning` is removed: an unstamped fork now gets the
stock refusal instead of the warning. `ops.domain_capture(...)` and
`FootfallResult.to_results(...)` stamp `opensees_backend` / `opensees_build`
on the `results.h5` they write; `ParallelModalResult.to_native(...)` and the
recorder transcodes do not, since another binary produced their data.
