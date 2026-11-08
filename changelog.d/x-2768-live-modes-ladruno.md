### CHANGED — `capture_live_modes` writes a stamped `.ladruno`; a bare FEMData archive pairs by `fem_hash` (ADR 0133, program slice #2768, folds in #2233)

`apeGmsh.results.live.capture_live_modes` now writes its modes to a `.ladruno` through the Python writer
`write_modes`, as `EigenResult.to_ladruno` does: one `eigen` stage holding every mode, with the same raw
eigenvectors, eigenvalues, frequencies and periods the native `mode_<k>` stages carried. Read it with
`Results.from_ladruno(path, model_h5=...)`. A `path` with an `.h5` suffix now refuses before the solve:
pass a `.ladruno` path. `stamp_provenance` stamps a file recorded against a bare FEMData archive
(`fem.to_h5`, no `model_hash`) with `fem_hash` alone, so the plain-openseespy route of #2342 reaches
`.ladruno`; a bridge archive is still stamped with both hashes. An archive with no `fem_hash`, or an empty
hash, refuses by name, and a refused stamp leaves no file behind. A re-stamp that would change or drop a
stamped hash needs `overwrite=True`. The viewer's `.ladruno` reader accepts a `fem_hash`-only stamp, and
`pairResults` pairs it by `fem_hash`.
