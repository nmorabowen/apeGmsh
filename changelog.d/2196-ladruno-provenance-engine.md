### ADDED — a stamped `.ladruno` names the engine that solved it (#2196)

`stamp_provenance()` now writes two more `INFO/PROVENANCE` attributes, as
the fork's schema names them (`ladruno_schema_v1` §7b.1, fork PR #997). Both
are scalar, variable-length UTF-8 strings, and both are always written:

- `opensees_backend` is `"ladruno"`, `"stock"` or `"unknown"`. A file the
  fork's recorder wrote is `"ladruno"`. A file `convert_mpco` wrote is
  `"unknown"` unless the caller passes `opensees_backend="stock"` (or
  `"ladruno"`).
- `opensees_build_sha` is the 40-character hash `ladrunoBuild` returns, or
  `"unknown"`. On a recorder file it defaults to the build of the backend
  the bridge binds in the stamping process. It is `"unknown"` when that
  backend is stock, has no `ladrunoBuild`, or does not import, and on a
  converted file.

Pass `opensees_backend=` and `opensees_build_sha=` to stamp a file that
another machine or another build produced. A re-stamp keeps a known value
already on the file, so stamping it again elsewhere changes nothing;
`overwrite=True` or an explicit value replaces it. A value outside §7b.1,
a build hash without the `"ladruno"` backend, or `"stock"` on a file the
recorder wrote raises `ValueError` before anything is written.
