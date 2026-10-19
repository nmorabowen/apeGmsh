### ADDED — a stamped `.ladruno` names the engine that solved it (#2196)

`stamp_provenance()` now writes two more `INFO/PROVENANCE` attributes, as
the fork's schema names them (`ladruno_schema_v1` §7b.1, fork PR #997). Both
are scalar, variable-length UTF-8 strings, and both are always written:

- `opensees_backend` is `"ladruno"`, `"stock"` or `"unknown"`. A file the
  fork's recorder wrote is `"ladruno"`. A converted file (its `writer` names
  `mpco->ladruno`) is `"unknown"` unless the caller passes
  `opensees_backend="stock"` (or `"ladruno"`).
- `opensees_build_sha` is the 40-character hash `ladrunoBuild` returns, or
  `"unknown"`. The file does not carry its build, so the stamp never asks an
  OpenSees backend. The default is `"unknown"`, and the caller that ran the
  analysis passes `opensees_build_sha=get_backend_build() or "unknown"`.

A default re-stamp keeps every value already stamped, `"unknown"` included.
With `writer` left out, it also keeps a converted file's `writer`, so the
conversion mark is never lost. An explicit value, or `overwrite=True`,
replaces a stamped value. These raise `ValueError` before anything is
written:

- a value outside §7b.1;
- a build hash without the `"ladruno"` backend;
- `"stock"` on a file not marked as converted;
- a `writer` that would unmark a converted file.
