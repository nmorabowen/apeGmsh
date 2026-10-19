### Added

- `tests/test_lazy_import_ratchet.py`, a shrink-only ratchet in the `lock-tests` lane on the lazy share of cross-package imports (the PROGRAM.md section 8 structure KPI). It fails when the share rises above the pinned baseline and names each new function-level import by file, enclosing function and imported module. Raising the baseline needs the `human-gate` label (#1967).
