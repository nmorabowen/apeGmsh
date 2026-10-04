### FIXED — ns default-parity test now compares the `LadrunoRC*` wrappers against `from_fc` (program slice B2-3, #1414)

`tests/opensees/unit/test_ns_wrapper_default_parity.py` resolved `ops.nDMaterial.LadrunoRCConcrete` and `LadrunoRCFiniteStrain` to the `_build_rc(cls, **kw)` helper, which has no keyword defaults, so a drifted default compared nothing. The resolver now follows `_build_rc(<Class>, ...)` to `<Class>.from_fc`, and a new test pins that those wrappers land in the compared set. Test-only; no behaviour change.

The ADR 0014 guard (`tests/test_viewers_pure_h5_consumer.py`) used to skip every relative import. It now resolves them against the importing file's package and applies the same banned-module check. That exposed two existing leaks (`viewers/diagrams/_kind_catalog.py`, `viewers/ui/_diagram_settings_tab.py`, both importing `opensees._response_catalog`); they are listed as commented exceptions under a ceiling of 2, tracked in #1421.
