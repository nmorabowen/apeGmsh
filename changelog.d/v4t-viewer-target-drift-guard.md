### FIXED — apeGmshViewer: zone targets catch up with main, and a target-drift guard (V4t, #1821)

The app's `OPENSEES_TARGET` moves from 2.27 to 2.28, so a `model.h5` fresh
from main no longer opens under a "newer than this app" banner. Opensees
2.28 adds only a provenance `label` field to some `decl_params` rows, which
the app already reads field by field; `read.ts` says so above the constant.
The other zones the app reads were already at their writers' versions
(results is V4g's, #1810). The new `tests/test_app_target_drift.py` fails
any PR whose app targets are behind or ahead of the Python writers, naming
the constant, its file and the `h5-schema.md` history entry to read first;
`/assembly` is listed as the one zone the app does not read.
