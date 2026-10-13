### FIXED — apeGmshViewer: zone targets catch up with main, and a target-drift guard (V4t, #1821)

The app's `OPENSEES_TARGET` moves from 2.27 to 2.28, so a `model.h5` fresh
from main no longer opens under a "newer than this app" banner. Opensees
2.28 adds a provenance `label` field to the `decl_params` rows of fix, mass
and Node / Element recorder declarations. The Recorders window now shows it
as the declaration's label on the targets line (`1 node · label mid`), never
as a parameter or an unset field, and shows nothing for a null label. The
other zones the app reads were already at their writers' versions (results
is V4g's, #1810). The new `tests/test_app_target_drift.py` fails any PR
whose app targets are behind or ahead of the Python writers, naming the
constant, its file and the `h5-schema.md` history entry to read first; it
reads the reader sources with comments stripped, accepts a multi-line
target, and says "could not parse" for a target it cannot read. `/assembly`
is listed as the one zone the app does not read.
