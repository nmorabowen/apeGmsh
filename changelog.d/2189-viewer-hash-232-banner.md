### CHANGED — apeGmshViewer: the results banner names the opensees schema 2.32.0 `model_hash` change, as `Results.lineage` does (#2189)

At 2.32.0 the six side-channel verbs' calls gained their declaration ids in
the hashed `/opensees/program` `decl` column, so a model with an initial
stress, an absorbing flip, `s.update_parameter` or `s.zero_velocities`
hashes differently once more. Both `program_decl` markers still read
"complete", so the viewer used to show this as a plain "written from another
model" mismatch.

The app now reads `/opensees/program` (method and `decl`) from the model and
from the results file's embedded opensees zone. It applies Python's rule
(#1911, #2198): both markers "complete", both programs call a side-channel
verb, and exactly one side's calls carry their declaration. The refusal
stays, and the reason carries Python's "likely cause: the opensees schema
2.32.0 hash change ..." sentence verbatim. The rule reads the column, not the
version stamps, so a `to_h5` rewrite is read correctly. A `.ladruno` never
names an era cause, as in Python. `tests/test_app_hash_232_drift.py` pins
the app's six verbs, the step and the phrase to `Results.py`.
