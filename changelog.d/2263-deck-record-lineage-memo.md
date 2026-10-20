### CHANGED — a partitioned deck exported after `ops.h5()` gets its run record again, with no archive composed (#2263)

`ops.h5(path)` now remembers the `model_hash` and `fem_hash` it wrote, keyed
by the digest of the bridge's declarations. So does the automatic
`model.h5`, which goes through the same call. An `ops.py(..., analyze_steps=N)`
or `ops.tcl(..., analyze_steps=N)` export from the same declarations then
stamps its run record with that lineage and composes no archive.

- **Partitioned models.** Call `ops.h5(path)` before the export, and
  `read_deck_mpco` can convert the run again. An export with no remembered
  lineage still composes nothing. It writes no record and warns once, and the
  warning now names the remedy. Changing a declaration after `ops.h5()` means
  a new digest, so the stale lineage is never reused: the next export warns
  again until `ops.h5()` runs again.
- **Serial models** with no automatic `model.h5` (`_artifacts=False`, or no
  model name) skip the in-memory compose once `ops.h5()` has run. Otherwise
  they still compose the archive in memory, once per declaration set.

The lineage does not depend on the name, directory, cuts or sweeps of the
archive written, so any `ops.h5()` of the same declarations serves.
