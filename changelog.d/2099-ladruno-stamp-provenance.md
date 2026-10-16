### ADDED — `stamp_provenance()` pairs a closed `.ladruno` with its `model.h5` through `INFO/PROVENANCE` (#2099)

ADR 0122 D3 pairs a `.ladruno` with its model by a stamp, not a sidecar
file. `apeGmsh.results.writers.stamp_provenance(path, model_h5)` writes the
stamp into the recorder's empty `INFO/PROVENANCE` group once the recorder
has closed the file. The fork's schema allows this write (`ladruno_schema_v1`
§7b.1). It stamps every `<stem>.part-<N>.ladruno` of a partitioned run.

All six attributes are lowercase, scalar, variable-length UTF-8 strings:

- `model_hash` and `fem_hash` are copied byte-equal from `/meta/lineage`.
- `stage_names` is a JSON list. It is written only when the file's
  `MODEL_STAGE` count and `KIND`s match the program's stages. Otherwise
  `StageNamesOmittedWarning` names the mismatch.
- `element_index_space` is `"fem"` only when every element id the file
  records is a FEM id that the archive pairs with itself. An element the
  bridge minted (`fem_eid` -1) leaves it absent.
- `unsupported` is a JSON list.
- `writer` defaults to `apeGmsh <version>`.

UPPERCASE attributes stay with the fork, and the stamp leaves them
untouched. Nothing outside the six attributes changes. A re-stamp with
identical values leaves the file byte-identical.

`ProvenanceStampRefused` is raised before anything is written, in any of
these cases:

- `HDF5_USE_FILE_LOCKING` turns file locking off, so an open file cannot
  be detected;
- a part is open in this process or another, or is locked;
- a part is not a `.ladruno` the reader supports;
- a part lacks `INFO/PROVENANCE`;
- a part declares more partitions than were found;
- the archive has no `/meta/lineage`;
- the parts disagree on their run or their MODEL;
- `overwrite=True` was not passed, and either a different `model_hash` is
  already stamped or an owned attribute is not a string.

The fork writes no close marker, so a recorder killed before closing its
file goes undetected once no process holds the file. The fork's
`ladruno_format.validate()` is not shipped with the build, so the stamp does
not run it; the `live-fork` lane checks stamped files with it.

The resolver (slice 5) catches this one class, warns, and skips the stamp,
which is §7b.1's "skip and warn".
