### ADDED — `results.h5` 1.2.0 stamps when transcoded element rows are provably FEM ids (program slice B5-d, #1747, closes #1746)

**Schema bump:** results zone 1.1.0 → 1.2.0. It is an additive minor
(ADR 0113): every 1.0.0 and 1.1.0 file still opens.

#1740 fixed the recorder emits, but a `.out` file does not record which
emit wrote it. A file transcoded from recorders written before the fix
therefore looked the same as one written after it. The recorder
transcoder (`Results.from_recorders`) now writes the root attribute
`element_index_space = "fem"`, but only when it can prove the element row
ids are FEM element ids, whatever apeGmsh version emitted the recorders:

- `model=` is given, and its `lineage.fem_hash` equals the spec's
  `fem_snapshot_id`;
- and either no record yields element rows, or every recorded FEM element
  id is its own OpenSees tag in the model's pairing.

Otherwise the attribute is absent and the file reads as unverified, which
is not a refusal. `NativeReader.element_index_verified` reports the
verdict, and `NativeReader.element_index_space` the raw value. `"fem"` is
the only value; `NativeWriter.stamp_element_index_space` refuses any
other. `architecture/h5-schema.md` lists the value set, which writer
writes it, what it does not certify, and what absence means.

**Fixed with it:** the `Results.from_recorders` cache key ignored `model=`.
A second call with another model on the same files and `cache_root` got
the first call's file, with that model's `/opensees` and, now, its stamp.
The key now includes a digest of the model's `lineage.fem_hash`, its
`lineage.model_hash` and its `(fem_eid, tag)` pairing.

**Limit:** element records under `element_tags="sequential"` (the bridge
default), whose FEM ids differ from their tags, stay unverified until an
emit-side manifest can prove the translation. `PARSER_VERSION` bumps to
`"1.1"`, so a cached pre-stamp transcode is made again.
