### CHANGED — `.ladruno` reader: hyperslab reads, part-set validation, `EMPTY_PARTITION` stitching

Follows the Ladruno fork's recorder work packages WP-163/164/165
(`ladruno_schema_v1.md` §2, `FORMAT_VERSION` 1 unchanged: the INFO rows for
`RUN_ID` / `RUN_ID_SCOPE` and the `EMPTY_PARTITION` paragraph).

- **Reads only what is asked for.** `LadrunoReader` used to load each whole
  `DATA[T × nIds × nComp]` array before slicing it. Node, energy, element,
  gauss, line-station and fiber reads now go through
  `readers/_ladruno_hyperslab.read_hyperslab`, which reads the requested
  steps, rows and columns. With the fork's ~1 MiB id-tiled chunks, one
  node's history reads one id column. Values are unchanged.
- **Stale part files are refused.** `LadrunoMultiPartitionReader` checks
  that every `<stem>.part-N.ladruno` reports the same `NUM_PARTITIONS`,
  equal to the file count, with `PARTITION_ID` covering `0..N-1`, and that
  `INFO/RUN_ID` matches across parts when `RUN_ID_SCOPE` is not
  `"process"`. A mismatch raises `ValueError` naming each file and value.
  Files written before the fork added `RUN_ID` skip that check.
- **Empty partitions stitch.** A stage marked `EMPTY_PARTITION = 1`
  (zero-length `MODEL/NODES`, no node or element results) no longer breaks
  the stage-signature check: it drops out of node, element and FEM stitching,
  and the stage's step count and time vector come from a part that holds
  results. New `LadrunoReader.is_empty_partition(stage_id)` and
  `LadrunoReader.partition_manifest()`.
- **Stages pair by order when ranks stamp them differently.** A rank-local
  topology change can give ranks different `MODEL_STAGE[<n>]` stamps for
  one logical stage, which the stitch used to refuse. Same stage names
  still pair by name. Different names with equal stage counts now pair by
  numeric stamp order (not lexicographic), with a `StageOrderMatchWarning`
  naming each file and its stages; callers see the first non-empty part's
  stage names. Different stage counts raise `ValueError` listing each
  file's stages. `EMPTY_PARTITION` stages keep their slot in the order.
