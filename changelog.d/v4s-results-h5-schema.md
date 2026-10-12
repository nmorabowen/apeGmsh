### DOCS — the `results.h5` layout is written down in `h5-schema.md` (V4s, #1706)

`architecture/h5-schema.md` has a new section, "`results.h5`: the native
results file". It documents what `NativeWriter` writes at results zone 1.1.0
(floor 1.0.0): the root attrs, the embedded `/model` and `/opensees`, the
`/meta/lineage` hashes and how to pair a results file with its `model.h5`,
stages and the time vector, partitions, and the five element categories
with their paths, dtypes and shapes. It also says how rows map to node and
element ids, what the component names are, that the file stores no units,
and what the file does not carry. apeGmshViewer reads this section as its
contract (ADR 0112 D4). No writer or reader changed.
