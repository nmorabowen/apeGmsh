### CHANGED — the partitioned element emit streams its rows (S3-b, #1894, ADR 0100)

`emit_element_spec_partitioned` no longer builds the `owned` list of every
owned `(eid, node_tags)` row, nor a per-row Python list of owner ranks,
before it writes the first element. Rows now stream from the element plan
to the emitter one at a time, behind a 1 B/row ownership mask; a rank
bucket that is owned whole skips the mask walk. The ADR 0044 ASDConcrete
size sweep reads a lazy view of the same rows, still before any element is
written, and still warns once with the same count and text. Decks are
byte-identical. This removes the per-element Python objects from the
element phase of a partitioned emit (#1766).
