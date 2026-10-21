### FIXED — a direct `RecorderTranscoder` with no model refuses a line-stations record by name (#2331)

`RecorderTranscoder` is exported from `apeGmsh.results.transcoders`. Called
with neither `model_h5_src=` nor `pairing=`, it used to slice a
`line_stations` record's `.out` pair into equal blocks per element. A record
whose elements have different point counts could then read one element's
station as another's, with no warning, when its columns divided evenly. Take
two displacement beams of lengths 1 and 13, with Legendre 2 and Legendre 4:
the 6 columns split 3 + 3, and both blocks pass the order check.

- **The record is now refused, naming the missing input.** The message names
  `model_h5_src=` and `pairing=`, which give each element's count. A record
  with equal counts is refused too, because without a model nothing proves
  the counts equal.
- `Results.from_recorders(model=)`, `emit_default`, elastic beams read from
  `localForce`, and `fibers` records are unchanged.
