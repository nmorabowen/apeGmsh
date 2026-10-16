### ADDED — `convert_mpco` turns an `.mpco` into a `.ladruno` (ADR 0122 D5, #2120)

`apeGmsh.results.convert_mpco(src, dst, *, model_h5, overwrite=False)`
converts an STKO `.mpco`, or every part of a `<stem>.part-<N>.mpco` set, into
the fork's `.ladruno` layout. It is explicit and one way: the `.mpco` is
opened read-only and kept, and no reader converts on open.

Bucket names and tokens, `ID`s, result values, fibre and layer tokens and the
modal groups are copied verbatim. The rest is transformed to schema v1:
`INFO` datasets become attributes (with `GENERATOR`, `FORMAT_VERSION=1`,
`STORED_PRECISION`, the partition attributes and an empty `PROVENANCE`);
element connectivity becomes a group with BASIS attributes and, where known,
`QUADRATURE/GP_PARAM`; `META` becomes `COLUMN_MAP` + `COMP_NAMES`; per-step
`DATA/STEP_k` datasets become one chunked `DATA[T x n x c]` with `STEP` and
`TIME`; section assignments gain `KIND`. Groups keep the recorder's creation
order.

`model_h5=` is required. Each stage's `KIND` comes from the program's stage
`analysis` record, a stage that holds only modes is `eigen`, and a stage count
that does not pair refuses. The converter refuses an existing target (unless
`overwrite=True`), a missing archive, a partition set with a gap or whose
parts disagree, and a source it cannot read or does not know, all with
`MPCOConversionRefused` and before anything is written. Parts are written into
a `<stem>.partial` staging directory under their final names, stamped with
`stamp_provenance` (`writer="apeGmsh mpco->ladruno <version>"`), then moved
into place; a failure removes them all. `PROVENANCE@unsupported` names what
the file lacks: `local_axes` (MPCO's local axes are dropped),
`quadrature:<group>` (a group `RESPONSE_CATALOG` cannot describe) and
`springs` (zero-length buckets are copied, but `from_ladruno` has no spring
level).

The `live-fork` lane records one model into both formats in one run: the
converted file passes the fork's own `ladruno_format.validate()` and equals the
fork's `.ladruno` under an information-equality comparator, which a swapped
column or a dropped bucket fails.
