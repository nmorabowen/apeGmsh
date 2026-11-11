### FIXED — `Results.from_fem` checks the fem's elements against the capture and warns on a mismatch instead of binding the wrong elements (#1408)

`Results.from_fem(fem, path)` on an `.mpco` or `.ladruno` file, and
`Results.from_mpco` / `from_ladruno` with `fem=` beside a `fem.to_h5`
archive, bound the given FEMData without any check. Such an archive has no
element tag map, so the reader takes the capture's ops tags as fem element
ids. When a hand-written deck numbered its elements differently, every
physical group still bound, and `gauss.get(pg=...)` returned the wrong
elements with no warning.

Now, when the archive has no element tag map, the capture's elements must
be the fem's: the same id on the same nodes. This is the check the
`model_h5=` route already makes. A mismatch warns
`ModelFemMismatchWarning`, naming the missing element ids, and binds the
capture's own FEMData, so no physical groups are bound and a `pg=` read
raises `KeyError`. A matching fem binds silently, as before. A fem with no
elements, or a file with no element records, is not checked. A model.h5
written by `ops.h5(...)` carries the tag map, so `fem=` beside it still
binds unchecked, and so does `kind="native"`.
