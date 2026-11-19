### FIXED — `Results.from_fem` checks the fem's elements against the capture and warns on a mismatch instead of binding the wrong elements (#1408)

`Results.from_fem(fem, path)` on an `.mpco` or `.ladruno` file, and
`Results.from_mpco` / `from_ladruno` with `fem=` beside a `fem.to_h5`
archive, bound the given FEMData without any check. Such an archive has no
element tag map, so the reader takes the capture's ops tags as fem element
ids. When a hand-written deck numbered its elements differently, every
physical group still bound, and `gauss.get(pg=...)` returned the wrong
elements with no warning.

Now, when the archive has no element tag map, the capture's tags are
checked against the fem's elements. It is a mismatch when a shared id sits
on other nodes, when a tag the fem lacks sits on the nodes of a fem element
the capture does not hold under that element's own id (a renumbering), or
when no capture element is a fem element. Any other tag the fem lacks is
an element the deck added, such as a spring or a link, and is not a
mismatch. A mismatch warns
`ModelFemMismatchWarning`, naming the missing element ids, and binds the
capture's own FEMData, so no physical groups are bound and a `pg=` read
raises `KeyError`. A matching fem binds silently, as before. A fem with no
elements, or a file with no element records, is not checked. A model.h5
written by `ops.h5(...)` carries the tag map, so `fem=` beside it still
binds unchecked, and so does `kind="native"`.
