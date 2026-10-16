### DOCS — ADR 0122 (proposed): `.ladruno` is the canonical results format, with ADR 0112 D1 and ADR 0020 amendments (X, #2085)

ADR 0122 records the design the maintainer ratified on #2079. apeGmshViewer
and `Results.from_ladruno` read the fork's `.ladruno` directly. Native
`results.h5` gets no new writer and stays readable until the human-gated
native deletion. apeGmsh pairs a `.ladruno` with its `<stem>.h5` by writing
six attributes into `INFO/PROVENANCE` after close, once the fork's doc PR
allows it; there is no sidecar. The fork owns the format and apeGmsh owns
only that stamp. MPCO converts to `.ladruno` through an explicit, one-way,
bucket-level converter that keeps the `.mpco`, and `.out` transcodes through
a Python writer. The ADR also sets the order of the ten slices that follow.
Dated amendments to ADR 0112 D1 (the results artifact becomes
`<stem>.ladruno`), ADR 0020 (a stamp is not a derived copy; INV-3 holds) and
ADR 0113 (its native-results clauses lapse at the native deletion) take
effect on acceptance. No code changes in this entry.
