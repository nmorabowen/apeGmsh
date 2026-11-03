### CHANGED — ADR 0131 (Proposed): apeGmsh reads only `.ladruno`; it supersedes parts of ADRs 0122 and 0115 (#2755)

ADR 0131 records the owner's rulings on #2655 and on #2079 (Q1–Q5).
apeGmshViewer, the v2 stills and the in-memory render read `.ladruno` only.
`convert_mpco` stays as the only consumer of an `.mpco`. `Results.from_mpco`
and the slab MPCO readers are to be deleted once the shared code has moved
out. Footfall moves to `<stem>.user.h5`. The ADR also re-slices the
remaining plan. For `ops.results`, an MPCO recorder converts when the stages
are declared as facts and is refused by name otherwise, and a Ladruno
recorder reads through `from_ladruno`. ADRs 0122 and 0115 gain pointers to
the clauses ADR 0131 supersedes. Only ADR text changes here. The code
changes in later slices.
