### CHANGED — ADR 0131: apeGmsh reads only `.ladruno`; it supersedes parts of ADRs 0122 and 0115 (#2755)

ADR 0131 is accepted by the merge of #2756. It records the owner's rulings
on #2655 and on #2079 (Q1–Q5).

- apeGmshViewer, the v2 stills and the in-memory render read `.ladruno`
  only.
- `convert_mpco` stays as the only consumer of an `.mpco`.
- `Results.from_mpco` and the slab MPCO readers are deleted once the shared
  code has moved out.
- Footfall moves to `<stem>.user.h5`.
- The remaining plan is re-sliced.
- `ops.results` converts an MPCO recorder when its stage kinds are facts,
  from `ops.stage` blocks or from a deck-run record that passes the
  converter's checks. Otherwise it refuses, naming `ops.stage` or a deck
  run. A Ladruno recorder reads through `from_ladruno`.

ADRs 0122 and 0115 gain pointers to the clauses ADR 0131 supersedes. Only
ADR text changes here. The code changes in later slices.
