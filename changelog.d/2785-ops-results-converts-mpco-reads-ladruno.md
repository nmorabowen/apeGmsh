### CHANGED — `ops.results(path)` converts an MPCO run to `.ladruno` and reads a `recorder.Ladruno` run (#2785, ADR 0131 D11)

`ops.results(path)` reads the deck run that `ops.py(path, run=True)` or
`ops.tcl(path, run=True)` just finished. Its two history routes change:

- **`recorder.MPCO`** is no longer read with `Results.from_mpco`. The
  `.mpco` is converted to `<stem>.ladruno` beside it, through
  `read_deck_mpco`, and the `.ladruno` is read. It converts only when the
  stage kinds are facts: the program declares `ops.stage(...)` blocks, or
  the unstaged deck was run with `analyze_steps=`, which writes
  `<file>.run.json` for the converter's checks. Otherwise the read refuses
  by name and points at those two remedies. Before this change such a run
  was read with `from_mpco`. A `.ladruno` beside the `.mpco` that the
  converter did not write is never replaced; a converted one left by an
  earlier read is.
- **`recorder.Ladruno`** is read with `Results.from_ladruno`. Before this
  change it was refused as "not a history ops.results reads".

The refusals that recommended `recorder.MPCO` now recommend routes that
read: `recorder.Ladruno` on the fork build, and `recorder.MPCO` under
declared stages on stock. Two history recorders refuse, a history recorder
beside text records refuses, and so does one declared with an empty `file=`.
The emitted decks are unchanged (ADR 0115 Invariant 7).
