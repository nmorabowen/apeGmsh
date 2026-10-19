### CHANGED — `ops.py` / `ops.tcl` archive the deck's `analyze` call, so `read_deck_mpco` converts an unstaged deck's `.mpco` when its kind and step count are facts (#2210)

When `ops.py(path, analyze_steps=N)` or `ops.tcl(path, analyze_steps=N)`
emits a deck, the bridge's model archive now records that call: one
`analyze` row in `/opensees/program` and `analyze_steps` (plus
`analyze_dt`) on `/opensees/analysis`. The order of the calls does not
matter:

- `ops.h5(path)` after the deck records the call directly.
- `ops.h5(path)` before the deck is rewritten by the deck, atomically and
  with one `analyze` record. If another write changed the file after
  `ops.h5` wrote it, the deck leaves it alone and warns with its path.
- If the rewrite replaces an earlier deck's call, it warns that a run of
  the earlier deck no longer matches the archive.

A bridge that never emits a deck with `analyze_steps` writes the same
archive as before. No schema version moves.

`read_deck_mpco` converts an unstaged program's `.mpco` to `.ladruno` only
when all four of these hold (ruling (ii) on #1950):

- (a) the program holds exactly one `analysis` call and one `analyze` call;
- (b) `analyze_steps` is archived;
- (c) the file holds exactly one stage of step data;
- (d) that stage records `analyze_steps` steps.

If any of them fails, it reads the `.mpco` with `Results.from_mpco` after
one `DefaultRouteWarning` that names the failed condition. An in-process
run never archives an `analyze` call, whether it uses raw `opspy.analyze`
or the bridge's `ops.analyze`, so it always takes that fallback. A bare
`convert_mpco` still refuses an unstaged program.

`emit_default`'s refusal for modal records now points at
`spec.emit_mpco(...)` and `ops.eigen(n)` instead of `ops.domain_capture`.
