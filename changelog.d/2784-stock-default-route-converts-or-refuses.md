### CHANGED — the stock default route converts a staged run's `.mpco` and refuses an unstaged one by name; `read_deck_mpco` no longer falls back to `from_mpco` (ADR 0131 D5, #2784)

Neither default-route call reads a stock `.mpco` with `Results.from_mpco`
any more. Each converts when the stage kinds are facts and otherwise
refuses with `StageKindRefused`, with no default stage kind and no
`DefaultRouteWarning`.

- **`spec.emit_default(...)` on a stock build with MPCO.** When `model_h5`
  declares `ops.stage` blocks, each with an analysis of known kind,
  `run.results()` converts `<path>.mpco` to `<path>.ladruno` beside it
  (atomically; the `.mpco` is kept) and opens that with
  `Results.from_ladruno`. A file whose stages do not pair with the declared
  ones refuses on `run.results()` and writes nothing. An unstaged program,
  a FEMData archive, or a stage with no known kind refuses at `__enter__`,
  before the loop runs. The recorder the MPCO probe made is removed and its
  new file deleted. A rerun onto an existing `<path>.ladruno` refuses
  unless you pass `overwrite=True`. `ops.run()` refuses a staged program
  until live staged emission lands (#2181), so in process this route
  refuses until then. The pip and fork routes are unchanged.
- **`read_deck_mpco`.** A stage-kind refusal (an unstaged deck whose run
  record is missing or fails a check, or a stage of unknown kind) is now
  raised again as a `StageKindRefused` that quotes it, and nothing is
  written. It used to read the `.mpco` with `from_mpco` after one warning.
- **The remedies.** Each refusal names the two that work: declare the
  stages with `ops.stage(...)`, or run the program as a deck emitted with
  `analyze_steps`, which writes the run record `read_deck_mpco` converts
  by.
