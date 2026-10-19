### ADDED — `spec.emit_default(...)` picks the results recorder for the openseespy build; `read_deck_mpco` and `convert_mpco` turn an unstaged deck's `.mpco` into `.ladruno` only when its stage kind is a fact (#1950)

`ResolvedRecorderSpec.emit_default(path, *, model_h5, ops=None,
overwrite=False)` is a context manager around an in-process run. It reads
the bound build with `backend_info_of` and picks the recorder:

- **The Ladruno fork:** the fork's own recorder writes `<path>.ladruno`.
  `run.results()` stamps `INFO/PROVENANCE` from `model_h5` and opens the file
  with `Results.from_ladruno`. If the file refuses the stamp, it warns once
  and reads the file unstamped (§7b.1 rule 3).
- **Stock openseespy with MPCO:** writes `<path>.mpco` and reads it with
  `Results.from_mpco`, after one `DefaultRouteWarning`. It does not convert
  the file.
- **pip openseespy (the MPCO probe fails):** deletes any file the probe
  left, then records classic `.out` files in `<path>_out/` over one stage
  and reads them with `Results.from_recorders`, after one
  `DefaultRouteWarning`. No recorder stays registered after the block.

A rerun onto an existing `.ladruno` refuses unless you pass
`overwrite=True`, and modal records refuse.

`convert_mpco` now maps a program with no `ops.stage` block to one implicit
stage when two things hold: the `.mpco` has exactly one `MODEL_STAGE` with
step data, and `/opensees/program` holds exactly one `analysis` call, whose
type gives the stage's `KIND`. Several calls, no call, an unknown type or a
second `MODEL_STAGE` with step data each refuse. The in-process route never
converts, because the loop's own `analysis` calls never reach `model.h5`.

`apeGmsh.results.live.read_deck_mpco(path, *, model_h5, overwrite=False)` is
the deck-side counterpart. It converts a deck's `.mpco` beside itself when
the conversion accepts it, and otherwise reads the `.mpco` with `from_mpco`
after one warning that quotes the refusal.
