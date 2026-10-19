### ADDED — `spec.emit_default(...)` picks the results recorder for the openseespy build; `read_deck_mpco` converts a staged deck's `.mpco` to `.ladruno` and reads an unstaged one with `from_mpco` (#1950)

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

The in-process stock route never converts, because the loop's own
`analysis` calls never reach `model.h5`.

`apeGmsh.results.live.read_deck_mpco(path, *, model_h5, overwrite=False)` is
the deck-side counterpart, for the `.mpco` of the bridge's own exported deck
run out of process.

- **Staged decks convert.** It converts the `.mpco` beside itself, keeps the
  `.mpco`, and opens the `.ladruno`.
- **Unstaged decks fall back.** An unstaged program's kind is not archived as
  fact: the archive does not record the deck's `analyze` call. #2210 adds
  it. Such a deck is read with `from_mpco` after one warning.
- **Other refusals raise.** These include an archive of another model and an
  incomplete partition set. Only a refusal about stage kinds
  (`StageKindRefused`) falls back. The lineage check runs before any kind
  refusal is raised, so a foreign archive raises even when its stages would
  not pair either.

A bare `convert_mpco` behaves as before.
