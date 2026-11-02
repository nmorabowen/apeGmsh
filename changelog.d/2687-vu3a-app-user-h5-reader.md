### apeGmshViewer: the set gains its `.user.h5` (ADR 0130 slice 3a)

Opening any file of a set now pairs `<stem>.user.h5` too, and the watcher reports
it like the other members. Before, the app read `run.user.h5` as the model of a
stem `run.user`. The app reads the file only when the window asks (`user:read`), so
the first frame reads none of it.

The new reader (`reader/userFile.ts`) reads `/meta`, the `/combinations` generation
in force, and the `/results` index from shapes, never `DATA`. A bad `/meta` refuses
the file, a bad zone refuses alone, and a bad row refuses alone with Python's reason.
The combination rules are a port of `_combination_rules.py`, model.h5 clashes
included. `reader/rf1.ts` computes the `rf1` results fingerprint byte for byte as
Python does, and equals every cross-language golden. The staleness table of
Amendment 1 A6 marks a re-run, another model, or a set it cannot compare as stale.
`/modes`, `/spectra` and `/entities` are listed by name and read by later slices.
Until slice VU8 reads `/entities`, an `ON_ENTITIES` user result is refused.
