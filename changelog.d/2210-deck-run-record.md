### ADDED — an exported deck stamps its own run, so `read_deck_mpco` converts an unstaged deck (#2210, route (c))

A deck written by `ops.py(path, analyze_steps=N)` or `ops.tcl(path,
analyze_steps=N)` now writes `<recorder file>.run.json` beside each MPCO
recorder's output.

When the deck runs:
- Before it opens the recorder, it removes any older record.
- Its last action, once the analyze loop has returned without error, is to
  close its recorders (`remove recorders`) and then write the record. A
  failed or killed run leaves no record.
- The write goes to a `.tmp` file and is then renamed.
- A partitioned deck writes one record, from rank 0, after a barrier.

The record holds 14 fields, with `schema` `apegmsh.deck_run/2`:
- the model archive's `model_hash` and `fem_hash`;
- `recorder_file`, and the closed file's `recorder_size` and
  `recorder_mtime` as the deck's runtime reports them (a list, one entry per
  part, for a partitioned deck);
- the `analysis` token, `analyze_steps`, `steps_completed` and `dt`;
- the engine's `opensees_backend` and `opensees_build_sha`;
- `writer` and `completed_iso`.

`read_deck_mpco` converts an unstaged program to `.ladruno` only when the
record passes six checks:

1. It exists and is well formed.
2. It names this `.mpco`.
3. Its hashes are the archive's.
4. The archive declares exactly one `analysis`, with the same token.
5. The file holds one stage of `steps_completed` steps, and that equals
   `analyze_steps`.
6. Every `.mpco` on disk has the size and mtime the deck recorded, so a later
   writer of the same file, such as an in-process run next to an old record,
   is caught.

In any other case it warns once, naming the failed check, and reads the file
with `Results.from_mpco`, as before. An in-process run (`ops.run()`, then
your own `opspy.analyze` loop) never writes a record, so it never converts.

A deck export never touches `model.h5` or `model_hash`. The hashes come from
the bridge's automatic `model.h5` when it holds the same declarations.
Otherwise the emitter builds the same archive in memory, so nothing is
written to a temporary file. A partitioned model is the exception: building
its archive at every export costs too much. When no cached lineage exists
for it, the export writes no record lines and warns once, and its run is
read with `Results.from_mpco`. #2263 restores conversion for this case.

The record cannot catch a deck that someone edited by hand to run a different
analysis.
