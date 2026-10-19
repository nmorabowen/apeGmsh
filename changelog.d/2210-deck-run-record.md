### ADDED — an exported deck stamps its own run, so `read_deck_mpco` converts an unstaged deck (#2210, route (c))

A deck written by `ops.py(path, analyze_steps=N)` or `ops.tcl(path,
analyze_steps=N)` now ends with a few lines that write
`<recorder file>.run.json` beside each MPCO recorder's output. The deck
writes the file last, and only once its analyze loop has returned without
error. A failed or killed run writes nothing. The write goes to a `.tmp` file
first and is then renamed. A partitioned deck writes one record, from rank 0,
after a barrier. The record holds the 12 fields of the ratified spec
(`schema` `apegmsh.deck_run/1`, the model archive's `model_hash` and
`fem_hash`, `recorder_file`, the `analysis` token, `analyze_steps`,
`steps_completed`, `dt`, the engine's `opensees_backend` and
`opensees_build_sha`, `writer`, and `completed_iso`).

`read_deck_mpco` converts an unstaged program to `.ladruno` only when the
record passes five checks:

1. it exists and is well formed;
2. it names this `.mpco`;
3. its hashes are the archive's;
4. the archive declares exactly one `analysis`, with the same token;
5. the file holds one stage of `steps_completed` steps, and that equals
   `analyze_steps`.

Any other case warns once, naming the failed check, and reads the file with
`Results.from_mpco`, as before. An in-process run (`ops.run()`, then your own
`opspy.analyze` loop) never writes a record, so it never converts.

`model.h5` and `model_hash` are not touched by an export. The emitter works
out the hashes by composing the same archive into a temporary file. The
record cannot catch a deck that someone edited by hand to run a different
analysis.
