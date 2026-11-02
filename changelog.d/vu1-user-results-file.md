### `<stem>.user.h5`: writer, reader, `rf1` and staleness (ADR 0130 slice 1)

`apeGmsh.results.user` reads and writes the `/meta`, `/combinations` and `/results`
zones of the set's user file. `write_user_file` is atomic (a killed writer leaves
the previous file whole), copies `/meta` hashes byte-equal from the model archive,
adds a combination generation on each save, and refuses to carry entries past a
hash change unless they are rewritten or `drop_stale=True` drops them.
`UserFile` refuses a bad row alone, with a reason, and a malformed zone without
closing the others. `results_fingerprint` computes `rf1`, which stamping does not
move, and `staleness` applies the table of Amendment 1 A6. The zones `user_meta`,
`combinations` and `user_results` join the schema registry. Nothing evaluates a
combination yet (slice 2), and the app does not read the file yet (slice 3).
