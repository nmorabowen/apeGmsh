### ADDED — Python reads and writes `<stem>.workspace.h5` and `<study>.variants.json` (#2802, ADR 0132 VU9a-py)

`apeGmsh.results.user` gains the Python half of ADR 0132's two files.

- **`<stem>.workspace.h5`.** `WorkspaceFile(path)` reads `/meta`, `/sets`,
  `/plots` and `/views`, and `write_workspace(path, sets=, plots=, views=,
  remove=)` writes them, taking each set member's coordinates (a node's position,
  an element's centroid) and the identity hashes from the paired `<stem>.h5`.
  `WorkspaceFile.verdicts()` judges each set `current`, `moved`, `partial` or
  `lost`. A node set's identity is `fem_hash`; an element set's is `fem_hash`
  and `model_hash`. Only when one differs is each member looked for at its
  stored coordinates. A position that fits two items counts as unmatched, so
  two zero-length springs on one node pair read `lost` after a declaration
  swap, never `moved`.
- **The write** rebuilds the whole file through a partial per writer, verifies
  it, and compares `/meta@revision` before the replace. It never writes over a
  newer file, and it renames an unreadable one aside to
  `<stem>.workspace.h5.unreadable-<UTC>` with a `WorkspaceRenamedAside`
  warning. Rows it did not change are copied verbatim, refused rows included.
- **`<study>.variants.json`.** `read_variants(path)` and
  `write_variants(path, rows)` apply the JSON Schema and the rules it cannot
  express: unique labels and paths, the NFC name rule, confinement to the
  manifest's folder after links and junctions resolve, the target exists, and
  reserved sibling suffixes refused case-insensitively. Python refuses a newer
  minor before it validates anything.

A bad row refuses alone, with the schema's reason; a malformed envelope or a
missing table refuses the file.
