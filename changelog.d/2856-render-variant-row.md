### ADDED — apeGmshViewer: one `--render` call covers a study; a shot names the variant row it draws (#2856, ADR 0132 D9)

A view spec that names `files.variants` can now draw every variant of the study
in one `--render` call. A shot names its row by index and label, the way the
Variants panel names one and never by path:
`"variant": {"index": 1, "label": "Cracked"}`. A shot without the key draws
the spec's own `files.model` set, so a spec without it renders as before.

- Main reads the manifest again for every shot that names a row, so a rewrite
  between two shots is seen. It then resolves the row as the Switch door does:
  index and label must both match, the row's file must still resolve inside the
  study's folder, and every sibling of its set is confined too. The window's
  file doors (`results:*`, `geometry:open`, `model:open`, the user and
  workspace doors) then read that row's set.
- A row that does not answer refuses its shot: a label that differs only in
  case, an index the manifest lacks, a refused row, or a row that left the
  study. No PNG is written for it, `manifest.json` gives the reason, and the
  run exits 1 while the other shots still render.
- The spec is refused whole, naming the key, for a `variant` without
  `files.variants`, a negative or non-integer index, an empty label, a key
  beside `index` and `label`, or a `compare` that names the shot's own row.
- A spec that names a row reads no `.user.h5`, so none of its shots may show a
  User Combination or a user result. A row's own user file is a follow-up.
