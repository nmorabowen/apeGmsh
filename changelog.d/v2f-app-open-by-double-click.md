### ADDED — apeGmshViewer opens by double-click: installer, `.h5` association, file pairing, watching, go-to-source (ADR 0112 P1, slice V2f phase 1, #1309)

`npm run dist` in `apeGmshViewer/` builds a per-user Windows installer
(electron-builder, NSIS) that registers the `.h5` association without admin
rights. The app takes the file from its argv, and a second launch forwards its
file to the running window. Opening `<stem>.h5` also opens
`<stem>.geometry.h5` and `<stem>.results.h5` (or `<stem>.mpco`) when they
exist, and opening a sibling finds its model. The open set is watched, and a
rewritten file is reported once its size and mtime have held still for
300 ms. The preload gains `onOpen`, `onFileChanged`, `goToSource` and
`requestOpen`. Go-to-source uses `$APEGMSH_EDITOR` (a `{file}`/`{line}`
template), else VS Code `code -g`, else the OS text editor, and answers
`{ok: false, reason}` for a missing file.
