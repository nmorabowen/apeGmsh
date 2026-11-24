### ADDED — `apegmsh-viewer <file>` opens a model from the command line

`apeGmshViewer/scripts/cli.mjs` is the `apegmsh-viewer` command, installed with
`npm link` (or a symlink on PATH) from a checkout. It opens `model.h5`, or any
file of its set, from any folder: a relative path means the caller's folder,
where `npm start -- x.h5` reads it against `apeGmshViewer/`. A second call opens
its file in the window already running. A missing file, a folder, a second file
or a flag is refused with exit 2. See `apeGmshViewer/README.md`.
