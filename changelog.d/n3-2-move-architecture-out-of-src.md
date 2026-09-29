### CHANGED — architecture docs and ADRs moved out of `src/` to `architecture/` (program slice N3.2)

The architecture docs and the 109 ADRs moved by `git mv` from `src/apeGmsh/opensees/architecture/` to `architecture/` at the repository root, so 2.8 MB of Markdown no longer ships inside the package (N3, #1197). The layout inside the folder is unchanged; every living path reference was rewritten. A new repo-level quirk rule `arch-path` flags any file that reappears at the old path, and `DECISIONS` in `scripts/check_quirks.py` and `scripts/adr_index.py` points at `architecture/decisions`.
