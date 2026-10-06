### CHANGED — Section documents are read from a floor, not a two-version window (ADR 0113 amendment, #1317)

`SectionDocument.open` now applies ADR 0113's compatibility floor to
`.section.json` files. A new `SECTION_DOC_FLOOR` (1.0.0) sits beside
`SECTION_DOC_VERSION`. Every document of the loader's major from the
floor up opens, so a saved 1.0 document no longer expires when the
version moves two minors. A document of a newer minor of the same major
now opens with one `SectionDocumentNewerWarning` naming both versions,
where it used to be refused. Its unknown optional keys are kept on save
and ignored by `build()`, and a value the loader cannot interpret (a
new shape kind, boolean op or material key, or a parameter a known
shape kind does not take) still refuses at load. The section builder
shows the newer-document warning on its status bar. Another
major, or a minor below the floor, refuses with a message that names
the floor. Both names are exported from `apeGmsh.sections`. ADR 0113
and ADR 0080 carry dated amendments.
