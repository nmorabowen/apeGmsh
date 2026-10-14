### CHANGED — apeGmshViewer analysis window: the chain↔declaration join notes say the link is not recorded by design, not "bug #1646" (program slice V4p, #1857)

The analysis window's notes ("cannot join", "declaration ambiguous") no longer point at open
bug #1646. Analysis-chain and control calls carry no declaration tag (`decl = -1` in
`/opensees/program`, `architecture/h5-schema.md`) on every file, so the notes now say the link is
not recorded for analysis calls, by design, and cite that section. Wording only: the join, the
refusals, and which notes fire are unchanged. The stages window's #1646 note (about `/sequence`) stays.
