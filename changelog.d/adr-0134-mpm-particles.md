### ADDED — ADR 0134 (Proposed): MPM in apeGmsh (program slice M1, #2975)

Proposes how apeGmsh carries the fork's Material Point Method driver: the particle set written at emit as
`<deck stem>.particles.h5` (read by the deck's `mpm particles -file`), particle results read from `.ladruno`,
and a three.js points layer in the viewer. The ADR is a transcription of the fork's WP-232a draft
(nmorabowen/OpenSees PR #1077) and admits nothing until the maintainer accepts it. No code changes.
