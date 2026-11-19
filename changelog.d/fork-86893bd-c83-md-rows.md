### CHANGED — fork build `86893bdf1` joins the C83 and ManzariDafalias-fix allowlists

Fork build `86893bdf1` is one commit past fork wp/209 (`0f2e73c2e`), on fork
branch `build/86893bd`. Its `ladrunoBuild` sha is now listed `True` in
`stage_emit.FORK_BUILD_HAS_C83` and `stage_emit.FORK_BUILD_HAS_MD_STATIC_FIX`,
so a deck that runs on it no longer stops on the ASD shell `-pDelta` guard or
the ManzariDafalias stage-flag guard. `FORK_BUILD_HAS_MEMBRANE_WP208` already
listed it. A GitHub compare puts both floor commits behind it: `c83ee812e` by
20 commits and `43b235bbb` by 35, with neither ahead. The live reset test
passes on a build of it: the ManzariDafalias stage flag is not reset.
