### CHANGED — fork build `86893bdf1` joins the C83 and ManzariDafalias-fix allowlists

Fork build `86893bdf1` is one commit past fork wp/209 (`0f2e73c2e`). Its
`ladrunoBuild` sha is now listed `True` in `stage_emit.FORK_BUILD_HAS_C83` and
`stage_emit.FORK_BUILD_HAS_MD_STATIC_FIX`. A deck that runs on it no longer
stops on the ASD shell `-pDelta` guard or the ManzariDafalias stage-flag guard.
`FORK_BUILD_HAS_MEMBRANE_WP208` already listed it. Both floor commits are
ancestors: `c83ee812e` is 20 commits behind it and `43b235bbb` 35, and neither
is ahead. That was checked in a local fork clone. The commit is on no fork
branch yet, so no GitHub compare backs it. The ManzariDafalias row rests on
ancestry alone: the live reset test has not run on this build.
