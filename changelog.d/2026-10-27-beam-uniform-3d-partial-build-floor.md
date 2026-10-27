### FIXED — a 3-D partial or trapezoidal `beam_uniform` refuses a fork build that would drop it (#2565)

A 3-D `beam_uniform` with `span=` or `(start, end)` values emits `eleLoad -type -beamUniform wy wz wx aL bL [wyb wzb wxb]`. Fork builds older than fork commit `34df942c7` (fork WP-098 `9d3c3216d` and WP-200) give a wrong answer with no warning: `elasticBeamColumn` keeps only the start values, and `ElasticTimoshenkoBeam`, `dispBeamColumn` and `LadrunoDispBeamColumn` drop the whole load while `analyze` returns 0. The installed build 79e0623 is one of them.

The live route now raises before the load reaches OpenSees unless the running build's `ladrunoBuild` sha is listed in `stage_emit.FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD`. A Tcl or py deck with such a load checks the same list at its start and stops on any other build, stock included; the live route already refused stock. Every listed build's ancestry is checked on GitHub. A full-span uniform 3-D load and every 2-D form are unchanged.
