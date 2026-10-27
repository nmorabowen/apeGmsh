### FIXED — a 3-D partial or trapezoidal `beam_uniform` refuses a fork build that would drop it (#2565)

A 3-D `beam_uniform` with `span=` or `(start, end)` values emits `eleLoad -type -beamUniform wy wz wx aL bL [wyb wzb wxb]`. Fork builds older than fork commit `34df942c7` (fork WP-098 `9d3c3216d` and WP-200) give a wrong answer with no warning: `elasticBeamColumn` keeps only the start values, and `ElasticTimoshenkoBeam`, `dispBeamColumn` and `LadrunoDispBeamColumn` drop the whole load while `analyze` returns 0. The installed build 79e0623 is one of them. `forceBeamColumn` honours the load on stock OpenSees and on every fork build.

When such a load reaches any element other than a `forceBeamColumn`, the build is now checked against `stage_emit.FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD`:

- The live route raises before the load reaches OpenSees unless the running build's `ladrunoBuild` sha is listed. Its earlier refusal of stock is unchanged.
- A Tcl or py deck checks the same list and stops on any other build, stock included. A streamed partitioned deck checks inside each rank block, right before that rank's load.

A load on `forceBeamColumn` alone, a full-span uniform 3-D load and every 2-D form are unchanged. Every listed build's ancestry is checked on GitHub.
