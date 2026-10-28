### FIXED — a 3-D partial or trapezoidal `beam_uniform` refuses a build that would drop it (#2565)

A 3-D `beam_uniform` with `span=` or `(start, end)` values emits `eleLoad -type -beamUniform wy wz wx aL bL [wyb wzb wxb]`. Several builds accept it and give a wrong answer with no warning.

- **Fork builds older than fork commit `34df942c7`** (fork WP-098 `9d3c3216d` and WP-200): `elasticBeamColumn` keeps only the start values. `ElasticTimoshenkoBeam`, `dispBeamColumn` and `LadrunoDispBeamColumn` drop the whole load, and `analyze` still returns 0. The installed build 79e0623 is one of these.
- **OpenSees builds older than upstream commit `f4c791df6`**: `forceBeamColumn` ignores the trapezoid part, and `OPS_VERSION` 3.6.0 drops the load entirely.

Every route now checks the running build first:

- **On any element but `forceBeamColumn`**, the load needs a fork build listed in `stage_emit.FORK_BUILD_HAS_BEAM3D_PARTIAL_LOAD`.
- **On `forceBeamColumn` alone**, it needs one of two builds:
  - a fork build listed in `FORK_BUILD_HAS_FORCE_BEAM_PARTIAL_LOAD`;
  - stock whose `version()` reports 3.7.0 or later. An unreadable version refuses. openseespy 3.6.0.3 carries the fix but reports 3.6.0, the same as the 3.6.0 builds without it, so it refuses too.
- **The live route** raises before the load reaches OpenSees. Its earlier refusal of stock is unchanged.
- **A Tcl or py deck** checks at its start. A streamed partitioned deck checks inside each rank block, right before that rank's load.

A full-span uniform 3-D load and every 2-D form are unchanged. Every listed build's ancestry is checked on GitHub.
