### CHANGED — `FORK_PIN` moves to `ladruno-v20261030b`, whose summed material return codes cannot wrap positive (#2633)

`FORK_PIN` now names fork release `ladruno-v20261030b` (fork commit `23d937bc4`). It still uses the `-mkl`
tarball. The release is `ladruno-v20261030` plus two fork PRs:

- nmorabowen/OpenSees#1043 (#2619): `Tri31`, `FourNodeQuad`, `LadrunoQuad` and `Domain::update` keep the
  most negative return code they see instead of summing them. With the refusal sentinel at −33086, a sum
  over 64,907 or more refusing points wrapped below `INT_MIN` to a positive value, every integrator read
  it as success, and the step was committed out of equilibrium. A refusal now reaches the integrator as
  itself.
- nmorabowen/OpenSees#1042 (#1037): a recorder regression test for the two-stage `MODEL_STAGE` `KIND`.
  No recorder code changes; the San Ramón file with the wrong `KIND` came from a build older than the
  fork's per-stage `KIND` (fork PR #986, pinned since `ladruno-v20261015`).

The fork build tables accept the new pin. The TEMP `MKL_CBWR=COMPATIBLE` corbel step in `live-fork`
stays while #2621 is open.
