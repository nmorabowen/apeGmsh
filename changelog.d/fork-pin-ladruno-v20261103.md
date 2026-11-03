### Fork pin: `ladruno-v20261103`, with the #2621 F2 LadrunoConcrete3D tangent

`FORK_PIN` now names fork release `ladruno-v20261103` (fork commit `6050d78e1`). It still uses the `-mkl` tarball. The release adds these fork PRs to `ladruno-v20261030b`:

- nmorabowen/OpenSees#1055 (#2621 F2): the LadrunoConcrete3D chain-rule tangent; a known residual of about 1.3e-6 in stratum 1 is tracked in nmorabowen/OpenSees#1058. #1052 and #1053 add its record-replay testbed.
- nmorabowen/OpenSees#877: concrete integration (CDPM2 kernel fixes, Phase C elements and the RC law).
- nmorabowen/OpenSees#1039: LadrunoForceBeamColumn trial-state iteration and revert hygiene; WP-217 IMPL-EX clock and B-K freeze.
- nmorabowen/OpenSees#1056 (WP-219): Concrete02 stale trial stress, the `-spanLoad` bubble solve and the rebar clamp tangent.
- nmorabowen/OpenSees#1057 (WP-220): path-control phat without the prior residual; `enforceSPs` reports element failures.
- nmorabowen/OpenSees#1059 (WP-221): the `ladrunoSymmetryGuard` query and `-strictSymmetry`.
- nmorabowen/OpenSees#1060 (WP-223): `responseSpectrumPerMode -linear`.
- nmorabowen/OpenSees#1061 (WP-222): LadrunoMembrane transfer build about 15x faster, bit-identical.
- nmorabowen/OpenSees#1018 (WP-209): ElasticTimoshenkoBeam3d `localAxes`, ASDShellQ4/T3 `-pDelta` preload freeze, the LadrunoMembrane reflex-corner projection cap.
- Contact and integrator fixes #1044, #1045, #1047, #1048, #1051; EnergyBalance `-v2` columns agree across MPI ranks (#1046).

The TEMP `MKL_CBWR=COMPATIBLE` corbel step in `live-fork` stays while #2621 is open.
