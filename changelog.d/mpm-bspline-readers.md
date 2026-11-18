### CHANGED — both particle readers accept the fork's B-spline grid bases in `.ladruno` output and refuse any other basis by name (#3013)

The fork's release `ladruno-v20261118` (WP-232i, OpenSees#1099) adds the
quadratic and cubic B-spline bases to its MPM output. Its contract
(232a §2.4) now allows a `.ladruno` particle grid `BASIS` of `gimp`,
`linear`, `bspline2` or `bspline3`, with no format version bump. The
particle-set file is unchanged: its `BASIS` is still `gimp` or `linear`.

The two apeGmsh readers did not agree on this. The viewer refused any
basis other than `gimp` or `linear`, so it could not open a B-spline run.
The Python reader (`Results.particles`) accepted any text, unknown values
included. Now both accept the four Contract 2 values and refuse any other
value by name (C2-GRID). In the viewer the stage is refused and the rest of
the file still reads. In Python every read of that stage raises
`ParticleFormatError`. The viewer's particle-set reader still refuses a
B-spline basis in a particle-set file (C1-BASIS).

The bridge does not offer `-basis bspline2` or `bspline3` yet. It needs
version 2 of the particle-set file, which the fork has not released.
