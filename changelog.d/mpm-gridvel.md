### ADDED — `ops.mpm.gridvel` and `ops.mpm.gridvelAffine` prescribe a grid velocity in an MPM deck (ADR 0134 Amendment 3)

Two new members of the `mpm` family emit the fork's `mpm gridvel`:

- `ops.mpm.gridvel(name, box, v, start=, translate=, profile=)` prescribes
  `v` (a number or `"free"` per component) on the grid nodes inside
  `box = (lo, hi)`. `translate` moves the box with the material, and
  `profile = (t0, Tr, t1)` scales `v` by a trapezoid in time.
- `ops.mpm.gridvelAffine(name, box, L, ref=)` prescribes `L (x - ref)`,
  with `L` given by its rows.

They need fork `ladruno-v20261118b` or later. That release is the first
whose explicit engine (mode F) applies `mpm gridvel`; earlier releases
accepted the line and ignored it. apeGmsh emits mode F only. The fork's
`track` option, `setAffine` and `mpm stage` are not offered: `track` and
`stage` may change in the fork's next MPM work package (WP-232l).

apeGmsh refuses each malformed case by name before the deck is written:

- a box of the wrong length, or with `lo > hi`;
- a `v` of the wrong length, or with every component free;
- `profile` together with `start`, or a trapezoid with `Tr < 0` or
  `t1 < t0 + 2 Tr`;
- a `translate` axis the box lacks, or one whose component is free;
- an `L` that is not ndm by ndm, or a `ref` of the wrong length;
- a second gridvel of one name;
- a box of another dimension than the grid's.

The live-fork lane runs a mode-F deck for each member on the pinned fork
and checks closed forms. Under `gridvel`, a body inside the box moves at
`v` and is displaced by `v T`, while a body outside the box stays at rest.
Under `gridvelAffine`, the particle velocity is `L (x - ref)`.
