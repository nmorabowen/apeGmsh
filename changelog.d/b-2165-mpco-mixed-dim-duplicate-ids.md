### FIXED — an MPCO Gauss read that mixes 2-D and 3-D classes refuses by name, and an element recorded twice refuses on every element-bucket read of `MPCOReader` (#2165)

A Gauss read (`elements.gauss.get(...)`, catalog stress/strain or a
material-state component such as `damage`) over an `.mpco` file holding a
2-D class beside a 3-D class, for example `FourNodeQuad` beside
`FourNodeTetrahedron`, crashed in `np.concatenate` on the two
`natural_coords` widths. A `GaussSlab` holds one `(sum_GP, dim)` array, so
the read now raises `MPCOGaussDimensionMixed`, which names one bucket of each
width and the file. The coordinates are never padded or truncated. A read
narrowed to one class's elements (`ids=` or a physical group) answers as
before.

MPCO writes each element once per result. Before this fix, an element id in
two buckets, or in two rows of one bucket, came back twice from the Gauss,
material-state, section.force, fiber, layer and spring reads, with no
warning. The `localForce` and `basicForce` reads kept the first bucket's copy
and dropped the second without saying so. Each of these reads now raises
`MPCOElementInTwoBuckets`. The error names the element by its OpenSees tag,
and by its fem id too when a tag map is attached, along with the bucket
path(s) and the file. The check covers only the ids a read returns. Across
the ranked line-station tokens (`section.force` before `localForce` before
`basicForce`, and a plural token after its singular), the first source still
wins. The nodal-force counterpart is #2173.
