### ADDED — story forces, story drifts, pier forces and story masses from the live domain (`apeGmsh.opensees.building`)

A new module computes, in-process from the domain the bridge built, the
building tables an ETABS model reports: *Story Forces*, *Story Drifts*
(and joint drifts), *Pier Forces* and *Mass Summary by Story* / *Centers
of Mass*. It takes its module from `ops=` or `get_ops()`.

- `StoryForces(levels, *, names=, location="bottom"|"top", reference=(x, y))`
  gives `VX, VY, P, MX, MY, MZ` per story. The values come from a
  node-aligned free body: the `eleForce` (global nodal force) of the
  elements above a level, at the level's nodes. Beams, columns, shells and
  membranes all enter it, and element loads are included. A story force
  equals the resultant of the loads above the cut.
  `base_shear_from_reactions(z_base)` (`-sum(R)`) cross-checks the base.
- `StoryDrifts(levels, *, points=None, method="nearest"|"rigid", max_offset=)`
  gives drift ratios `(u_top - u_bottom) / h`. With no `points`, it covers
  every joint: each node of a level, which must have a node directly
  below it (one that does not, such as a cantilever tip, raises, named).
  `.story_max()` gives the ETABS story maximum in X and Y, with the ASCE 7
  average beside it (`AvgX`, `AvgY`: `|d_A + d_B| / 2`, the signed drifts
  at the two ends of the structure across the direction, the denominator
  of the Table 12.3-1 torsional irregularity ratio). With `points=`, it
  reads the nearest node, or the in-plane rigid fit of the level, at each
  plan point.
- `StoryDisplacements(levels, *, points=None, method=, max_offset=)` gives
  the level displacements `UX, UY` (every node on each level, or plan
  points). Its `.story_max()` gives each level's largest `|UX|`, `|UY|`
  and the same ASCE 7 signed end average: the max/avg pair of the
  torsional amplification factor `Ax`.
- `Pier(name, elements, bottom=, top=, axis2=)`, `Pier.from_pg(bridge, pg)`,
  `piers_by_story(pier, levels)` and `PierForces(piers)` give
  `P, V2, V3, T, M2, M3` at the bottom and top of each pier. These are in
  pier local axes (axis 1 up, axis 2 along the wall), with moments taken
  about the section centroid at the cut.
- `story_masses(levels)` gives `MassX, MassY, MassZ, XCM, YCM` per level.
  It counts nodal masses and the mass the elements lump to their nodes
  (read through `eleDynamicalForce` at a unit acceleration, so it covers
  every element class that has mass). `levels_from_pgs(fem, pgs)` reads
  the level elevations from horizontal physical groups.

Forces and displacements are read at the trial state. Forces come from
`eleForce`; displacements are `nodeDisp` plus `nodeResponse` `IncrDisp`
(trial minus committed). This reads the uncommitted gravity + mode state
that `responseSpectrumAnalysis -increment` leaves, and it matches
`nodeDisp` after a committed solve.

`StoryForces`, `StoryDrifts` and `PierForces` are extractors:
`extractor(ops) -> ndarray` with a fixed-length `labels` tuple, read from
the current domain state. A modal driver can evaluate one at each mode and
combine the results entry by entry; `.table(values)` reshapes the combined
vector.

The module refuses four cases rather than return a partial free body: a
cut that crosses an element without a node on it, an MP constraint that
joins the two sides of a cut, any EQ constraint in the domain
(`equationConstraint`, `LadrunoTie`: OpenSees does not report their nodes,
so `StoryForces` reads their count from one `printModel` dump at
construction; build it before `eigen`, since the dump then also writes
every eigenvector), and an
element whose `forces` vector is not one value per node DOF.

Validated live, on stock openseespy and on the fork, on a 3-storey frame +
wall building. Story forces, both top and bottom, match hand statics to
1e-10 of the largest component, and the base matches `-sum(R)`. A
cantilever wall's pier forces, in ShellMITC4 and in ASDShellQ4 at 0° and
30° in plan, match closed-form statics. Drifts match `nodeDisp`
differences. Story masses match hand sums, and their total matches
`modalProperties` `totalMass` and `totalFreeMass`. A fork-only test does
the same for a LadrunoMembrane slab, with its `rho t A` mass.
