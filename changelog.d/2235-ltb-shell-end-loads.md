### FIXED — the `ltb_shell` EOS notebook runs end to end: its end moments and supports are declared on the curve groups the supported way (#2235)

`examples/EOS Examples/case_studies/ltb_shell/ltb_shell.ipynb` stopped at
its load cell, because it declared the end moments with
`g.loads.surface.force_resultant_center_mass` and the end supports with
`g.displacements.surface` on `left_end` / `right_end`, which are curve
groups. Surface loads and surface displacements refuse a curve target.

The cell now declares the same loads and supports with node-level verbs:

- Each end moment becomes its least-norm statically equivalent nodal
  forces, with zero net force and the moment about the end's node centroid.
  This is the distribution the surface resultant used to compute. The
  notebook computes it from `g.physical.get_nodes` and places each force with
  `g.loads.point.force_closest(..., pg=<end>)`, which snaps at distance 0. It
  asserts the zero net force and the moment.
- The supports become `g.displacements.point(pg=<end>, dofs=...)`. For a
  homogeneous fix this gives the same records as the surface form.

The problem is unchanged: same span, section, mesh, imperfection, supports,
load magnitude and distribution. The nodal forces equal the old
resolver's least-norm forces on the same 31 nodes per end. The analysis
subprocess timeout goes from 120 s to 1800 s, because the 80-step run takes
about 8 minutes.

The notebook now says, at the top and before its results cell, that this
model is not yet a lateral-torsional buckling check. The end moment bends the
section about its weak axis, the web and flange plates share no nodes, and
the twist stays 0. The classical $M_{cr}$ it shows leaves out the warping
term. And the left end's moment reacts into its own supports. A rewrite is
carded as #2252.
