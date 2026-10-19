### CHANGED — the `ltb_shell` EOS notebook is now a lateral-torsional buckling check, asserted against the closed-form M_cr with warping (#2252)

`examples/EOS Examples/case_studies/ltb_shell/ltb_shell.ipynb` now buckles
laterally-torsionally. Before, it bent two unconnected flange strips about
the weak axis, and the twist stayed 0.

- The web and flanges are connected. Each flange is built as two
  half-width plates that share the web-flange junction line with the web,
  and the web spans flange mid-surface to flange mid-surface. The notebook
  asserts the node count of the connected section.
- The end moments act about the strong axis (±M about x), as statically
  equivalent nodal forces.
- The supports are forks: ux and uy fixed at every end node, and uz at one
  node only. The left moment no longer reacts into its own supports.
- The deck uses load control to λ = 0.6, then displacement control on
  the shear centre's lateral displacement past the bifurcation, with
  `KrylovNewton`: 22 steps, 225 iterations, about 20 s per run instead of
  about 13 minutes.
- The closed form now includes warping:
  M_cr = 75.95 kN·m, for fork supports. A Southwell plot of the computed
  path gives 76.74 kN·m (+1.0 %). The notebook asserts it within 5 % and
  asserts that the intercept recovers the 4 mm imperfection within 10 %.
  It also asserts that the twist matches the closed-form second-order twist
  within 5 % and grows past the bifurcation.
- `LTB_SHELL_MOMENT_AXIS=weak` is the mutation. With it, the end moments
  act about the weak axis and the M_cr assert fails.

The notebook runs end to end on the Ladruno fork and on stock openseespy,
and records on each build's default route as before.
