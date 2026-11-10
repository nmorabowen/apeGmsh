### ADDED — `SSPquadUP(alpha_per_area=)`: McGann's stabilization per element

`ops.element.SSPquadUP(..., alpha_per_area=c)` gives each element
`alpha_e = c * A_e` from its own area, measured from the FEM snapshot when the
element is declared; `alpha=` stays the one-value form, and exactly one of the
two is required. McGann's rule is `c = 0.25 / (K + 4G/3)`. On a graded mesh a
single value per group is wrong for most cells (alpha scales with h²): a
graded u-p strip diverged in its geostatic stage with every per-group value
tried and converged per element. The h5 replay re-emits the per-element values.
