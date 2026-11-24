### ADDED — ADR 0136 (Proposed): adaptive mesh convergence. No code changes.

- Design only: an error estimator on `Results` (`results.error.zz`,
  `results.error.indicator`), a size field as a list-data gmsh view folded
  into the recipe's Min field (`g.mesh.adapt.size_field`), and a driver that
  reruns a user's `build(g, size_field)` on a fresh session per iteration.
- Level A reruns the whole history per mesh. Level B restarts from an
  injected state and depends on the fork's ADR 236.
- A session is still never remeshed (ADR 0041 stays). Each iteration is a
  row in the ADR 0132 variants manifest.
