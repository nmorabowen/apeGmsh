### FIXED — two silent traps: J2Plasticity in Pa, and recombine() before generate() (#1321, #1327)

`ops.nDMaterial.J2Plasticity` now warns `J2PlasticityNoYieldWarning` when
`sig0 >= 1e8`. Upstream `J2Plasticity::plastic_integrator` seeds its residual
at `1.0` and loops on `|resid| > 1e-8*sig0`, so at that magnitude (any steel
in Pa) the return map never runs and the material stays elastic past yield
with no error, on stock OpenSees and the fork. The warning names the fix:
work in MPa, or use `LadrunoJ2`. `sigInf` does not enter the tolerance and
does not trigger it.

`g.mesh.structured.recombine()` now warns `RecombineEmptyMeshWarning` when
the model holds no 2-D elements, which is the case before `generate()`, and
points to `set_recombine(...)` / `g.mesh.recipe.structured`. A quad-only
element (`ShellMITC4`, `ShellDKGQ`, `ASDShellQ4`, `FourNodeQuad`,
`LadrunoQuad`) declared on a triangle PG now raises a `BridgeError` at tag
allocation that names the PG and the recombination fix, instead of the bare
"expected 4 node tags, got 3".
