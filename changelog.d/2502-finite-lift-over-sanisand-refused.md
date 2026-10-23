### FIXED — `LogStrain` / `LogStrain2D` over a SANISAND or DruckerPrager inner are refused at declaration instead of ending the fork process (#2502, #2499)

On the Ladruno fork, the `nDMaterial LogStrain` and `nDMaterial LogStrain2D`
lines ask their inner material for `getType()`. `ManzariDafalias`,
`LadrunoSANISAND`, `SAniSandMS` and `DruckerPrager` do not implement it.
The fork prints `<Class>::getType -- subclass responsibility` and exits
with rc 255. `LadrunoSANISAND` exits through its `ManzariDafalias` base.
With an in-process `analyze`, the Python session goes down too. Nothing in
the bridge refused the pair, and the `LogStrain` docstrings recommended
`DruckerPrager` as an inner.

Now `LogStrain(inner=...)` and `LogStrain2D(inner=...)` raise `ValueError`
at construction for those four inners and their subclasses. The error
names the inner, the fork's exit message and #2499. It suggests only
inners the live probe ran under a lift: `ElasticIsotropic` and
`LadrunoJ2`. The namespace route refuses the same way. A refused inner
`LogStrain` also refuses `InitDefGrad(LogStrain(<inner>))`.

The live-fork probe measured each pair at the pinned build. Pairs that do
not end the process still build:

- a `StagedStrain` between the lift and the inner;
- `InitDefGrad` directly over the inner, which the fork refuses with its
  own error;
- `ElasticIsotropic` and `LadrunoJ2`.

The refused set is `apeGmsh.opensees.material.nd.FINITE_LIFT_ABORTING_INNERS`.
Once the fork fixes #2499, the pinned probe in
`tests/opensees/unit/test_finite_wrapper_over_md_refused.py` fails and
asks for the set to be emptied.
