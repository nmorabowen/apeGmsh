### FIXED — `LogStrain` / `LogStrain2D` over a SANISAND inner are refused at declaration instead of ending the fork process (#2502, #2499)

On the Ladruno fork, the `nDMaterial LogStrain` and `nDMaterial LogStrain2D`
lines ask their inner material for `getType()`. `ManzariDafalias`,
`LadrunoSANISAND` and `SAniSandMS` do not implement it, so the fork prints
`ManzariDafalias::getType -- subclass responsibility` (or the
`SAniSandMS` equivalent) and exits with rc 255. With an in-process
`analyze`, the Python session goes down too. Nothing in the bridge refused
the pair.

Now `LogStrain(inner=...)` and `LogStrain2D(inner=...)` raise `ValueError`
at construction for those three inners. The error names the inner, the
fork's exit message and #2499. The namespace route refuses the same way,
so `InitDefGrad(LogStrain(<sand>))` is refused at its inner `LogStrain`.
The live-fork probe measured each pair at the pinned build. Pairs that do
not end the process still build:

- a `StagedStrain` between the lift and the sand;
- `InitDefGrad` directly over the sand, which the fork refuses with its
  own error;
- any isotropic inner.

The refused set is `apeGmsh.opensees.material.nd.FINITE_LIFT_ABORTING_INNERS`.
Once the fork fixes #2499, the pinned probe in
`tests/opensees/unit/test_finite_wrapper_over_md_refused.py` fails and
asks for the set to be emptied.
